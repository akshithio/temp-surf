"""Regional screening over FINALIZED geographic source-only predictions and embeddings.

Pure post-processing: reads what a completed run already wrote and never fits a probe, touches a
split, or influences probe execution.

The question is operational rather than descriptive. A geographic evaluation reports how a
source-trained model scores on each held-out region *after* labelling that region. A deployer does not
have those labels -- they have the model, the source data, and an unlabelled candidate region, and they
need to decide where the model can be trusted. Screening asks whether a LABEL-FREE signal (how far a
region sits from the source distribution in embedding space) predicts the labelled degradation we
measured. If it does, the signal can triage regions before anyone annotates them; if it does not, that
is itself a finding worth stating plainly rather than leaving implicit.

Two distances are computed because they fail differently. Centroid distance is cheap and stable but
blind to shape: a region straddling the source cloud can sit near its centre while overlapping almost
nothing. Mean-nearest-source distance is sensitive to exactly that, at a higher cost. Reporting both,
and their separate correlations with realized score, keeps a single convenient number from quietly
becoming the claim.
"""

from __future__ import annotations

from typing import Any

import numpy as np

#: Rank correlation is the headline: screening only needs the ORDERING of regions to be right, and a
#: Pearson coefficient would let one extreme region dominate the verdict.
MIN_REGIONS_FOR_CORRELATION = 3


def _centroid_distance(source_emb: np.ndarray, region_emb: np.ndarray) -> float:
    """Euclidean distance between the region centroid and the source centroid."""
    return float(np.linalg.norm(region_emb.mean(axis=0) - source_emb.mean(axis=0)))


def _mean_nearest_source_distance(
    source_emb: np.ndarray, region_emb: np.ndarray, *, sample_cap: int = 2000, seed: int = 0,
) -> float:
    """Mean over region points of the distance to the NEAREST source point.

    Both sides are subsampled to ``sample_cap`` with a fixed seed: the full pairwise matrix is
    quadratic and this statistic is an expectation, so a deterministic subsample estimates it at a
    fraction of the cost while staying reproducible across machines.
    """
    rng = np.random.default_rng(seed)

    def _sub(a: np.ndarray) -> np.ndarray:
        if len(a) <= sample_cap:
            return a
        return a[rng.choice(len(a), sample_cap, replace=False)]

    s, r = _sub(source_emb), _sub(region_emb)
    # chunked to bound peak memory on wide embeddings
    total, n = 0.0, 0
    for start in range(0, len(r), 256):
        block = r[start:start + 256]
        d = np.linalg.norm(block[:, None, :] - s[None, :, :], axis=-1)
        total += float(d.min(axis=1).sum())
        n += len(block)
    return total / max(1, n)


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman rank correlation with average ranks for ties (no SciPy dependency here)."""
    def _rank(v: np.ndarray) -> np.ndarray:
        order = np.argsort(v, kind="stable")
        ranks = np.empty(len(v), dtype=float)
        ranks[order] = np.arange(len(v), dtype=float)
        # average tied ranks so a plateau cannot fake an ordering
        _, inv, counts = np.unique(v, return_inverse=True, return_counts=True)
        sums = np.zeros(len(counts))
        np.add.at(sums, inv, ranks)
        return (sums / counts)[inv]

    if len(a) < 2:
        return float("nan")
    ra, rb = _rank(np.asarray(a, dtype=float)), _rank(np.asarray(b, dtype=float))
    sa, sb = ra.std(), rb.std()
    if sa == 0 or sb == 0:
        return float("nan")
    return float(((ra - ra.mean()) * (rb - rb.mean())).mean() / (sa * sb))


def screen_regions(
    *, source_emb: np.ndarray, region_emb: dict[str, np.ndarray],
    realized_score: dict[str, float] | None = None, seed: int = 0,
) -> list[dict[str, Any]]:
    """Per-region label-free distance signals, ranked most- to least-distant.

    ``realized_score`` (the labelled geographic result, when available) is attached purely so the
    screening signal can be VALIDATED against it -- it is never an input to the ranking, which must
    stay computable without any target labels.
    """
    source_emb = np.asarray(source_emb, dtype=float)
    if source_emb.ndim != 2 or len(source_emb) == 0:
        raise ValueError("source_emb must be a non-empty 2-D embedding matrix")
    out: list[dict[str, Any]] = []
    for name in sorted(region_emb):
        emb = np.asarray(region_emb[name], dtype=float)
        if emb.ndim != 2 or len(emb) == 0:
            raise ValueError(f"region {name!r}: empty or non-2-D embedding matrix")
        if emb.shape[1] != source_emb.shape[1]:
            raise ValueError(
                f"region {name!r}: embedding width {emb.shape[1]} != source width {source_emb.shape[1]}"
            )
        rec: dict[str, Any] = {
            "region": name,
            "n_region": int(len(emb)),
            "centroid_distance": _centroid_distance(source_emb, emb),
            "mean_nearest_source_distance": _mean_nearest_source_distance(source_emb, emb, seed=seed),
        }
        if realized_score is not None and name in realized_score:
            rec["realized_score"] = float(realized_score[name])
        out.append(rec)
    out.sort(key=lambda r: -r["mean_nearest_source_distance"])
    for rank, rec in enumerate(out):
        rec["distance_rank"] = rank
    return out


def screening_validity(screened: list[dict[str, Any]]) -> dict[str, Any]:
    """How well each label-free distance ORDERS regions by their realized score.

    Correlations are negative when the signal works: more distant regions should score worse. Fewer
    than ``MIN_REGIONS_FOR_CORRELATION`` regions yields NaN and ``usable=False`` rather than a
    coefficient over two points, which would be noise presented as evidence.
    """
    scored = [r for r in screened if "realized_score" in r]
    n = len(scored)
    base: dict[str, Any] = {"n_regions": n, "usable": n >= MIN_REGIONS_FOR_CORRELATION}
    if not base["usable"]:
        return {**base, "spearman_centroid": float("nan"), "spearman_nearest": float("nan")}
    score = np.asarray([r["realized_score"] for r in scored], dtype=float)
    return {
        **base,
        "spearman_centroid": _spearman(
            np.asarray([r["centroid_distance"] for r in scored], dtype=float), score
        ),
        "spearman_nearest": _spearman(
            np.asarray([r["mean_nearest_source_distance"] for r in scored], dtype=float), score
        ),
    }
