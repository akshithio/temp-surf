"""Selective prediction (abstention) over FINALIZED per-sample predictions.

Pure post-processing: this module reads prediction rows that a completed run already wrote and never
fits a probe, touches a split, or influences probe execution. It answers the operational question the
headline accuracy cannot -- "if the model may abstain on its least confident cases, how much does the
error on what it *does* answer improve, and at what coverage cost?"

The risk--coverage curve is built by sorting examples by a confidence score (descending) and sweeping
the abstention threshold. Two conventions matter and are made explicit rather than assumed:

* **Ties are resolved as whole blocks.** Examples sharing a confidence value cannot be separated by any
  threshold, so a curve that splits them reports a coverage the deployed system could never realize.
  Every reported point therefore sits on a tie boundary.
* **Risk is conditional on answering.** ``risk(c)`` is the error rate among the covered examples only,
  which is what an operator experiences -- not error over the whole population.

``aurc`` (area under the risk--coverage curve, lower is better) summarizes the whole curve in one
number; a perfectly ordered confidence signal drives it toward the irreducible error, while an
uninformative one leaves it at the base error rate.
"""

from __future__ import annotations

from typing import Any

import numpy as np

#: Coverage levels reported alongside the full curve. Chosen once here so every caller and table uses
#: the same grid rather than re-literaling its own.
COVERAGE_GRID: tuple[float, ...] = (0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0)


def risk_coverage_curve(
    correct: np.ndarray, confidence: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """The (coverage, risk) curve, one point per distinct confidence level (tie blocks kept intact).

    ``correct`` is a boolean/0-1 array of per-example correctness, ``confidence`` the score used to
    rank them (higher = more confident). Returns ascending coverage and the conditional error rate at
    each. An empty input yields empty arrays rather than raising -- an absent evaluation population is
    the caller's to report, not this function's to guess at.
    """
    correct = np.asarray(correct).astype(bool)
    confidence = np.asarray(confidence, dtype=float)
    if correct.size != confidence.size:
        raise ValueError(f"correct/confidence length mismatch: {correct.size} vs {confidence.size}")
    n = int(correct.size)
    if n == 0:
        return np.empty(0), np.empty(0)
    if not np.isfinite(confidence).all():
        raise ValueError("confidence contains non-finite values -- refuse to rank on NaN/inf")

    order = np.argsort(-confidence, kind="stable")
    conf_sorted = confidence[order]
    err_sorted = (~correct[order]).astype(np.float64)
    cum_err = np.cumsum(err_sorted)

    # Cut only where the confidence actually changes: the last index of each tie block.
    is_block_end = np.empty(n, dtype=bool)
    is_block_end[:-1] = conf_sorted[:-1] != conf_sorted[1:]
    is_block_end[-1] = True
    idx = np.flatnonzero(is_block_end)

    k = (idx + 1).astype(np.float64)          # number of covered examples at each cut
    coverage = k / float(n)
    risk = cum_err[idx] / k
    return coverage, risk


def aurc(coverage: np.ndarray, risk: np.ndarray) -> float:
    """Area under the risk--coverage curve by the trapezoid rule (lower is better).

    Undefined for a single point (no interval to integrate), which is reported as NaN rather than
    silently collapsed to that point's risk.
    """
    coverage = np.asarray(coverage, dtype=float)
    risk = np.asarray(risk, dtype=float)
    if coverage.size < 2:
        return float("nan")
    # explicit trapezoid: np.trapz/np.trapezoid changed name across numpy majors and this
    # module must give byte-identical numbers on every box in the fleet.
    widths = np.diff(coverage)
    heights = 0.5 * (risk[1:] + risk[:-1])
    return float(np.sum(widths * heights) / (coverage[-1] - coverage[0]))


def risk_at_coverage(coverage: np.ndarray, risk: np.ndarray, target: float) -> float:
    """Risk at the largest achievable coverage that does not exceed ``target``.

    Rounding DOWN (rather than interpolating) keeps every reported number realizable by an actual
    threshold: an interpolated point corresponds to abstaining on part of a tie block, which no
    deployed rule can do.
    """
    coverage = np.asarray(coverage, dtype=float)
    risk = np.asarray(risk, dtype=float)
    ok = coverage <= float(target) + 1e-12
    if not ok.any():
        return float("nan")
    return float(risk[np.flatnonzero(ok)[-1]])


def summarize(
    correct: np.ndarray, confidence: np.ndarray, *, grid: tuple[float, ...] = COVERAGE_GRID,
) -> dict[str, Any]:
    """One selective-prediction record: base error, AURC, and risk at each grid coverage."""
    coverage, risk = risk_coverage_curve(correct, confidence)
    out: dict[str, Any] = {
        "n": int(np.asarray(correct).size),
        "base_error": float(1.0 - np.mean(np.asarray(correct).astype(bool))) if np.asarray(correct).size else float("nan"),
        "aurc": aurc(coverage, risk),
        "n_curve_points": int(coverage.size),
    }
    for c in grid:
        out[f"risk_at_coverage_{int(round(c * 100))}"] = risk_at_coverage(coverage, risk, c)
    return out


def summarize_rows(
    rows: list[dict[str, Any]], *, confidence_field: str = "confidence",
    correct_field: str = "correct", group_fields: tuple[str, ...] = ("benchmark", "model", "holdout", "seed"),
) -> list[dict[str, Any]]:
    """Group finalized prediction rows and summarize each group.

    Rows missing either field are skipped and COUNTED (``n_skipped``) rather than dropped silently, so
    a partially-written prediction file can never masquerade as a clean selective-prediction result.
    """
    groups: dict[tuple, list[dict[str, Any]]] = {}
    skipped = 0
    for r in rows:
        if r.get(confidence_field) is None or r.get(correct_field) is None:
            skipped += 1
            continue
        groups.setdefault(tuple(r.get(f) for f in group_fields), []).append(r)
    out: list[dict[str, Any]] = []
    for key in sorted(groups, key=lambda k: tuple(str(x) for x in k)):
        g = groups[key]
        rec = summarize(
            np.asarray([bool(r[correct_field]) for r in g]),
            np.asarray([float(r[confidence_field]) for r in g], dtype=float),
        )
        out.append({**dict(zip(group_fields, key, strict=True)), **rec, "n_skipped": skipped})
    return out
