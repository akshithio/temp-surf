"""Result IO and summary aggregation for experiment runners.

Kept dependency-free (numpy only) so it never imports the eval/method modules:
the caller passes the metric list to summarize, since this project has three benchmark
families (binary, multiclass, regression) with different metrics.
"""

from __future__ import annotations

import csv
import json
import os
import shutil
from pathlib import Path
from typing import Any

import numpy as np

# Metrics where LOWER is better (error/calibration metrics) -- the worst region is their MAX,
# not their min, and the deployment gap reads in the opposite direction.
_LOWER_BETTER = frozenset({"brier", "nll", "ece"})


def hf_download_to(repo_id: str, filename: str, dest: Path, revision: str | None = None) -> Path:
    """Download ``filename`` from ``repo_id`` (pinned to ``revision`` if given) into
    ``dest.parent`` and return ``dest``. Pinning an immutable commit revision makes the
    downloaded bytes reproducible (a moved branch tag could otherwise serve different weights)."""
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise ImportError(
            "huggingface-hub is required to download model weights. Run `uv pip install -e .` "
            "from the project env or set an explicit local weights path."
        ) from exc

    dest = Path(dest).expanduser()
    dest.parent.mkdir(parents=True, exist_ok=True)
    downloaded = Path(
        hf_hub_download(repo_id=repo_id, filename=filename, local_dir=str(dest.parent), revision=revision)
    )
    if downloaded.resolve() != dest.resolve():
        shutil.copy2(downloaded, dest)
    return dest


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write rows to CSV using the union of all keys (rows may be heterogeneous)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read a JSON-lines result log (one row dict per line). Skips blank/degraded lines."""
    path = Path(path)
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue  # tolerate a half-written final line from a crash
    return rows


def append_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    """Append rows to a JSON-lines log (creates parents). Used for crash-resumable results.

    The whole batch is serialized first and written in a single ``write`` + ``fsync`` so a
    crash cannot persist a partial subset of one caller's rows (which would make a resumed run
    treat an interrupted sweep as finished). Pass one logical unit (one cell/family) per call
    so that batch is the atomic resume granularity.
    """
    if not rows:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(json.dumps(row, default=_json_default) + "\n" for row in rows)
    with path.open("a", encoding="utf-8") as f:
        f.write(payload)
        f.flush()
        os.fsync(f.fileno())


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)


def read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text())


def write_json(path: Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True))
    tmp.replace(path)


def summarize_rows(
    rows: list[dict[str, Any]],
    keys: list[str],
    metrics: list[str],
) -> list[dict[str, Any]]:
    """Group rows by ``keys`` and report mean/std of each metric (over seeds/holdouts).

    Adds ``n_rows`` always, ``n_seeds`` / ``n_holdouts`` when those columns exist,
    and aggregates probe-convergence bookkeeping when present.
    """
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        if not all(key in row for key in keys):
            continue
        grouped.setdefault(tuple(row[key] for key in keys), []).append(row)
    out: list[dict[str, Any]] = []
    for key_values, vals in sorted(grouped.items(), key=lambda kv: tuple(str(x) for x in kv[0])):
        row: dict[str, Any] = dict(zip(keys, key_values, strict=False))
        for metric in metrics:
            present = [float(v[metric]) for v in vals if metric in v and v[metric] is not None]
            finite = [x for x in present if np.isfinite(x)]
            row[f"mean_{metric}"] = float(np.mean(finite)) if finite else float("nan")
            row[f"std_{metric}"] = float(np.std(finite)) if finite else float("nan")
        row["n_rows"] = len(vals)
        if "seed" in vals[0]:
            row["n_seeds"] = len({v["seed"] for v in vals})
        if "holdout" in vals[0]:
            row["n_holdouts"] = len({str(v["holdout"]) for v in vals})
        if "probe_converged" in vals[0]:
            row["all_probes_converged"] = int(all(int(v["probe_converged"]) == 1 for v in vals))
        if "probe_convergence_warnings" in vals[0]:
            row["total_probe_convergence_warnings"] = int(
                sum(int(v["probe_convergence_warnings"]) for v in vals)
            )
        out.append(row)
    return out


def _close(a: Any, b: float) -> bool:
    try:
        return abs(float(a) - b) < 1e-9
    except (TypeError, ValueError):
        return False


# --- chance / no-skill floor per metric (for the floor-normalized drop) ----------------
def _chance(metric: str, pos_rate: float | None, n_classes: float | None, majority_rate: float | None):
    """The no-skill baseline a metric would score, or None if not well-defined.

    Used as the floor in ``floor_norm_drop = (id - ood) / (id - chance)`` -- the fraction
    of *above-chance* performance erased by going OOD. Only defined for higher-is-better
    metrics with a clean no-skill reference; error metrics (brier/nll/ece) and macro/weighted
    F1 return None.
    """
    if metric in ("auc", "macro_auc"):
        return 0.5
    if metric in ("balanced_accuracy", "calibrated_balanced_accuracy"):
        return 1.0 / max(int(n_classes or 2), 2)
    if metric == "accuracy":
        return majority_rate
    if metric in ("f1", "calibrated_f1"):
        return (2.0 * pos_rate / (1.0 + pos_rate)) if (pos_rate and pos_rate > 0) else None  # always-positive no-skill F1
    return None


# --- numpy binary-metric recomputation (for per-sample bootstrap; keeps ioutils sklearn-free) ---
def _f1(y: np.ndarray, pred: np.ndarray) -> float:
    tp = float(((pred == 1) & (y == 1)).sum())
    fp = float(((pred == 1) & (y == 0)).sum())
    fn = float(((pred == 0) & (y == 1)).sum())
    denom = 2 * tp + fp + fn
    return 2 * tp / denom if denom > 0 else 0.0


def _balanced_acc(y: np.ndarray, pred: np.ndarray) -> float:
    tp = ((pred == 1) & (y == 1)).sum()
    fn = ((pred == 0) & (y == 1)).sum()
    tn = ((pred == 0) & (y == 0)).sum()
    fp = ((pred == 1) & (y == 0)).sum()
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    tnr = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    return 0.5 * (tpr + tnr)


def _rankdata_avg(a: np.ndarray) -> np.ndarray:
    """Ranks (1-indexed) with ties resolved by their MIDRANK (average rank of the tied group).

    This is what the Mann-Whitney form of AUC requires; plain argsort-of-argsort breaks ties
    by arbitrary sort order, which biases AUC for discrete-probability predictors like KNN.
    """
    order = np.argsort(a, kind="mergesort")
    a_sorted = a[order]
    ranks_sorted = np.empty(len(a), dtype=float)
    i = 0
    n = len(a)
    while i < n:
        j = i
        while j + 1 < n and a_sorted[j + 1] == a_sorted[i]:
            j += 1
        ranks_sorted[i : j + 1] = (i + j) / 2.0 + 1.0  # average of 1-indexed positions i..j
        i = j + 1
    out = np.empty(n, dtype=float)
    out[order] = ranks_sorted
    return out


def _auc(y: np.ndarray, p: np.ndarray) -> float:
    n_pos = int((y == 1).sum())
    n_neg = len(y) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = _rankdata_avg(p)  # midrank-corrected, so tied (e.g. KNN) probabilities score correctly
    return float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def _ece_np(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> float:
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.digitize(p, edges[1:-1])
    e = 0.0
    for b in range(n_bins):
        m = idx == b
        if m.any():
            e += m.sum() * abs(y[m].mean() - p[m].mean())
    return float(e / len(y))


def _bin_metric(name: str, y, prob, pred_def, pred_cal) -> float:
    if name == "f1":
        return _f1(y, pred_def)
    if name == "calibrated_f1":
        return _f1(y, pred_cal)
    if name == "balanced_accuracy":
        return _balanced_acc(y, pred_def)
    if name == "calibrated_balanced_accuracy":
        return _balanced_acc(y, pred_cal)
    if name == "auc":
        return _auc(y, prob)
    if name == "brier":
        return float(np.mean((prob - y) ** 2))
    if name == "nll":
        pc = np.clip(prob, 1e-12, 1 - 1e-12)
        return float(-np.mean(y * np.log(pc) + (1 - y) * np.log(1 - pc)))
    if name == "ece":
        return _ece_np(y, prob)
    return float("nan")


def _by_seed_arrays(preds: list[dict[str, Any]] | None) -> dict[Any, tuple]:
    """Group per-sample prediction dicts by SEED into per-seed
    ``(y, prob, pred_default, pred_calibrated, region_cluster)`` arrays.

    No cross-seed collapsing: each seed has its OWN test set (random_id and target membership
    both vary with seed), so collapsing by ``(holdout, sample_id)`` would average variable-
    membership predictions. Exact ``(holdout, sample_id)`` duplicates WITHIN a seed (crash-
    recovery re-appends) are de-duplicated so no observation is double-counted.
    """
    out: dict[Any, tuple] = {}
    if not preds:
        return out
    by_seed: dict[Any, dict[tuple, dict[str, Any]]] = {}
    for p in preds:
        if "prob" not in p or "pred_default" not in p or "pred_calibrated" not in p:
            continue
        by_seed.setdefault(p.get("seed"), {})[(p.get("holdout"), p.get("sample_id"))] = p
    for seed, uniq in by_seed.items():
        ps = list(uniq.values())
        out[seed] = (
            np.array([p["y_true"] for p in ps], dtype=np.float64),
            np.array([p["prob"] for p in ps], dtype=np.float64),
            np.array([p["pred_default"] for p in ps], dtype=np.int64),
            np.array([p["pred_calibrated"] for p in ps], dtype=np.int64),
            np.array([str(p.get("holdout", p.get("group", ""))) for p in ps], dtype=object),
        )
    return out


def _cluster_resample(rng: np.random.Generator, clusters: np.ndarray) -> np.ndarray:
    """One two-stage (nested) bootstrap draw: resample whole clusters (regions) with replacement,
    AND resample observations within each drawn cluster with replacement. With a single cluster
    (e.g. the ID anchor) this reduces to a plain per-sample resample."""
    uniq = np.unique(clusters)
    n = len(clusters)
    if len(uniq) <= 1:
        return rng.integers(0, n, n)
    idx_by_cluster = {c: np.flatnonzero(clusters == c) for c in uniq}
    out = []
    for c in rng.choice(uniq, size=len(uniq), replace=True):
        members = idx_by_cluster[c]
        out.append(members[rng.integers(0, len(members), len(members))])  # resample WITHIN the region
    return np.concatenate(out)


def _sample_delta_ci(metric, id_preds, ood_preds, n_boot, rng):
    """Hierarchical seed -> region -> sample bootstrap of the delta, WITHOUT collapsing seeds.

    Each seed is a complete, independent evaluation (its own source-trained model and its own
    test set). Point estimate = mean over seeds of the per-seed (id - ood) metric (matching the
    seed-mean ``delta``). Each bootstrap draw resamples seeds with replacement; within each drawn
    seed it resamples OOD by region/sample cluster and ID by sample, then averages the per-seed
    deltas. Returns ``(ci_lo, ci_hi, id_pt, ood_pt)``; NaN for non-binary metrics.
    """
    id_by_seed = _by_seed_arrays(id_preds)
    ood_by_seed = _by_seed_arrays(ood_preds)
    seeds = sorted(set(id_by_seed) & set(ood_by_seed), key=lambda s: (s is None, s))
    per_id, per_ood = [], []
    for s in seeds:
        ip = _bin_metric(metric, *id_by_seed[s][:4])
        op = _bin_metric(metric, *ood_by_seed[s][:4])
        if not (np.isnan(ip) or np.isnan(op)):
            per_id.append(ip)
            per_ood.append(op)
    if not per_id:
        return float("nan"), float("nan"), float("nan"), float("nan")
    id_pt, ood_pt = float(np.mean(per_id)), float(np.mean(per_ood))
    if not n_boot:
        return float("nan"), float("nan"), id_pt, ood_pt
    seeds_arr = np.array(seeds, dtype=object)
    deltas = []
    for _ in range(n_boot):
        draw = []
        for s in seeds_arr[rng.integers(0, len(seeds_arr), len(seeds_arr))]:
            yi, pi, di, ci, gi = id_by_seed[s]
            yo, po, do, co, go = ood_by_seed[s]
            ii, oo = _cluster_resample(rng, gi), _cluster_resample(rng, go)
            ip = _bin_metric(metric, yi[ii], pi[ii], di[ii], ci[ii])
            op = _bin_metric(metric, yo[oo], po[oo], do[oo], co[oo])
            if not (np.isnan(ip) or np.isnan(op)):
                draw.append(ip - op)
        if draw:
            deltas.append(float(np.mean(draw)))
    if not deltas:
        return float("nan"), float("nan"), id_pt, ood_pt
    return float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5)), id_pt, ood_pt


def compute_deltas(
    rows: list[dict[str, Any]],
    metrics: list[str],
    *,
    predictions: list[dict[str, Any]] | None = None,
    id_source_budget: float | int = 1.0,
    n_boot: int = 2000,
    n_boot_sample: int = 1000,
    seed: int = 0,
) -> list[dict[str, Any]]:
    """ID→OOD drop per metric and per (model, benchmark, method).

    ``id``  = metric on the configured random-split in-distribution source-budget anchor.
    ``ood`` = metric on the curated geographic holdout (geographic_ood, target budget 0).
    ``target_id`` = metric on the *target-ID upper bound* (geographic_ood, target budget -1):
      train on 80 % of the target region (no source), test on remaining 20 %.  This
      separates "this region is intrinsically harder" from "transfer to this region is
      harder".  Only reported when the underlying row exists.

    Reports, per row:
      * ``delta`` = id - ood (deployment gap; higher-better metrics +=drop, error metrics -=worse)
      * ``relative_drop`` = delta / id
      * ``chance`` / ``floor_norm_drop`` = (id-ood)/(id-chance): fraction of *above-chance*
        performance erased OOD (only where a no-skill floor is defined; NaN otherwise)
      * ``delta_ci_lo/hi`` = bootstrap 95% CI over region×seed estimates ("consistent across regions")
      * ``delta_sample_pt`` + ``delta_sample_ci_lo/hi`` = hierarchical (region/sample) bootstrap CI
        on the ensemble delta, centred on its own point ``delta_sample_pt`` (recomputed from
        ``predictions`` when supplied; a distinct quantity from the seed-mean ``delta``)
      * ``ood_std/min/max`` = spread across holdout regions
      * ``target_id`` / ``inherent_difficulty`` / ``adjusted_delta`` / ``adjusted_relative_drop`` /
        ``adjusted_floor_norm_drop`` = inherent-difficulty decomposition (see below).
    """
    if isinstance(metrics, str):
        metrics = [metrics]
    rng = np.random.default_rng(seed)
    keys = ("model", "benchmark", "method", "probe_family") if any("probe_family" in r for r in rows) else (
        "model", "benchmark", "method"
    )

    def _matches(r, split_regime, budget_type, budget, combo, metric, es=None):
        # es filters target rows by evaluation_split ("full" = full-target zero-shot anchor,
        # "held_out" = the matched fixed-20% test). Source rows without an evaluation_split are
        # the test anchor; old target rows without one are the held-out anchor.
        if es is not None:
            actual = r.get("evaluation_split")
            if actual is None:
                actual = "held_out" if budget_type == "target" else "test"
            if actual != es:
                return False
        return (r.get("split_regime") == split_regime and r.get("budget_type") == budget_type
                and _close(r.get("label_budget"), budget) and metric in r and r[metric] is not None
                and all(r.get(k) == combo[i] for i, k in enumerate(keys)))

    def _eval_matches(row: dict[str, Any], accepted: set[str], *, budget_type: str) -> bool:
        actual = row.get("evaluation_split")
        if actual is None:
            actual = "held_out" if budget_type == "target" else "test"
        return str(actual) in accepted

    def vals(split_regime, budget_type, budget, combo, metric, es=None) -> list[float]:
        out: list[float] = []
        for r in rows:
            if _matches(r, split_regime, budget_type, budget, combo, metric, es=es):
                v = float(r[metric])
                if np.isfinite(v):
                    out.append(v)
        return out

    def vals_by_region(split_regime, budget_type, budget, combo, metric, es=None):
        """Like ``vals`` but grouped by held-out region (each region: one value per seed) --
        the cluster structure the delta CI needs for a hierarchical region/seed bootstrap."""
        out: dict[Any, list[float]] = {}
        for r in rows:
            if _matches(r, split_regime, budget_type, budget, combo, metric, es=es):
                v = float(r[metric])
                if np.isfinite(v):
                    out.setdefault(r.get("holdout"), []).append(v)
        return out

    def _first_vals(split_regime, budget_type, budget, combo, metric, eval_splits: tuple[str, ...]) -> list[float]:
        for eval_split in eval_splits:
            found = vals(split_regime, budget_type, budget, combo, metric, es=eval_split)
            if found:
                return found
        return []

    def _first_vals_by_region(split_regime, budget_type, budget, combo, metric, eval_splits: tuple[str, ...]):
        for eval_split in eval_splits:
            found = vals_by_region(split_regime, budget_type, budget, combo, metric, es=eval_split)
            if found:
                return found
        return {}

    # per-sample predictions indexed by combo -> split_regime, for the within-region bootstrap
    pred_by: dict[tuple, dict[str, list]] = {}
    for p in (predictions or []):
        pred_by.setdefault(tuple(p.get(k) for k in keys), {}).setdefault(p.get("split_regime"), []).append(p)

    combos = sorted({tuple(r.get(k) for k in keys) for r in rows}, key=lambda t: tuple(str(x) for x in t))
    out_rows: list[dict[str, Any]] = []
    for combo in combos:
        # ID test-set label stats (from the random_id anchor rows) -> chance/no-skill floor
        id_stat = [r for r in rows if r.get("split_regime") == "random_id" and r.get("budget_type") == "source"
                   and _close(r.get("label_budget"), id_source_budget)
                   and _eval_matches(r, {"test"}, budget_type="source")
                   and all(r.get(k) == combo[i] for i, k in enumerate(keys))]

        def _avg(field: str, id_stat=id_stat):
            xs = [float(r[field]) for r in id_stat if r.get(field) is not None]
            return float(np.mean(xs)) if xs else None
        pos_rate, n_cls, majority = _avg("test_pos_rate"), _avg("test_n_classes"), _avg("test_majority_rate")

        # Sample-level bootstrap inputs MUST be the same quantity as `delta`: the ID anchor
        # (configured random_id source-budget anchor) and the OOD anchor (geographic_ood target budget 0).
        # `pred_by` holds every budget's predictions, so filter to the anchors before stacking
        # (otherwise the CI would mix all source fractions / zero-shot+few-shot+oracle).
        id_preds = [
            p for p in pred_by.get(combo, {}).get("random_id", [])
            if p.get("budget_type") == "source" and _close(p.get("label_budget"), id_source_budget)
            and _eval_matches(p, {"test"}, budget_type="source")
        ]
        # Primary OOD anchor = the FULL-target zero-shot (evaluation_split "full"); fall back to
        # the held-out rows for older results that predate the full-target anchor.
        _ood_preds_all = [
            p for p in pred_by.get(combo, {}).get("geographic_ood", [])
            if p.get("budget_type") == "target" and _close(p.get("label_budget"), 0.0)
        ]
        _ood_full = [p for p in _ood_preds_all if _eval_matches(p, {"full"}, budget_type="target")]
        _ood_test = [p for p in _ood_preds_all if _eval_matches(p, {"test"}, budget_type="target")]
        ood_preds = _ood_full or _ood_test or _ood_preds_all

        for metric in metrics:
            id_vals = vals("random_id", "source", id_source_budget, combo, metric, es="test")
            ood_vals = _first_vals(
                "geographic_ood", "target", 0.0, combo, metric, ("full", "test", "held_out")
            )
            if not id_vals or not ood_vals:
                continue
            idm, oodm = float(np.mean(id_vals)), float(np.mean(ood_vals))
            id_arr, ood_arr = np.asarray(id_vals), np.asarray(ood_vals)
            # delta CI: HIERARCHICAL bootstrap -- resample held-out regions, then seed-values
            # WITHIN each drawn region (OOD side), and resample seeds for the single-region ID
            # anchor. Flattening region×seed and resampling iid (the old behaviour) treats a
            # region's correlated per-seed replicates as independent and understates the interval.
            ood_by_region = _first_vals_by_region(
                "geographic_ood", "target", 0.0, combo, metric, ("full", "test", "held_out")
            )
            if n_boot and ood_by_region:
                region_vals = [np.asarray(v) for v in ood_by_region.values()]
                n_reg = len(region_vals)
                boot = np.empty(n_boot)
                for b in range(n_boot):
                    id_mean = id_arr[rng.integers(0, len(id_arr), len(id_arr))].mean()
                    ood_draw = np.concatenate([
                        region_vals[ri][rng.integers(0, len(region_vals[ri]), len(region_vals[ri]))]
                        for ri in rng.integers(0, n_reg, n_reg)
                    ])
                    boot[b] = id_mean - ood_draw.mean()
                lo, hi = float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))
            else:
                lo = hi = float("nan")
            row = dict(zip(keys, combo, strict=False))
            row.update({
                "metric": metric, "id": idm, "ood": oodm, "delta": idm - oodm,
                "relative_drop": (idm - oodm) / idm if idm > 0 else float("nan"),
                "delta_ci_lo": lo, "delta_ci_hi": hi,
                "n_id": len(id_vals), "n_ood": len(ood_vals),
                "ood_std": float(np.std(ood_arr)), "ood_min": float(ood_arr.min()), "ood_max": float(ood_arr.max()),
            })
            # worst-region metric: for each seed, find min across holdouts, then average over seeds.
            # Use the FULL-target zero-shot rows (the deployment scope), falling back to held_out
            # for older results -- NEVER mix the two scopes (the held-out 20% is noisier and would
            # masquerade as the worst region).
            _seed_holdout: dict[int, list[float]] = {}
            for want in ("full", "test", "held_out"):
                for r in rows:
                    if (r.get("split_regime") == "geographic_ood" and r.get("budget_type") == "target"
                            and _close(r.get("label_budget"), 0.0) and metric in r and r[metric] is not None
                            and (r.get("evaluation_split") or "held_out") == want
                            and all(r.get(k) == combo[i] for i, k in enumerate(keys))):
                        s = r.get("seed")
                        if s is not None:
                            _seed_holdout.setdefault(int(s), []).append(float(r[metric]))
                if _seed_holdout:
                    break
            if _seed_holdout:
                # worst region = MIN for higher-better metrics, MAX for error metrics (brier/nll/ece)
                pick_worst = max if metric in _LOWER_BETTER else min
                seed_worst = np.asarray([pick_worst(vs) for vs in _seed_holdout.values()])
                row.update({
                    "ood_worst_region": float(np.mean(seed_worst)),
                    "ood_worst_region_std": float(np.std(seed_worst)) if len(seed_worst) > 1 else float("nan"),
                })
            # floor-normalized drop: fraction of above-chance ID performance erased OOD
            chance = _chance(metric, pos_rate, n_cls, majority)
            row["chance"] = float(chance) if chance is not None else float("nan")
            row["floor_norm_drop"] = (
                (idm - oodm) / (idm - chance) if (chance is not None and (idm - chance) > 1e-9) else float("nan")
            )
            # hierarchical (region/sample) bootstrap CI on the ENSEMBLE delta, centred on its own
            # point estimate delta_sample_pt = id_pt - ood_pt (distinct from the seed-mean `delta`).
            if id_preds and ood_preds:
                lo_s, hi_s, id_pt, ood_pt = _sample_delta_ci(metric, id_preds, ood_preds, n_boot_sample, rng)
                if not np.isnan(id_pt):
                    row.update({
                        "delta_sample_pt": id_pt - ood_pt,
                        "delta_sample_ci_lo": lo_s, "delta_sample_ci_hi": hi_s,
                        "n_id_samples": len(id_preds), "n_ood_samples": len(ood_preds),
                    })
            # --- inherent-difficulty decomposition ---
            # target-ID upper-bound (budget -1): train on the 80% target pool, test on the fixed
            # held-out 20%. The decomposition compares it to the zero-shot on that SAME held-out
            # 20% (``ood_matched``) -- like-with-like -- NOT the full-target primary ``ood``.
            tid_vals = _first_vals("geographic_ood", "target", -1.0, combo, metric, ("held_out", "test"))
            if tid_vals:
                tidm = float(np.mean(tid_vals))
                ood_matched_vals = _first_vals(
                    "geographic_ood", "target", 0.0, combo, metric, ("held_out", "test")
                )
                ood_matched = float(np.mean(ood_matched_vals)) if ood_matched_vals else oodm
                row.update({
                    "target_id": tidm,
                    "ood_matched": ood_matched,
                    "inherent_difficulty": idm - tidm,
                    "adjusted_delta": tidm - ood_matched,
                    "adjusted_relative_drop": (tidm - ood_matched) / idm if idm > 0 else float("nan"),
                    "adjusted_floor_norm_drop": (
                        (tidm - ood_matched) / (idm - chance)
                        if (chance is not None and (idm - chance) > 1e-9) else float("nan")
                    ),
                })
            out_rows.append(row)
    return out_rows


def load_env_file(path: Path) -> None:
    """Populate os.environ from a simple .env file (used for data/model paths, tokens)."""
    path = Path(path)
    if not path.exists():
        return
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        key, separator, value = line.partition("=")
        if not separator:
            continue
        key = key.strip()
        value = value.strip()
        if value[:1] == value[-1:] and value.startswith(("'", '"')):
            value = value[1:-1]
        if key:
            os.environ.setdefault(key, value)
