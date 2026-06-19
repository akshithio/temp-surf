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


def hf_download_to(repo_id: str, filename: str, dest: Path) -> Path:
    """Download ``filename`` from ``repo_id`` into ``dest.parent`` and return ``dest``."""
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise ImportError(
            "huggingface-hub is required to download model weights. Run `uv pip install -e .` "
            "from the project env or set an explicit local weights path."
        ) from exc

    dest = Path(dest).expanduser()
    dest.parent.mkdir(parents=True, exist_ok=True)
    downloaded = Path(hf_hub_download(repo_id=repo_id, filename=filename, local_dir=str(dest.parent)))
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
    """Append rows to a JSON-lines log (creates parents). Used for crash-resumable results."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, default=_json_default) + "\n")


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


def _auc(y: np.ndarray, p: np.ndarray) -> float:
    n_pos = int((y == 1).sum())
    n_neg = len(y) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = np.argsort(np.argsort(p, kind="mergesort"), kind="mergesort").astype(float) + 1  # ties: negligible for continuous probs
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


def _pred_arrays(preds: list[dict[str, Any]] | None):
    """Stack per-sample prediction dicts into (y, prob, pred_default, pred_calibrated) arrays."""
    if not preds:
        return None
    preds = [p for p in preds if "prob" in p and "pred_default" in p and "pred_calibrated" in p]
    if not preds:
        return None
    return (
        np.array([p["y_true"] for p in preds], dtype=np.float64),
        np.array([p["prob"] for p in preds], dtype=np.float64),
        np.array([p["pred_default"] for p in preds], dtype=np.int64),
        np.array([p["pred_calibrated"] for p in preds], dtype=np.int64),
    )


def _sample_delta_ci(metric, id_arrs, ood_arrs, n_boot, rng):
    """Within-region bootstrap of the delta over individual test SAMPLES (pooled).

    Resamples ID and OOD test samples with replacement and recomputes the metric each
    time -> a CI that reflects test-sample uncertainty (complements the region×seed CI).
    Returns (ci_lo, ci_hi, id_point, ood_point); NaN points for non-binary metrics.
    """
    yi, pi, di, ci = id_arrs
    yo, po, do, co = ood_arrs
    id_pt, ood_pt = _bin_metric(metric, yi, pi, di, ci), _bin_metric(metric, yo, po, do, co)
    if np.isnan(id_pt) or np.isnan(ood_pt) or not n_boot:
        return float("nan"), float("nan"), id_pt, ood_pt
    # raw id - ood, matching the region-level `delta` convention (higher-better: +=drop;
    # lower-better metrics like brier/nll/ece: -=worse OOD). Read direction per metric.
    ni, no = len(yi), len(yo)
    deltas = np.empty(n_boot)
    for b in range(n_boot):
        ii, oo = rng.integers(0, ni, ni), rng.integers(0, no, no)
        deltas[b] = (_bin_metric(metric, yi[ii], pi[ii], di[ii], ci[ii])
                     - _bin_metric(metric, yo[oo], po[oo], do[oo], co[oo]))
    return float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5)), id_pt, ood_pt


def compute_deltas(
    rows: list[dict[str, Any]],
    metrics: list[str],
    *,
    predictions: list[dict[str, Any]] | None = None,
    n_boot: int = 2000,
    n_boot_sample: int = 1000,
    seed: int = 0,
) -> list[dict[str, Any]]:
    """ID→OOD drop per metric and per (model, benchmark, method).

    ``id``  = metric on the random-split in-distribution anchor (random_id, source budget 1.0).
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
      * ``delta_sample_ci_lo/hi`` = bootstrap 95% CI over individual test samples (pooled),
        recomputed from ``predictions`` when supplied (within-region precision)
      * ``ood_std/min/max`` = spread across holdout regions; ``grouped_ood`` as a secondary anchor
      * ``ood_hybrid`` / ``ood_hybrid_std`` / ``ood_hybrid_min`` / ``delta_hybrid`` = hybrid OOD
        anchor (geographic holdout × source grouped folds): how OOD performance varies with
        source-region composition
      * ``target_id`` / ``inherent_difficulty`` / ``adjusted_delta`` / ``adjusted_relative_drop`` /
        ``adjusted_floor_norm_drop`` = WILDS-style decomposition (see below).
    """
    if isinstance(metrics, str):
        metrics = [metrics]
    rng = np.random.default_rng(seed)
    keys = ("model", "benchmark", "method", "probe_family") if any("probe_family" in r for r in rows) else (
        "model", "benchmark", "method"
    )

    def vals(split_regime: str, budget_type: str, budget: float, combo: tuple, metric: str) -> list[float]:
        out: list[float] = []
        for r in rows:
            if (r.get("split_regime") == split_regime and r.get("budget_type") == budget_type
                    and _close(r.get("label_budget"), budget) and metric in r and r[metric] is not None
                    and all(r.get(k) == combo[i] for i, k in enumerate(keys))):
                v = float(r[metric])
                if np.isfinite(v):
                    out.append(v)
        return out

    # per-sample predictions indexed by combo -> split_regime, for the within-region bootstrap
    pred_by: dict[tuple, dict[str, list]] = {}
    for p in (predictions or []):
        pred_by.setdefault(tuple(p.get(k) for k in keys), {}).setdefault(p.get("split_regime"), []).append(p)

    combos = sorted({tuple(r.get(k) for k in keys) for r in rows}, key=lambda t: tuple(str(x) for x in t))
    out_rows: list[dict[str, Any]] = []
    for combo in combos:
        # ID test-set label stats (from the random_id anchor rows) -> chance/no-skill floor
        id_stat = [r for r in rows if r.get("split_regime") == "random_id" and r.get("budget_type") == "source"
                   and _close(r.get("label_budget"), 1.0) and all(r.get(k) == combo[i] for i, k in enumerate(keys))]

        def _avg(field: str, id_stat=id_stat):
            xs = [float(r[field]) for r in id_stat if r.get(field) is not None]
            return float(np.mean(xs)) if xs else None
        pos_rate, n_cls, majority = _avg("test_pos_rate"), _avg("test_n_classes"), _avg("test_majority_rate")

        id_samp = _pred_arrays(pred_by.get(combo, {}).get("random_id"))
        ood_samp = _pred_arrays(pred_by.get(combo, {}).get("geographic_ood"))

        for metric in metrics:
            id_vals = vals("random_id", "source", 1.0, combo, metric)
            ood_vals = vals("geographic_ood", "target", 0.0, combo, metric)
            if not id_vals or not ood_vals:
                continue
            idm, oodm = float(np.mean(id_vals)), float(np.mean(ood_vals))
            id_arr, ood_arr = np.asarray(id_vals), np.asarray(ood_vals)
            # region×seed bootstrap CI on the delta (captures "consistent across regions")
            if n_boot:
                boot = (id_arr[rng.integers(0, len(id_arr), (n_boot, len(id_arr)))].mean(1)
                        - ood_arr[rng.integers(0, len(ood_arr), (n_boot, len(ood_arr)))].mean(1))
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
            # worst-region metric: for each seed, find min across holdouts,
            # then average over seeds.
            _seed_holdout: dict[int, list[float]] = {}
            for r in rows:
                if (r.get("split_regime") == "geographic_ood" and r.get("budget_type") == "target"
                        and _close(r.get("label_budget"), 0.0) and metric in r and r[metric] is not None
                        and all(r.get(k) == combo[i] for i, k in enumerate(keys))):
                    s = r.get("seed")
                    if s is not None:
                        _seed_holdout.setdefault(int(s), []).append(float(r[metric]))
            if _seed_holdout:
                seed_worst = np.asarray([min(vs) for vs in _seed_holdout.values()])
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
            # within-region (per-test-sample) bootstrap CI on the delta, from pooled predictions
            if id_samp is not None and ood_samp is not None:
                lo_s, hi_s, id_pt, ood_pt = _sample_delta_ci(metric, id_samp, ood_samp, n_boot_sample, rng)
                if not np.isnan(id_pt):
                    row.update({
                        "delta_sample_ci_lo": lo_s, "delta_sample_ci_hi": hi_s,
                        "n_id_samples": len(id_samp[0]), "n_ood_samples": len(ood_samp[0]),
                    })
            grouped = vals("grouped_ood", "target", 0.0, combo, metric)
            if grouped:
                row["ood_grouped"] = float(np.mean(grouped))
                row["delta_grouped"] = idm - float(np.mean(grouped))
            phenology = vals("phenology_ood", "target", 0.0, combo, metric)
            if phenology:
                row["ood_phenology"] = float(np.mean(phenology))
                row["ood_phenology_std"] = float(np.std(phenology))
                row["ood_phenology_min"] = float(np.min(phenology))
                row["delta_phenology"] = idm - float(np.mean(phenology))
            # --- Hybrid OOD: geographic holdout × source grouped folds ---
            hybrid = vals("hybrid_ood", "target", 0.0, combo, metric)
            if hybrid:
                row["ood_hybrid"] = float(np.mean(hybrid))
                row["ood_hybrid_std"] = float(np.std(hybrid))
                row["ood_hybrid_min"] = float(np.min(hybrid))
                row["delta_hybrid"] = idm - float(np.mean(hybrid))
            # --- WILDS-style inherent-difficulty decomposition ---
            # target-ID upper-bound: train on 80% of target (no source),
            # test on remaining 20%.  Budget sentinel = -1.
            tid_vals = vals("geographic_ood", "target", -1.0, combo, metric)
            if tid_vals:
                tidm = float(np.mean(tid_vals))
                row.update({
                    "target_id": tidm,
                    "inherent_difficulty": idm - tidm,
                    "adjusted_delta": tidm - oodm,
                    "adjusted_relative_drop": (tidm - oodm) / idm if idm > 0 else float("nan"),
                    "adjusted_floor_norm_drop": (
                        (tidm - oodm) / (idm - chance)
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
