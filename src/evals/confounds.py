from __future__ import annotations

import warnings
from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, f1_score

from utils import perfutils as perf
from utils.ioutils import (
    _LOWER_BETTER,
    _chance,
    _close,
    _sample_delta_ci,
)


def _entropy(counts: np.ndarray) -> float:
    total = counts.sum()
    if total <= 0:
        return 0.0
    p = counts[counts > 0] / total
    return float(-(p * np.log(p)).sum())


def _contingency(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ua, ai = np.unique(np.asarray(a).astype(str), return_inverse=True)
    ub, bi = np.unique(np.asarray(b).astype(str), return_inverse=True)
    counts = np.zeros((len(ua), len(ub)), dtype=np.int64)
    np.add.at(counts, (ai, bi), 1)
    return ua, ub, counts


def confound_pair(a: np.ndarray, b: np.ndarray, max_cells: int = 30) -> dict:
    """Entanglement stats between two per-sample label axes."""
    ua, ub, counts = _contingency(a, b)
    n = int(counts.sum())
    base = {"n_a": int(len(ua)), "n_b": int(len(ub))}
    if n == 0 or len(ua) < 2 or len(ub) < 2:
        return {**base, "nmi": 0.0, "determines_b_given_a": 0.0, "determines_a_given_b": 0.0}
    h_a = _entropy(counts.sum(1))
    h_b = _entropy(counts.sum(0))
    h_ab = _entropy(counts.reshape(-1))
    mi = h_a + h_b - h_ab
    nmi = mi / np.sqrt(h_a * h_b) if h_a > 0 and h_b > 0 else 0.0
    det_b = 1.0 - (h_ab - h_a) / h_b if h_b > 0 else 0.0  # how much a determines b
    det_a = 1.0 - (h_ab - h_b) / h_a if h_a > 0 else 0.0  # how much b determines a
    out = {
        **base,
        "nmi": round(float(np.clip(nmi, 0.0, 1.0)), 4),
        "determines_b_given_a": round(float(np.clip(det_b, 0.0, 1.0)), 4),
        "determines_a_given_b": round(float(np.clip(det_a, 0.0, 1.0)), 4),
    }
    if len(ua) <= max_cells and len(ub) <= max_cells:
        out["contingency"] = {
            str(av): {str(bv): int(counts[i, j]) for j, bv in enumerate(ub) if counts[i, j]} for i, av in enumerate(ua)
        }
    return out


def domain_confound_report(axes: dict[str, np.ndarray | None]) -> dict:
    """Pairwise confound stats between usable per-sample domain axes."""
    usable = [
        name
        for name, values in axes.items()
        if values is not None and len(np.unique(np.asarray(values).astype(str))) > 1
    ]
    report: dict = {
        "n_samples": int(len(axes[usable[0]])) if usable else 0,
        "axis_cardinality": {name: int(len(np.unique(np.asarray(axes[name]).astype(str)))) for name in usable},
        "pairs": {},
    }
    for i in range(len(usable)):
        for j in range(i + 1, len(usable)):
            report["pairs"][f"{usable[i]}__vs__{usable[j]}"] = confound_pair(axes[usable[i]], axes[usable[j]])
    return report


def _apply_transform(transform: Any | None, x: np.ndarray) -> np.ndarray:
    return x if transform is None else transform.transform(x)


def per_class_iou(y_true: np.ndarray, y_pred: np.ndarray, classes: np.ndarray) -> dict[int, float]:
    """IoU per class from flattened per-pixel labels. NaN for classes absent from ``y_true``."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    out: dict[int, float] = {}
    for c in classes:
        c = int(c)
        t = y_true == c
        if not t.any():
            out[c] = float("nan")  # class not present in this test region -> excluded from mIoU
            continue
        p = y_pred == c
        union = int(np.logical_or(t, p).sum())
        out[c] = float(int(np.logical_and(t, p).sum()) / union) if union > 0 else 0.0
    return out


def score_segmentation(
    clf: Any,
    x_test: np.ndarray,
    y_test: np.ndarray,
    *,
    eval_classes: np.ndarray | None = None,
    return_per_sample: bool = False,
) -> dict[str, float] | tuple[dict[str, float], dict[str, np.ndarray]]:
    """Per-pixel multiclass segmentation scores."""
    with perf.measure("probe.score/segmentation", n_samples=len(y_test), n_features=x_test.shape[1]):
        pred = clf.predict(x_test)
    classes = np.asarray(eval_classes if eval_classes is not None else getattr(clf, "classes_", np.unique(y_test)))
    ious = per_class_iou(y_test, pred, classes)
    present = [v for v in ious.values() if not np.isnan(v)]
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="y_pred contains classes not in y_true")
        scores = {
            "miou": float(np.mean(present)) if present else float("nan"),
            "pixel_accuracy": float(accuracy_score(y_test, pred)),
            "macro_f1": float(f1_score(y_test, pred, average="macro", zero_division=0)),
            "weighted_f1": float(f1_score(y_test, pred, average="weighted", zero_division=0)),
            "n_eval_classes": int(len(classes)),
            "n_present_classes": int(len(present)),
        }
    if return_per_sample:
        per_sample = {
            "y_true": np.asarray(y_test, dtype=np.int64),
            "pred": np.asarray(pred, dtype=np.int64),
            "classes": classes.astype(np.int64),
        }
        return scores, per_sample
    return scores


def _miou_from_confusion(conf: np.ndarray) -> float:
    """mIoU over classes present in y_true (rows), from a confusion matrix; NaN if none present."""
    conf = conf.astype(np.float64)
    tp = np.diag(conf)
    row, col = conf.sum(1), conf.sum(0)  # true support, predicted count
    present = row > 0
    union = row + col - tp
    iou = np.divide(tp, union, out=np.zeros_like(tp), where=union > 0)
    return float(iou[present].mean()) if present.any() else float("nan")


def _as_eval_indices(values: np.ndarray, classes: np.ndarray, name: str) -> np.ndarray:
    values = np.asarray(values, dtype=np.int64)
    classes = np.asarray(classes, dtype=np.int64)
    if classes.size == 0:
        raise ValueError("eval_classes is empty")
    if np.array_equal(classes, np.arange(classes.size)):
        bad = (values < 0) | (values >= classes.size)
        if bad.any():
            raise ValueError(f"{name} contains values outside eval_classes: {np.unique(values[bad])[:10].tolist()}")
        return values
    order = np.argsort(classes)
    sorted_classes = classes[order]
    pos = np.searchsorted(sorted_classes, values)
    valid = pos < len(sorted_classes)
    safe_pos = np.minimum(pos, len(sorted_classes) - 1)
    valid &= sorted_classes[safe_pos] == values
    if not valid.all():
        raise ValueError(f"{name} contains values outside eval_classes: {np.unique(values[~valid])[:10].tolist()}")
    return order[pos]


def _segmentation_metrics_from_confusion(conf: np.ndarray, tile_mious: list[float]) -> dict[str, float]:
    """Segmentation metrics from an accumulated confusion matrix."""
    c = conf.astype(np.float64)
    tp = np.diag(c)
    row, col, total = c.sum(1), c.sum(0), c.sum()
    present = row > 0
    union = row + col - tp
    iou = np.divide(tp, union, out=np.zeros_like(tp), where=union > 0)
    prec = np.divide(tp, col, out=np.zeros_like(tp), where=col > 0)
    rec = np.divide(tp, row, out=np.zeros_like(tp), where=row > 0)
    f1 = np.divide(2 * prec * rec, prec + rec, out=np.zeros_like(tp), where=(prec + rec) > 0)
    labelset = (row > 0) | (col > 0)  # classes present in y_true OR y_pred (sklearn macro default)
    out = {
        "miou": float(iou[present].mean()) if present.any() else float("nan"),
        "pixel_accuracy": float(tp.sum() / total) if total > 0 else float("nan"),
        "macro_f1": float(f1[labelset].mean()) if labelset.any() else 0.0,
        "weighted_f1": float((row * f1).sum() / row.sum()) if row.sum() > 0 else 0.0,
        "n_eval_classes": int(conf.shape[0]),
        "n_present_classes": int(present.sum()),
    }
    if tile_mious:
        arr = np.asarray(tile_mious)
        out.update(
            {
                "mean_per_tile_miou": float(arr.mean()),
                "worst_tile_miou": float(arr.min()),
                "n_tiles_scored": len(arr),
            }
        )
    else:
        out.update({"mean_per_tile_miou": float("nan"), "worst_tile_miou": float("nan"), "n_tiles_scored": 0})
    return out


def score_segmentation_streamed(clf, tiles, eval_classes: np.ndarray, transform: Any | None = None) -> dict[str, float]:
    """Exact full-fold segmentation scoring by streaming dense tiles."""
    k = len(eval_classes)
    conf = np.zeros((k, k), dtype=np.int64)
    tile_mious: list[float] = []
    n_pixels = 0
    with perf.measure("probe.score/segmentation_streamed", n_features=-1):
        for features, labels in tiles:
            labels = np.asarray(labels)
            if labels.size == 0:
                continue
            features = _apply_transform(transform, np.asarray(features, dtype=np.float32))
            pred = np.asarray(clf.predict(features))
            lab = _as_eval_indices(labels, eval_classes, "segmentation labels")
            prd = _as_eval_indices(pred, eval_classes, "segmentation predictions")
            tile_conf = np.bincount(lab * k + prd, minlength=k * k).reshape(k, k)
            conf += tile_conf
            miou_t = _miou_from_confusion(tile_conf)
            if not np.isnan(miou_t):
                tile_mious.append(miou_t)
            n_pixels += int(labels.size)
    metrics = _segmentation_metrics_from_confusion(conf, tile_mious)
    metrics["n_test"] = n_pixels
    return metrics


def compute_deltas(
    rows: list[dict[str, Any]],
    metrics: list[str],
    *,
    predictions: list[dict[str, Any]] | None = None,
    id_source_budget: float | int = 1.0,
    ood_target_budget: float | int = 0.0,
    target_id_budget: float | int | None = -1.0,
    n_boot: int = 2000,
    n_boot_sample: int = 1000,
    seed: int = 0,
) -> list[dict[str, Any]]:
    """ID-to-OOD deployment-gap rows for configured metrics."""
    if isinstance(metrics, str):
        metrics = [metrics]
    rng = np.random.default_rng(seed)
    keys = (
        ("model", "benchmark", "method", "probe_family")
        if any("probe_family" in r for r in rows)
        else ("model", "benchmark", "method")
    )

    def _matches(r, split_regime, budget_type, budget, combo, metric, es=None):
        if es is not None:
            actual = r.get("evaluation_split")
            if actual is None:
                actual = "held_out" if budget_type == "target" else "test"
            if actual != es:
                return False
        return (
            r.get("split_regime") == split_regime
            and r.get("budget_type") == budget_type
            and _close(r.get("label_budget"), budget)
            and metric in r
            and r[metric] is not None
            and all(r.get(k) == combo[i] for i, k in enumerate(keys))
        )

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

    pred_by: dict[tuple, dict[str, list]] = {}
    for p in predictions or []:
        pred_by.setdefault(tuple(p.get(k) for k in keys), {}).setdefault(p.get("split_regime"), []).append(p)

    combos = sorted({tuple(r.get(k) for k in keys) for r in rows}, key=lambda t: tuple(str(x) for x in t))
    out_rows: list[dict[str, Any]] = []
    for combo in combos:
        id_stat = [
            r
            for r in rows
            if r.get("split_regime") == "random_id"
            and r.get("budget_type") == "source"
            and _close(r.get("label_budget"), id_source_budget)
            and _eval_matches(r, {"test"}, budget_type="source")
            and all(r.get(k) == combo[i] for i, k in enumerate(keys))
        ]

        def _avg(field: str, id_stat=id_stat):
            xs = [float(r[field]) for r in id_stat if r.get(field) is not None]
            return float(np.mean(xs)) if xs else None

        pos_rate, n_cls, majority = _avg("test_pos_rate"), _avg("test_n_classes"), _avg("test_majority_rate")

        id_preds = [
            p
            for p in pred_by.get(combo, {}).get("random_id", [])
            if p.get("budget_type") == "source"
            and _close(p.get("label_budget"), id_source_budget)
            and _eval_matches(p, {"test"}, budget_type="source")
        ]
        _ood_preds_all = [
            p
            for p in pred_by.get(combo, {}).get("geographic_ood", [])
            if p.get("budget_type") == "target" and _close(p.get("label_budget"), ood_target_budget)
        ]
        _ood_full = [p for p in _ood_preds_all if _eval_matches(p, {"full"}, budget_type="target")]
        _ood_test = [p for p in _ood_preds_all if _eval_matches(p, {"test"}, budget_type="target")]
        ood_preds = _ood_full or _ood_test or _ood_preds_all

        for metric in metrics:
            id_vals = vals("random_id", "source", id_source_budget, combo, metric, es="test")
            ood_vals = _first_vals(
                "geographic_ood", "target", ood_target_budget, combo, metric, ("full", "test", "held_out")
            )
            if not id_vals or not ood_vals:
                continue
            idm, oodm = float(np.mean(id_vals)), float(np.mean(ood_vals))
            id_arr, ood_arr = np.asarray(id_vals), np.asarray(ood_vals)
            ood_by_region = _first_vals_by_region(
                "geographic_ood", "target", ood_target_budget, combo, metric, ("full", "test", "held_out")
            )
            if n_boot and ood_by_region:
                region_vals = [np.asarray(v) for v in ood_by_region.values()]
                n_reg = len(region_vals)
                boot = np.empty(n_boot)
                for b in range(n_boot):
                    id_mean = id_arr[rng.integers(0, len(id_arr), len(id_arr))].mean()
                    ood_draw = np.concatenate(
                        [
                            region_vals[ri][rng.integers(0, len(region_vals[ri]), len(region_vals[ri]))]
                            for ri in rng.integers(0, n_reg, n_reg)
                        ]
                    )
                    boot[b] = id_mean - ood_draw.mean()
                lo, hi = float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))
            else:
                lo = hi = float("nan")
            row = dict(zip(keys, combo, strict=False))
            row.update(
                {
                    "metric": metric,
                    "id": idm,
                    "ood": oodm,
                    "delta": idm - oodm,
                    "relative_drop": (idm - oodm) / idm if idm > 0 else float("nan"),
                    "delta_ci_lo": lo,
                    "delta_ci_hi": hi,
                    "n_id": len(id_vals),
                    "n_ood": len(ood_vals),
                    "ood_std": float(np.std(ood_arr)),
                    "ood_min": float(ood_arr.min()),
                    "ood_max": float(ood_arr.max()),
                }
            )
            _seed_holdout: dict[int, list[float]] = {}
            for want in ("full", "test", "held_out"):
                for r in rows:
                    if (
                        r.get("split_regime") == "geographic_ood"
                        and r.get("budget_type") == "target"
                        and _close(r.get("label_budget"), ood_target_budget)
                        and metric in r
                        and r[metric] is not None
                        and (r.get("evaluation_split") or "held_out") == want
                        and all(r.get(k) == combo[i] for i, k in enumerate(keys))
                    ):
                        s = r.get("seed")
                        if s is not None:
                            _seed_holdout.setdefault(int(s), []).append(float(r[metric]))
                if _seed_holdout:
                    break
            if _seed_holdout:
                pick_worst = max if metric in _LOWER_BETTER else min
                seed_worst = np.asarray([pick_worst(vs) for vs in _seed_holdout.values()])
                row.update(
                    {
                        "ood_worst_region": float(np.mean(seed_worst)),
                        "ood_worst_region_std": float(np.std(seed_worst)) if len(seed_worst) > 1 else float("nan"),
                    }
                )
            chance = _chance(metric, pos_rate, n_cls, majority)
            row["chance"] = float(chance) if chance is not None else float("nan")
            row["floor_norm_drop"] = (
                (idm - oodm) / (idm - chance) if (chance is not None and (idm - chance) > 1e-9) else float("nan")
            )
            if id_preds and ood_preds:
                lo_s, hi_s, id_pt, ood_pt = _sample_delta_ci(metric, id_preds, ood_preds, n_boot_sample, rng)
                if not np.isnan(id_pt):
                    row.update(
                        {
                            "delta_sample_pt": id_pt - ood_pt,
                            "delta_sample_ci_lo": lo_s,
                            "delta_sample_ci_hi": hi_s,
                            "n_id_samples": len(id_preds),
                            "n_ood_samples": len(ood_preds),
                        }
                    )
            tid_vals = (
                _first_vals("geographic_ood", "target", target_id_budget, combo, metric, ("held_out", "test"))
                if target_id_budget is not None
                else []
            )
            if tid_vals:
                tidm = float(np.mean(tid_vals))
                ood_matched_vals = _first_vals(
                    "geographic_ood", "target", ood_target_budget, combo, metric, ("held_out", "test")
                )
                ood_matched = float(np.mean(ood_matched_vals)) if ood_matched_vals else oodm
                row.update(
                    {
                        "target_id": tidm,
                        "ood_matched": ood_matched,
                        "inherent_difficulty": idm - tidm,
                        "adjusted_delta": tidm - ood_matched,
                        "adjusted_relative_drop": (tidm - ood_matched) / idm if idm > 0 else float("nan"),
                        "adjusted_floor_norm_drop": (
                            (tidm - ood_matched) / (idm - chance)
                            if (chance is not None and (idm - chance) > 1e-9)
                            else float("nan")
                        ),
                    }
                )
            out_rows.append(row)
    return out_rows
