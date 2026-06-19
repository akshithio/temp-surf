"""Benchmark-agnostic evaluation protocol: constants, sweep orchestration.

Scope:
  * shared protocol constants (holdouts, budgets, metrics)
  * budget-sweep orchestration (``_sweep_budgets``, ``_sweep_target_budgets``)
  * per-benchmark public runners (``run_probes*``)
"""

from __future__ import annotations

import os
import warnings
from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import train_test_split

from evals.probes import (  # noqa: F401
    FeatureTransform,
    _apply,
    best_f1_threshold,
    expected_calibration_error,
    fit_probe_multiclass,
    fit_probe_with_calibration,
    score_binary,
    score_multiclass,
    score_segmentation,
    score_segmentation_per_tile,
)

# Split-regime constructors now live with their regimes (evals/regimes/<regime>.py);
# re-exported here so existing callers (EV.make_splits, ...) keep working.
from evals.regimes.geographic_ood import make_strict_holdout_splits  # noqa: F401
from evals.regimes.grouped_ood import make_grouped_holdout_folds  # noqa: F401
from evals.regimes.random_id import make_splits  # noqa: F401
from utils import perfutils as perf

for _thread_var in [
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
]:
    os.environ.setdefault(_thread_var, "1")


def subset_indices(y: np.ndarray, budget: float, seed: int, stratify: bool = True) -> np.ndarray:
    """Sub-sample of indices for a sparse-label budget.

    Falls back to non-stratified if a class is too small to stratify at
    the requested budget (expected for multiclass at tiny budgets -- unseen-class
    drops are part of the EuroCropsML transfer story).
    """
    idx = np.arange(len(y))
    if budget >= 1.0:
        return idx
    k = min(len(idx) - 1, max(2, int(round(budget * len(idx)))))
    strat = y if stratify else None
    try:
        sub, _ = train_test_split(idx, train_size=k, random_state=seed, stratify=strat)
    except ValueError:
        sub, _ = train_test_split(idx, train_size=k, random_state=seed, stratify=None)
    return np.sort(sub)


# --------------------------------------------------------------------------- #
# Protocol constants
# --------------------------------------------------------------------------- #

# Sparse-label probe budgets.
#
# SOURCE_BUDGETS  — fraction of available source-pool labels (secondary diagnostic).
#                   Answers: "Does more source-region training data help geographic
#                   generalization?"
# TARGET_BUDGETS — absolute count of target-region labels used for training (main
#                   experiment).
SOURCE_BUDGETS: list[float] = [0.05, 0.10, 0.25, 1.00]
TARGET_BUDGETS: list[int] = [5, 10, 25, 50]

# Sentinel for the in-distribution target upper bound:
#   Train on 80% of the target region (no source), test on the remaining 20%.
#   Provides the "how good can we get on this region if trained in-distribution?"
#   baseline, used to separate transfer loss from inherent regional difficulty.
TARGET_ID_UPPER_BOUND: int = -1

# Extended target budgets including the strict geographic holdout (0) and the
# target-ID upper bound (-1).
ALL_TARGET_BUDGETS: list[int] = [0, *TARGET_BUDGETS, TARGET_ID_UPPER_BOUND]

# Reported metrics per label family. calibrated_* use a source-validation threshold
# rather than 0.5 -- default-0.5 F1 misrepresents transfer under distribution shift.
METRICS_BINARY_BASE: list[str] = [
    "f1",
    "auc",
    "balanced_accuracy",
    "calibrated_f1",
    "calibrated_balanced_accuracy",
    "calibrated_f1_target_optimal",
    "optimal_threshold_test",
    "ece",
    "brier",  # proper scoring rule: mean squared prob error (lower better)
    "nll",    # negative log-likelihood / log-loss (lower better)
]
METRICS_BINARY_WORST_GROUP: list[str] = [
    "worst_group_f1",
    "worst_group_balanced_accuracy",
    "worst_group_calibrated_f1",
    "worst_group_calibrated_balanced_accuracy",
    "worst_group_score",
]
METRICS: list[str] = METRICS_BINARY_BASE
METRICS_BINARY: list[str] = [*METRICS_BINARY_BASE, *METRICS_BINARY_WORST_GROUP]
METRICS_MULTICLASS_BASE: list[str] = [
    "macro_f1",
    "weighted_f1",
    "balanced_accuracy",
    "accuracy",
    "macro_auc",
]
METRICS_MULTICLASS_WORST_GROUP: list[str] = [
    "worst_group_macro_f1",
    "worst_group_weighted_f1",
    "worst_group_balanced_accuracy",
    "worst_group_accuracy",
    "worst_group_score",
]
METRICS_MULTICLASS: list[str] = [*METRICS_MULTICLASS_BASE, *METRICS_MULTICLASS_WORST_GROUP]
METRICS_SEGMENTATION: list[str] = [
    "miou", "pixel_accuracy", "macro_f1", "weighted_f1",
    "mean_per_tile_miou", "worst_tile_miou", "n_tiles_scored",
]
METRIC_ROLES: dict[str, dict[str, list[str]]] = {
    "binary": {
        "deployment": [
            "calibrated_f1",
            "calibrated_balanced_accuracy",
            "worst_group_calibrated_f1",
            "worst_group_calibrated_balanced_accuracy",
        ],
        "diagnostic": [
            "f1", "auc", "balanced_accuracy", "calibrated_f1_target_optimal",
            "optimal_threshold_test", "ece", "brier", "nll",
        ],
    },
    "multiclass": {
        "deployment": ["macro_f1", "balanced_accuracy", "worst_group_macro_f1", "worst_group_balanced_accuracy"],
        "diagnostic": ["weighted_f1", "accuracy", "macro_auc"],
    },
    "segmentation": {
        "deployment": ["miou", "mean_per_tile_miou", "worst_tile_miou"],
        "diagnostic": ["pixel_accuracy", "macro_f1", "weighted_f1", "n_tiles_scored"],
    },
}

# --------------------------------------------------------------------------- #
# Shared budget sweep + per-benchmark runners
# --------------------------------------------------------------------------- #

def _append_prediction_rows(
    predictions: list[dict[str, Any]] | None,
    *,
    meta: dict[str, Any],
    seed: int,
    budget_type: str,
    label_budget: float | int,
    n_train_sub: int,
    sample_ids: np.ndarray,
    groups_test: np.ndarray | None,
    per_sample: dict[str, np.ndarray] | None,
) -> None:
    """Append one prediction row per test sample for a completed probe budget cell."""
    if predictions is None or per_sample is None:
        return

    sample_ids = np.asarray(sample_ids)
    groups_arr = np.asarray(groups_test) if groups_test is not None else np.asarray([""] * len(sample_ids))
    base = {
        **meta,
        "budget_type": budget_type,
        "label_budget": label_budget,
        "seed": seed,
        "n_train_sub": int(n_train_sub),
    }
    y_true = np.asarray(per_sample["y_true"])
    if "prob" in per_sample:
        prob = np.asarray(per_sample["prob"], dtype=np.float64)
        pred_default = np.asarray(per_sample["pred_default"], dtype=np.int64)
        pred_calibrated = np.asarray(per_sample["pred_calibrated"], dtype=np.int64)
        for i in range(len(y_true)):
            predictions.append({
                **base,
                "sample_id": int(sample_ids[i]),
                "group": str(groups_arr[i]),
                "y_true": int(y_true[i]),
                "prob": float(prob[i]),
                "pred_default": int(pred_default[i]),
                "pred_calibrated": int(pred_calibrated[i]),
            })
        return

    classes = np.asarray(per_sample.get("classes", []), dtype=np.int64)
    proba = np.asarray(per_sample.get("proba", np.zeros((len(y_true), 0))), dtype=np.float64)
    pred = np.asarray(per_sample["pred"], dtype=np.int64)
    class_to_col = {int(c): j for j, c in enumerate(classes)}
    for i in range(len(y_true)):
        pred_col = class_to_col.get(int(pred[i]))
        true_col = class_to_col.get(int(y_true[i]))
        probs = proba[i].astype(float).tolist() if proba.size else []
        predictions.append({
            **base,
            "sample_id": int(sample_ids[i]),
            "group": str(groups_arr[i]),
            "y_true": int(y_true[i]),
            "pred": int(pred[i]),
            "prob_pred": float(proba[i, pred_col]) if pred_col is not None and proba.size else float("nan"),
            "prob_true": float(proba[i, true_col]) if true_col is not None and proba.size else float("nan"),
            "classes": classes.astype(int).tolist(),
            "probs": probs,
        })


def _safe_balanced_accuracy(y_true: np.ndarray, pred: np.ndarray) -> float:
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="y_pred contains classes not in y_true")
        return float(balanced_accuracy_score(y_true, pred))


def _score_binary_group(per_sample: dict[str, np.ndarray], mask: np.ndarray) -> dict[str, float]:
    y = np.asarray(per_sample["y_true"])[mask]
    pred_default = np.asarray(per_sample["pred_default"])[mask]
    pred_cal = np.asarray(per_sample["pred_calibrated"])[mask]
    return {
        "f1": float(f1_score(y, pred_default, zero_division=0)),
        "balanced_accuracy": _safe_balanced_accuracy(y, pred_default),
        "calibrated_f1": float(f1_score(y, pred_cal, zero_division=0)),
        "calibrated_balanced_accuracy": _safe_balanced_accuracy(y, pred_cal),
    }


def _score_multiclass_group(per_sample: dict[str, np.ndarray], mask: np.ndarray) -> dict[str, float]:
    y = np.asarray(per_sample["y_true"])[mask]
    pred = np.asarray(per_sample["pred"])[mask]
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="y_pred contains classes not in y_true")
        return {
            "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
            "weighted_f1": float(f1_score(y, pred, average="weighted", zero_division=0)),
            "balanced_accuracy": _safe_balanced_accuracy(y, pred),
            "accuracy": float(accuracy_score(y, pred)),
        }


def _worst_group_scores(
    per_sample: dict[str, np.ndarray] | None,
    groups_test: np.ndarray | None,
) -> dict[str, Any]:
    """Compute first-class subpopulation worst-group metrics for one evaluated split.

    This is the WILDS-style "average can hide a bad deployment subgroup" view. For
    ``random_id`` rows it is a true subpopulation metric: train/test include all
    domains, and the row also reports the worst test domain. For strict OOD rows
    with one target domain, worst-group equals the target-domain score.
    """
    if per_sample is None or groups_test is None:
        return {}
    groups_arr = np.asarray(groups_test, dtype=object)
    y_true = np.asarray(per_sample["y_true"])
    if len(groups_arr) != len(y_true) or len(y_true) == 0:
        return {}

    if "prob" in per_sample:
        scorer = _score_binary_group
        metric_names = ["f1", "balanced_accuracy", "calibrated_f1", "calibrated_balanced_accuracy"]
        primary = "calibrated_f1"
    else:
        scorer = _score_multiclass_group
        metric_names = ["macro_f1", "weighted_f1", "balanced_accuracy", "accuracy"]
        primary = "macro_f1"

    by_metric: dict[str, list[tuple[str, float]]] = {name: [] for name in metric_names}
    for group in sorted({str(g) for g in groups_arr.tolist()}):
        mask = np.array([str(g) == group for g in groups_arr], dtype=bool)
        if not mask.any():
            continue
        scores = scorer(per_sample, mask)
        for name, value in scores.items():
            if np.isfinite(value):
                by_metric[name].append((group, value))

    out: dict[str, Any] = {"n_groups_scored": int(len({str(g) for g in groups_arr.tolist()}))}
    for name, values in by_metric.items():
        if not values:
            continue
        group, value = min(values, key=lambda item: item[1])
        out[f"worst_group_{name}"] = float(value)
        out[f"worst_group_name_{name}"] = group
    if primary in by_metric and by_metric[primary]:
        group, value = min(by_metric[primary], key=lambda item: item[1])
        out["worst_group"] = group
        out["worst_group_metric"] = primary
        out["worst_group_score"] = float(value)
    return out


def _sweep_budgets(
    rows: list[dict[str, Any]],
    x_train: np.ndarray,
    x_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    seed: int,
    fit_score: Any,
    *,
    transform: FeatureTransform | None,
    budgets: list[float],
    meta: dict[str, Any] | None,
    stratify: bool,
    groups_train: np.ndarray | None,
    predictions: list[dict[str, Any]] | None = None,
    sample_ids_test: np.ndarray | None = None,
    groups_test: np.ndarray | None = None,
    x_val: np.ndarray | None = None,
    y_val: np.ndarray | None = None,
    family: str = "logistic",
) -> None:
    """Apply the fitted transform, then sweep source-fraction budgets, appending one
    row each.

    ``fit_score(x_tr_sub, y_tr_sub, x_test, y_test, probe_seed, x_cal, y_cal) ->
    (scores, extra)`` is the only benchmark-specific piece; everything else
    (transform application, budget sub-sampling, metadata, bookkeeping) is shared.
    ``x_val``/``y_val`` is the regime's held-out validation set, used by the binary
    probe to calibrate its threshold (ignored by the multiclass probe).
    """
    meta = dict(meta or {})
    x_train = _apply(transform, x_train)
    x_test = _apply(transform, x_test)
    x_val = _apply(transform, x_val) if x_val is not None and len(x_val) else None
    for budget in budgets:
        sub_seed = seed + int(round(budget * 1000))
        if transform is not None and hasattr(transform, "subset_indices"):
            sub = transform.subset_indices(y_train, groups_train, budget, sub_seed)
        else:
            sub = subset_indices(y_train, budget, sub_seed, stratify=stratify)

        identity = {
            "seed": seed,
            "holdout": meta.get("holdout"),
            "method": meta.get("method"),
            "budget_type": "source",
            "label_budget": budget,
        }
        perf.set_identity(identity)
        with perf.measure(f"probe.sweep.source/{meta.get('benchmark', '?')}/{meta.get('method', '?')}",
                          n_train=len(sub), n_test=len(y_test)):
            result = fit_score(
                x_train[sub], y_train[sub], x_test, y_test, seed + int(round(budget * 1000)) + 17,
                x_val, y_val,
            )
            if len(result) == 3:
                scores, extra, per_sample = result
            else:
                scores, extra = result
                per_sample = None
        perf.set_identity(None)
        scores = {**scores, **_worst_group_scores(per_sample, groups_test)}

        rows.append(
            {
                **meta,
                "budget_type": "source",
                "label_budget": budget,
                "seed": seed,
                "n_train_sub": int(len(sub)),
                "n_test": int(len(y_test)),
                **extra,
                **scores,
            }
        )
        _append_prediction_rows(
            predictions,
            meta=meta,
            seed=seed,
            budget_type="source",
            label_budget=budget,
            n_train_sub=len(sub),
            sample_ids=np.asarray(sample_ids_test if sample_ids_test is not None else np.arange(len(y_test))),
            groups_test=groups_test,
            per_sample=per_sample,
        )


# --------------------------------------------------------------------------- #
# Target-budget sweep: sample N target labels for training, rest stays as test
# --------------------------------------------------------------------------- #


def _sweep_target_budgets(
    rows: list[dict[str, Any]],
    x_source: np.ndarray,
    x_target_full: np.ndarray,
    y_source: np.ndarray,
    y_target_full: np.ndarray,
    seed: int,
    fit_score: Any,
    *,
    transform: FeatureTransform | None,
    budgets: list[int],
    meta: dict[str, Any] | None,
    stratify: bool,
    groups_source: np.ndarray | None,
    predictions: list[dict[str, Any]] | None = None,
    sample_ids_target: np.ndarray | None = None,
    groups_target: np.ndarray | None = None,
    x_val: np.ndarray | None = None,
    y_val: np.ndarray | None = None,
    family: str = "logistic",
) -> None:
    """Sweep target-region label budgets (absolute counts).

    Budget = 0 → strict geographic holdout (train only on source).
    Budget = TARGET_ID_UPPER_BOUND → target-ID upper bound (train only on
    80 % of target, test on remaining 20 %; no source used).
    Budget > 0 → sample that many *target* labels for training, keep the
    remaining target samples for testing.

    ``x_val``/``y_val`` is the regime's source-side held-out validation set, used by
    the binary probe to calibrate its threshold without peeking at the target.
    """
    meta = dict(meta or {})
    x_source_t = _apply(transform, x_source)
    x_target_t = _apply(transform, x_target_full)
    x_val_t = _apply(transform, x_val) if x_val is not None and len(x_val) else None
    sample_ids_full = np.asarray(sample_ids_target if sample_ids_target is not None else np.arange(len(y_target_full)))
    groups_full = np.asarray(groups_target) if groups_target is not None else None

    for budget in budgets:
        if budget == 0:
            sub_seed = seed
            x_tr, y_tr = x_source_t, y_source
            x_te, y_te = x_target_t, y_target_full
            n_tr = len(x_source_t)
            test_idx = np.arange(len(y_target_full))
        elif budget == TARGET_ID_UPPER_BOUND:
            sub_seed = seed + int(round(budget * 1000))
            n_target = len(y_target_full)
            train_size = max(1, min(int(n_target * 0.8), n_target - 1))
            idx = np.arange(n_target)
            strat = y_target_full if stratify else None
            try:
                train_idx, test_idx = train_test_split(
                    idx, train_size=train_size, random_state=sub_seed, stratify=strat
                )
            except ValueError:
                train_idx, test_idx = train_test_split(
                    idx, train_size=train_size, random_state=sub_seed, stratify=None
                )
            x_tr = x_target_t[train_idx]
            y_tr = y_target_full[train_idx]
            x_te = x_target_t[test_idx]
            y_te = y_target_full[test_idx]
            n_tr = len(train_idx)
        else:
            sub_seed = seed + int(round(budget * 1000))
            n_target = len(y_target_full)
            k = min(n_target - 1, max(1, int(budget)))
            idx = np.arange(n_target)
            strat = y_target_full if stratify else None
            try:
                few, _ = train_test_split(idx, train_size=k, random_state=sub_seed, stratify=strat)
            except ValueError:
                few, _ = train_test_split(idx, train_size=k, random_state=sub_seed, stratify=None)
            remaining = np.setdiff1d(idx, few)
            if len(remaining) == 0:
                few = few[:-1]
                remaining = np.array([few[-1]])
            x_tr = np.concatenate([x_source_t, x_target_t[few]])
            y_tr = np.concatenate([y_source, y_target_full[few]])
            x_te = x_target_t[remaining]
            y_te = y_target_full[remaining]
            n_tr = len(x_tr)
            test_idx = remaining

        identity = {
            "seed": seed,
            "holdout": meta.get("holdout"),
            "method": meta.get("method"),
            "budget_type": "target",
            "label_budget": budget,
        }
        perf.set_identity(identity)
        with perf.measure(f"probe.sweep.target/{meta.get('benchmark', '?')}/{meta.get('method', '?')}",
                          n_train=n_tr, n_test=len(y_te)):
            result = fit_score(x_tr, y_tr, x_te, y_te, sub_seed, x_val_t, y_val)
            if len(result) == 3:
                scores, extra, per_sample = result
            else:
                scores, extra = result
                per_sample = None
        perf.set_identity(None)
        current_groups = groups_full[test_idx] if groups_full is not None else None
        scores = {**scores, **_worst_group_scores(per_sample, current_groups)}

        rows.append({
            **meta,
            "budget_type": "target",
            "label_budget": budget,
            "seed": seed,
            "n_train_sub": n_tr,
            "n_test": len(y_te),
            **extra,
            **scores,
        })
        _append_prediction_rows(
            predictions,
            meta=meta,
            seed=seed,
            budget_type="target",
            label_budget=budget,
            n_train_sub=n_tr,
            sample_ids=sample_ids_full[test_idx],
            groups_test=current_groups,
            per_sample=per_sample,
        )


def run_probes(
    rows: list[dict[str, Any]],
    x_train: np.ndarray,
    x_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    seed: int,
    *,
    transform: FeatureTransform | None = None,
    budgets: list[float] = SOURCE_BUDGETS,
    meta: dict[str, Any] | None = None,
    groups_train: np.ndarray | None = None,
    predictions: list[dict[str, Any]] | None = None,
    sample_ids_test: np.ndarray | None = None,
    groups_test: np.ndarray | None = None,
    x_val: np.ndarray | None = None,
    y_val: np.ndarray | None = None,
    family: str = "logistic",
) -> None:
    """Binary calibrated-probe budget sweep (source-fraction budgets)."""

    def fit_score(x_tr, y_tr, x_te, y_te, probe_seed, x_cal=None, y_cal=None):
        clf, threshold, n_fit, n_cal, probe_meta = fit_probe_with_calibration(
            x_tr, y_tr, probe_seed, x_cal=x_cal, y_cal=y_cal, family=family
        )
        extra = {
            "n_probe_fit": n_fit,
            "n_probe_calibration": n_cal,
            "threshold_source": probe_meta["calibration_source"],
            "threshold": threshold,
            **probe_meta,
        }
        if transform is not None and hasattr(transform, "adapt_test_features"):
            x_te = transform.adapt_test_features(clf, x_te)
        scores, per_sample = score_binary(clf, threshold, x_te, y_te, return_per_sample=True)
        return scores, extra, per_sample

    _sweep_budgets(
        rows, x_train, x_test, y_train, y_test, seed, fit_score,
        transform=transform, budgets=budgets, meta=meta, stratify=True, groups_train=groups_train,
        predictions=predictions, sample_ids_test=sample_ids_test, groups_test=groups_test,
        x_val=x_val, y_val=y_val,
    )


def run_probes_target(
    rows: list[dict[str, Any]],
    x_source: np.ndarray,
    x_target_full: np.ndarray,
    y_source: np.ndarray,
    y_target_full: np.ndarray,
    seed: int,
    *,
    transform: FeatureTransform | None = None,
    budgets: list[int] = ALL_TARGET_BUDGETS,
    meta: dict[str, Any] | None = None,
    groups_source: np.ndarray | None = None,
    predictions: list[dict[str, Any]] | None = None,
    sample_ids_target: np.ndarray | None = None,
    groups_target: np.ndarray | None = None,
    x_val: np.ndarray | None = None,
    y_val: np.ndarray | None = None,
    family: str = "logistic",
) -> None:
    """Binary calibrated-probe *target-budget* sweep.

    Includes budgets 0 (strict geographic holdout, train on source only),
    TARGET_BUDGETS (few-shot target training), and TARGET_ID_UPPER_BOUND
    (target-ID upper bound, train on 80 % of target only).
    """

    def fit_score(x_tr, y_tr, x_te, y_te, probe_seed, x_cal=None, y_cal=None):
        clf, threshold, n_fit, n_cal, probe_meta = fit_probe_with_calibration(
            x_tr, y_tr, probe_seed, x_cal=x_cal, y_cal=y_cal, family=family
        )
        extra = {
            "n_probe_fit": n_fit,
            "n_probe_calibration": n_cal,
            "threshold_source": probe_meta["calibration_source"],
            "threshold": threshold,
            **probe_meta,
        }
        if transform is not None and hasattr(transform, "adapt_test_features"):
            x_te = transform.adapt_test_features(clf, x_te)
        scores, per_sample = score_binary(clf, threshold, x_te, y_te, return_per_sample=True)
        return scores, extra, per_sample

    _sweep_target_budgets(
        rows, x_source, x_target_full, y_source, y_target_full, seed, fit_score,
        transform=transform, budgets=budgets, meta=meta, stratify=True, groups_source=groups_source,
        predictions=predictions, sample_ids_target=sample_ids_target, groups_target=groups_target,
        x_val=x_val, y_val=y_val,
    )


def run_probes_multiclass(
    rows: list[dict[str, Any]],
    x_train: np.ndarray,
    x_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    seed: int,
    *,
    transform: FeatureTransform | None = None,
    budgets: list[float] = SOURCE_BUDGETS,
    meta: dict[str, Any] | None = None,
    groups_train: np.ndarray | None = None,
    predictions: list[dict[str, Any]] | None = None,
    sample_ids_test: np.ndarray | None = None,
    groups_test: np.ndarray | None = None,
    x_val: np.ndarray | None = None,
    y_val: np.ndarray | None = None,
    family: str = "logistic",
) -> None:
    """Multiclass logistic-probe source-budget sweep (crop-type classification).

    The multiclass probe has no threshold to calibrate, but ``x_val``/``y_val`` (the
    regime's held-out val) are used to select the L2 strength on a fixed grid.
    """

    def fit_score(x_tr, y_tr, x_te, y_te, probe_seed, x_cal=None, y_cal=None):
        clf, probe_meta = fit_probe_multiclass(x_tr, y_tr, probe_seed, x_val=x_cal, y_val=y_cal, family=family)
        if transform is not None and hasattr(transform, "adapt_test_features"):
            x_te = transform.adapt_test_features(clf, x_te)
        scores, per_sample = score_multiclass(clf, x_te, y_te, return_per_sample=True)
        return scores, probe_meta, per_sample

    _sweep_budgets(
        rows, x_train, x_test, y_train, y_test, seed, fit_score,
        transform=transform, budgets=budgets, meta=meta, stratify=True, groups_train=groups_train,
        predictions=predictions, sample_ids_test=sample_ids_test, groups_test=groups_test,
        x_val=x_val, y_val=y_val,
    )


def run_probes_multiclass_target(
    rows: list[dict[str, Any]],
    x_source: np.ndarray,
    x_target_full: np.ndarray,
    y_source: np.ndarray,
    y_target_full: np.ndarray,
    seed: int,
    *,
    transform: FeatureTransform | None = None,
    budgets: list[int] = ALL_TARGET_BUDGETS,
    meta: dict[str, Any] | None = None,
    groups_source: np.ndarray | None = None,
    predictions: list[dict[str, Any]] | None = None,
    sample_ids_target: np.ndarray | None = None,
    groups_target: np.ndarray | None = None,
    x_val: np.ndarray | None = None,
    y_val: np.ndarray | None = None,
    family: str = "logistic",
) -> None:
    """Multiclass target-budget sweep.

    Includes budgets 0 (strict geographic holdout, train on source only),
    TARGET_BUDGETS (few-shot target training), and TARGET_ID_UPPER_BOUND
    (target-ID upper bound, train on 80 % of target only). ``x_val``/``y_val`` (the
    regime's source-side val) select the L2 strength on a fixed grid.
    """

    def fit_score(x_tr, y_tr, x_te, y_te, probe_seed, x_cal=None, y_cal=None):
        clf, probe_meta = fit_probe_multiclass(x_tr, y_tr, probe_seed, x_val=x_cal, y_val=y_cal, family=family)
        if transform is not None and hasattr(transform, "adapt_test_features"):
            x_te = transform.adapt_test_features(clf, x_te)
        scores, per_sample = score_multiclass(clf, x_te, y_te, return_per_sample=True)
        return scores, probe_meta, per_sample

    _sweep_target_budgets(
        rows, x_source, x_target_full, y_source, y_target_full, seed, fit_score,
        transform=transform, budgets=budgets, meta=meta, stratify=True, groups_source=groups_source,
        predictions=predictions, sample_ids_target=sample_ids_target, groups_target=groups_target,
        x_val=x_val, y_val=y_val,
    )


def run_probes_segmentation(
    rows: list[dict[str, Any]],
    x_train: np.ndarray,
    x_val: np.ndarray,
    x_test: np.ndarray,
    y_train: np.ndarray,
    y_val: np.ndarray,
    y_test: np.ndarray,
    seed: int,
    *,
    transform: FeatureTransform | None = None,
    budgets: list[float] = SOURCE_BUDGETS,
    meta: dict[str, Any] | None = None,
    tile_ids_test: np.ndarray | None = None,
    family: str = "logistic",
) -> None:
    """PASTIS-R probe sweep using folds 1-3/4/5 as train/val/test.

    When ``tile_ids_test`` is provided, per-tile metrics (mean per-tile mIoU,
    worst-tile mIoU) are computed for the test split in addition to the global
    scores.
    """
    meta = dict(meta or {})
    x_train = _apply(transform, x_train)
    x_val = _apply(transform, x_val)
    x_test = _apply(transform, x_test)
    eval_classes = np.arange(19, dtype=np.int64)
    for budget in budgets:
        sub_seed = seed + int(round(budget * 1000))
        sub = subset_indices(y_train, budget, sub_seed, stratify=True)
        # The official val fold doubles as the hyperparameter-selection set (no test peeking).
        clf, probe_meta = fit_probe_multiclass(
            x_train[sub], y_train[sub], sub_seed, x_val=x_val, y_val=y_val, family=family
        )
        for split_name, x_eval, y_eval in (
            ("validation", x_val, y_val),
            ("test", x_test, y_test),
        ):
            row = {
                **meta,
                "evaluation_split": split_name,
                "budget_type": "source",
                "label_budget": budget,
                "seed": seed,
                "n_train_sub": int(len(sub)),
                "n_test": int(len(y_eval)),
                **probe_meta,
                **score_segmentation(clf, x_eval, y_eval, eval_classes=eval_classes),
            }
            if split_name == "test" and tile_ids_test is not None:
                row.update(
                    score_segmentation_per_tile(clf, x_eval, y_eval, tile_ids_test, eval_classes=eval_classes)
                )
            rows.append(row)
