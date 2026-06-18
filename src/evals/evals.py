"""Task-agnostic evaluation protocol: constants, condition filtering, sweep orchestration.

This module re-exports everything from ``protocol.{splits,probes}`` so that
existing callers (``from evals import evals as EV``) continue to work.

Scope:
  * shared protocol constants (holdouts, conditions, budgets, metrics)
  * condition filtering by robustness axis (``filter_conditions_by_axes``)
  * budget-sweep orchestration (``_sweep_budgets``, ``_sweep_target_budgets``)
  * per-task public runners (``run_probes*``)
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np
from sklearn.model_selection import train_test_split

from evals.protocol.probes import (  # noqa: F401 — re-exported for backward compat
    FeatureTransform,
    _apply,
    best_f1_threshold,
    expected_calibration_error,
    fit_probe_multiclass,
    fit_probe_with_calibration,
    score_binary,
    score_multiclass,
    score_segmentation,
)
from evals.protocol.splits import (  # noqa: F401 — re-exported for backward compat
    make_grouped_holdout_folds,
    make_splits,
    make_strict_holdout_splits,
    subset_indices,
)
from utils import perfutils as perf

for _thread_var in [
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
]:
    os.environ.setdefault(_thread_var, "1")


# --------------------------------------------------------------------------- #
# Protocol constants
# --------------------------------------------------------------------------- #

# CropHarvest crop/non-crop strict-holdout groups: the small, agriculturally
# meaningful datasets held out one-at-a-time for strict geographic transfer.
# Each is two-class. (Counts from data labels at setup: togo 1276, ethiopia 830,
# lem-brazil 800, rwanda 565, togo-eval 306.) LEM Brazil is the known failure
# case where locally discriminative NIR magnitude does not transfer.
STRICT_HOLDOUTS: list[str] = [
    "togo",
    "ethiopia",
    "lem-brazil",
    "rwanda",
    "togo-eval",
]

# Structured stress conditions: (name, sensor_off, temporal_drop_fraction).
# These names are the contract between extraction (src/models/*) and evaluation:
# encoders emit one embedding matrix per condition name, keyed identically.
#   sensor_off in {"none", "s2", "s1", "climate"}; temporal_drop in [0, 1).
CONDITIONS: list[tuple[str, str, float]] = [
    ("baseline", "none", 0.0),
    ("sensor_off_s2", "s2", 0.0),
    ("sensor_off_s1", "s1", 0.0),
    ("sensor_off_climate", "climate", 0.0),
    ("temporal_drop_30", "none", 0.3),
    ("temporal_drop_50", "none", 0.5),
    ("temporal_drop_70", "none", 0.7),
    ("s2_off_tdrop50", "s2", 0.5),
    ("s1_off_tdrop50", "s1", 0.5),
]

# Which robustness axes each condition exercises. Used by ACTIVE_AXES in main.py
# to filter conditions at startup. "baseline" is always included.
CONDITION_AXES: dict[str, set[str]] = {
    "baseline": set(),
    "sensor_off_s2": {"sensorial"},
    "sensor_off_s1": {"sensorial"},
    "sensor_off_climate": {"sensorial"},
    "temporal_drop_30": {"temporal"},
    "temporal_drop_50": {"temporal"},
    "temporal_drop_70": {"temporal"},
    "s2_off_tdrop50": {"sensorial", "temporal"},
    "s1_off_tdrop50": {"sensorial", "temporal"},
}


def filter_conditions_by_axes(
    conditions: list[tuple[str, str, float]],
    active_axes: list[str],
) -> list[tuple[str, str, float]]:
    """Return only conditions whose axes are a subset of *active_axes*.

    "baseline" (no axes) is always included since it is the unstressed baseline.
    A compound condition like ``s2_off_tdrop50`` requires both ``sensorial``
    and ``temporal`` to be active.
    """
    axes_set = set(active_axes)
    return [c for c in conditions if CONDITION_AXES.get(c[0], set()).issubset(axes_set)]


# Which robustness axes each split regime exercises. Used by ACTIVE_AXES in main.py
# to filter split regimes at startup. "random_id" is always included.
SPLIT_AXES: dict[str, set[str]] = {
    "random_id": set(),
    "grouped_ood": {"geographic"},
    "geographic_ood": {"geographic"},
}


def filter_split_regimes_by_axes(
    split_regimes: list[str],
    active_axes: list[str],
) -> list[str]:
    """Return only split regimes whose required axes are a subset of *active_axes*.

    ``random_id`` (no axes) is always included since it is the in-distribution
    baseline. ``grouped_ood`` and ``geographic_ood`` require a ``geographic``
    robustness axis to be active.
    """
    axes_set = set(active_axes)
    return [r for r in split_regimes if SPLIT_AXES.get(r, set()).issubset(axes_set)]


# Sparse-label probe budgets.
#
# SOURCE_BUDGETS  — fraction of available source-pool labels (secondary diagnostic).
#                   Answers: "Does more source-region training data help geographic
#                   generalization?"
# TARGET_BUDGETS — absolute count of target-region labels used for training (main
#                   experiment). 0  = strict geographic holdout (zero-shot transfer);
#                   5–50 = few-shot target adaptation.
SOURCE_BUDGETS: list[float] = [0.05, 0.10, 0.25, 1.00]
TARGET_BUDGETS: list[int] = [5, 10, 25, 50]

# Reported metrics per task family. calibrated_* use a source-validation threshold
# rather than 0.5 -- default-0.5 F1 misrepresents transfer under distribution shift.
METRICS: list[str] = [
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
METRICS_BINARY: list[str] = METRICS
METRICS_MULTICLASS: list[str] = [
    "macro_f1",
    "weighted_f1",
    "balanced_accuracy",
    "accuracy",
    "macro_auc",
]
METRICS_SEGMENTATION: list[str] = ["miou", "pixel_accuracy", "macro_f1", "weighted_f1"]

def condition_names() -> list[str]:
    return [name for name, _, _ in CONDITIONS]


# --------------------------------------------------------------------------- #
# Shared budget sweep + per-task runners
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
) -> None:
    """Apply the fitted transform, then sweep source-fraction budgets, appending one
    row each.

    ``fit_score(x_tr_sub, y_tr_sub, x_test, y_test, probe_seed) -> (scores, extra)``
    is the only task-specific piece; everything else (transform application,
    budget sub-sampling, metadata, bookkeeping) is shared.
    """
    meta = dict(meta or {})
    x_train = _apply(transform, x_train)
    x_test = _apply(transform, x_test)
    for budget in budgets:
        sub_seed = seed + int(round(budget * 1000))
        if transform is not None and hasattr(transform, "subset_indices"):
            sub = transform.subset_indices(y_train, groups_train, budget, sub_seed)
        else:
            sub = subset_indices(y_train, budget, sub_seed, stratify=stratify)

        identity = {
            "seed": seed,
            "holdout": meta.get("holdout"),
            "condition": meta.get("condition"),
            "method": meta.get("method"),
            "budget_type": "source",
            "label_budget": budget,
        }
        perf.set_identity(identity)
        with perf.measure(f"probe.sweep.source/{meta.get('task', '?')}/{meta.get('method', '?')}",
                          n_train=len(sub), n_test=len(y_test)):
            result = fit_score(
                x_train[sub], y_train[sub], x_test, y_test, seed + int(round(budget * 1000)) + 17
            )
            if len(result) == 3:
                scores, extra, per_sample = result
            else:
                scores, extra = result
                per_sample = None
        perf.set_identity(None)

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
) -> None:
    """Sweep target-region label budgets (absolute counts).

    Budget = 0 → strict geographic holdout (train only on source).
    Budget > 0 → sample that many *target* labels for training, keep the
    remaining target samples for testing.
    """
    meta = dict(meta or {})
    x_source_t = _apply(transform, x_source)
    x_target_t = _apply(transform, x_target_full)

    for budget in budgets:
        if budget == 0:
            sub_seed = seed
            x_tr, y_tr = x_source_t, y_source
            x_te, y_te = x_target_t, y_target_full
            n_tr = len(x_source_t)
            test_idx = np.arange(len(y_target_full))
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
            "condition": meta.get("condition"),
            "method": meta.get("method"),
            "budget_type": "target",
            "label_budget": budget,
        }
        perf.set_identity(identity)
        with perf.measure(f"probe.sweep.target/{meta.get('task', '?')}/{meta.get('method', '?')}",
                          n_train=n_tr, n_test=len(y_te)):
            result = fit_score(x_tr, y_tr, x_te, y_te, sub_seed)
            if len(result) == 3:
                scores, extra, per_sample = result
            else:
                scores, extra = result
                per_sample = None
        perf.set_identity(None)

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
        sample_ids_full = np.asarray(sample_ids_target if sample_ids_target is not None else np.arange(len(y_target_full)))
        groups_full = np.asarray(groups_target) if groups_target is not None else None
        _append_prediction_rows(
            predictions,
            meta=meta,
            seed=seed,
            budget_type="target",
            label_budget=budget,
            n_train_sub=n_tr,
            sample_ids=sample_ids_full[test_idx],
            groups_test=groups_full[test_idx] if groups_full is not None else None,
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
) -> None:
    """Binary calibrated-probe budget sweep (source-fraction budgets)."""

    def fit_score(x_tr: np.ndarray, y_tr: np.ndarray, x_te: np.ndarray, y_te: np.ndarray, probe_seed: int):
        clf, threshold, n_fit, n_cal, probe_meta = fit_probe_with_calibration(x_tr, y_tr, probe_seed)
        extra = {
            "n_probe_fit": n_fit,
            "n_probe_calibration": n_cal,
            "threshold_source": "source_validation",
            "threshold": threshold,
            **probe_meta,
        }
        if transform is not None and hasattr(transform, "adapt_test_features"):
            x_te = transform.adapt_test_features(clf, x_te)
        if predictions is not None:
            scores, per_sample = score_binary(clf, threshold, x_te, y_te, return_per_sample=True)
            return scores, extra, per_sample
        return score_binary(clf, threshold, x_te, y_te), extra

    _sweep_budgets(
        rows, x_train, x_test, y_train, y_test, seed, fit_score,
        transform=transform, budgets=budgets, meta=meta, stratify=True, groups_train=groups_train,
        predictions=predictions, sample_ids_test=sample_ids_test, groups_test=groups_test,
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
    budgets: list[int] = TARGET_BUDGETS,
    meta: dict[str, Any] | None = None,
    groups_source: np.ndarray | None = None,
    predictions: list[dict[str, Any]] | None = None,
    sample_ids_target: np.ndarray | None = None,
    groups_target: np.ndarray | None = None,
) -> None:
    """Binary calibrated-probe *target-budget* sweep."""

    def fit_score(x_tr: np.ndarray, y_tr: np.ndarray, x_te: np.ndarray, y_te: np.ndarray, probe_seed: int):
        clf, threshold, n_fit, n_cal, probe_meta = fit_probe_with_calibration(x_tr, y_tr, probe_seed)
        extra = {
            "n_probe_fit": n_fit,
            "n_probe_calibration": n_cal,
            "threshold_source": "source_validation",
            "threshold": threshold,
            **probe_meta,
        }
        if transform is not None and hasattr(transform, "adapt_test_features"):
            x_te = transform.adapt_test_features(clf, x_te)
        if predictions is not None:
            scores, per_sample = score_binary(clf, threshold, x_te, y_te, return_per_sample=True)
            return scores, extra, per_sample
        return score_binary(clf, threshold, x_te, y_te), extra

    _sweep_target_budgets(
        rows, x_source, x_target_full, y_source, y_target_full, seed, fit_score,
        transform=transform, budgets=budgets, meta=meta, stratify=True, groups_source=groups_source,
        predictions=predictions, sample_ids_target=sample_ids_target, groups_target=groups_target,
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
) -> None:
    """Multiclass logistic-probe source-budget sweep (crop-type classification)."""

    def fit_score(x_tr: np.ndarray, y_tr: np.ndarray, x_te: np.ndarray, y_te: np.ndarray, probe_seed: int):
        clf, probe_meta = fit_probe_multiclass(x_tr, y_tr, probe_seed)
        if transform is not None and hasattr(transform, "adapt_test_features"):
            x_te = transform.adapt_test_features(clf, x_te)
        if predictions is not None:
            scores, per_sample = score_multiclass(clf, x_te, y_te, return_per_sample=True)
            return scores, probe_meta, per_sample
        return score_multiclass(clf, x_te, y_te), probe_meta

    _sweep_budgets(
        rows, x_train, x_test, y_train, y_test, seed, fit_score,
        transform=transform, budgets=budgets, meta=meta, stratify=True, groups_train=groups_train,
        predictions=predictions, sample_ids_test=sample_ids_test, groups_test=groups_test,
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
    budgets: list[int] = TARGET_BUDGETS,
    meta: dict[str, Any] | None = None,
    groups_source: np.ndarray | None = None,
    predictions: list[dict[str, Any]] | None = None,
    sample_ids_target: np.ndarray | None = None,
    groups_target: np.ndarray | None = None,
) -> None:
    """Multiclass target-budget sweep."""

    def fit_score(x_tr: np.ndarray, y_tr: np.ndarray, x_te: np.ndarray, y_te: np.ndarray, probe_seed: int):
        clf, probe_meta = fit_probe_multiclass(x_tr, y_tr, probe_seed)
        if transform is not None and hasattr(transform, "adapt_test_features"):
            x_te = transform.adapt_test_features(clf, x_te)
        if predictions is not None:
            scores, per_sample = score_multiclass(clf, x_te, y_te, return_per_sample=True)
            return scores, probe_meta, per_sample
        return score_multiclass(clf, x_te, y_te), probe_meta

    _sweep_target_budgets(
        rows, x_source, x_target_full, y_source, y_target_full, seed, fit_score,
        transform=transform, budgets=budgets, meta=meta, stratify=True, groups_source=groups_source,
        predictions=predictions, sample_ids_target=sample_ids_target, groups_target=groups_target,
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
) -> None:
    """PASTIS-R linear-probe sweep using folds 1-3/4/5 as train/val/test."""
    meta = dict(meta or {})
    x_train = _apply(transform, x_train)
    x_val = _apply(transform, x_val)
    x_test = _apply(transform, x_test)
    eval_classes = np.arange(19, dtype=np.int64)
    for budget in budgets:
        sub_seed = seed + int(round(budget * 1000))
        sub = subset_indices(y_train, budget, sub_seed, stratify=True)
        clf, probe_meta = fit_probe_multiclass(x_train[sub], y_train[sub], sub_seed)
        for split_name, x_eval, y_eval in (
            ("validation", x_val, y_val),
            ("test", x_test, y_test),
        ):
            rows.append(
                {
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
            )
