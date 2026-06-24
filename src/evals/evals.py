"""Benchmark-agnostic evaluation protocol: constants, sweep orchestration.

Scope:
  * shared protocol constants (holdouts, budgets, metrics)
  * budget-sweep orchestration (``_sweep_budgets``, ``_sweep_target_budgets``)
  * per-benchmark public runners (``run_probes*``)
"""

from __future__ import annotations

import importlib
import os
import warnings
from typing import Any

import numpy as np
from joblib import Parallel, delayed
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import train_test_split

from evals import compat
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
    score_segmentation_streamed,
)
from evals.regimes import base as regime_base

# Split-regime constructors now live with their regimes (evals/regimes/<regime>.py);
# re-exported here so existing callers (EV.make_splits, ...) keep working.
from evals.regimes.geographic_ood import make_strict_holdout_splits  # noqa: F401
from evals.regimes.random_id import make_splits  # noqa: F401
from utils import cacheutils, runstate
from utils import ioutils as IOU
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
    if budget >= 1.0 or len(idx) < 2:
        return idx  # nothing to subsample (and a <2 pool would make k=0 crash train_test_split)
    k = min(len(idx) - 1, max(2, int(round(budget * len(idx)))))
    strat = y if stratify else None
    try:
        sub, _ = train_test_split(idx, train_size=k, random_state=seed, stratify=strat)
    except ValueError:
        sub, _ = train_test_split(idx, train_size=k, random_state=seed, stratify=None)
    return np.sort(sub)


def _source_fit_budget(budget: float | int) -> float:
    """Source-sweep budget ``0`` is the full-source anchor, not a zero-label fit."""
    return 1.0 if float(budget) == 0.0 else float(budget)


def _target_budget_count(budget: float | int, pool_size: int) -> int:
    """Target budget values in (0, 1) are fractions; values >=1 are absolute counts."""
    b = float(budget)
    if 0.0 < b < 1.0:
        return max(1, int(round(b * pool_size)))
    return max(1, int(budget))


def _budget_lists(budget_regimes: dict[str, list[float | int]] | None) -> tuple[list[float | int], list[float | int]]:
    """Return source and target budget lists, defaulting to the protocol constants."""
    if budget_regimes is None:
        return list(SOURCE_BUDGETS), list(ALL_TARGET_BUDGETS)
    missing = {"source", "target"} - set(budget_regimes)
    if missing:
        raise ValueError(f"BUDGET_REGIMES missing required key(s): {sorted(missing)}")
    source = list(budget_regimes["source"])
    target = list(budget_regimes["target"])
    if not source or not target:
        raise ValueError("BUDGET_REGIMES source and target lists must both be non-empty.")
    return source, target


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
TARGET_BUDGETS: list[int | float] = [5, 10, 25, 50]

# Sentinel for the in-distribution target upper bound:
#   Train on 80% of the target region (no source), test on the remaining 20%.
#   Provides the "how good can we get on this region if trained in-distribution?"
#   baseline, used to separate transfer loss from inherent regional difficulty.
TARGET_ID_UPPER_BOUND: int = -1

# Extended target budgets including the strict geographic holdout (0) and the
# target-ID upper bound (-1).
ALL_TARGET_BUDGETS: list[int | float] = [0, *TARGET_BUDGETS, TARGET_ID_UPPER_BOUND]

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
# Shared-vs-unseen-class decomposition (see probes.score_multiclass): isolates representation loss
# (shared_*) from target-only label-support mismatch (unseen_prevalence / n_classes_unseen).
METRICS_MULTICLASS_SHARED: list[str] = [
    "shared_macro_f1",
    "shared_balanced_accuracy",
    "shared_accuracy",
    "unseen_prevalence",
    "n_classes_unseen",
    "n_classes_seen",
]
METRICS_MULTICLASS: list[str] = [
    *METRICS_MULTICLASS_BASE, *METRICS_MULTICLASS_WORST_GROUP, *METRICS_MULTICLASS_SHARED,
]
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
        "diagnostic": [
            "weighted_f1", "accuracy", "macro_auc",
            # representation-only (shared classes) vs the target-only support mismatch that the
            # full-label gap also absorbs -- report these alongside, never collapsed into the gap.
            "shared_macro_f1", "shared_balanced_accuracy", "shared_accuracy",
            "unseen_prevalence", "n_classes_unseen", "n_classes_seen",
        ],
    },
    "segmentation": {
        "deployment": ["miou", "mean_per_tile_miou", "worst_tile_miou"],
        "diagnostic": ["pixel_accuracy", "macro_f1", "weighted_f1", "n_tiles_scored"],
    },
}

# --------------------------------------------------------------------------- #
# Shared budget sweep + per-benchmark runners
# --------------------------------------------------------------------------- #

def _budget_seed(seed: int, budget: float) -> int:
    """Non-negative per-budget seed. ``abs`` guards the target-ID-upper-bound budget
    (-1), where ``seed + round(budget*1000)`` would go negative and crash RNG/sklearn."""
    return abs(seed + int(round(budget * 1000)))


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
    evaluation_split: str = "held_out",
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
        "evaluation_split": evaluation_split,
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

    This captures the "average can hide a bad deployment subgroup" view. For
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
        sub_seed = _budget_seed(seed, budget)
        fit_budget = _source_fit_budget(budget)
        if transform is not None and hasattr(transform, "subset_indices"):
            sub = transform.subset_indices(y_train, groups_train, fit_budget, sub_seed)
        else:
            sub = subset_indices(y_train, fit_budget, sub_seed, stratify=stratify)

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
    budgets: list[int | float],
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
    """Sweep target-region label budgets (absolute counts) on a SHARED fixed target test set.

    A single 80/20 target split is drawn once (budget-independent): the 20% is the fixed test
    set every budget is scored on, the 80% is the train pool with a fixed nested ordering.

    Budget = 0 → strict geographic holdout (train on source only).
    Budget > 0 → add the first ``budget`` target labels from the nested ordering to the source
                 training pool (so the few-shot label sets are nested).
    Budget = TARGET_ID_UPPER_BOUND → target-ID upper bound: train on the FULL 80% pool (no
                 source), HP tuned + refit internally on the pool (source-free).

    ``x_val``/``y_val`` is the source-side regime val, used to calibrate/select for the
    zero-shot and few-shot budgets without peeking at the target; the oracle tunes on an
    internal target val instead, so it uses no source labels.
    """
    meta = dict(meta or {})
    x_source_t = _apply(transform, x_source)
    x_target_t = _apply(transform, x_target_full)
    x_val_t = _apply(transform, x_val) if x_val is not None and len(x_val) else None
    sample_ids_full = np.asarray(sample_ids_target if sample_ids_target is not None else np.arange(len(y_target_full)))
    groups_full = np.asarray(groups_target) if groups_target is not None else None
    n_target = len(y_target_full)

    # ONE fixed target test split + ONE fixed nested ordering of the train pool, shared by EVERY
    # budget. Consequences: (a) zero-shot, few-shot and the oracle are all scored on the SAME
    # target test set, so the inherent-difficulty decomposition compares like-with-like; (b) the
    # few-shot label sets are NESTED (budget 10's labels ⊇ budget 5's), so the curve isolates
    # "more target labels" instead of confounding it with a fresh random draw + a fresh test set.
    # The split seed is budget-independent.
    split_seed = _budget_seed(seed, 0.5)
    idx = np.arange(n_target)
    degenerate = n_target < 5
    if degenerate:
        pool_idx = test_idx = idx
    else:
        test_size = max(1, min(int(round(0.2 * n_target)), n_target - 1))
        strat = y_target_full if stratify else None
        try:
            pool_idx, test_idx = train_test_split(idx, test_size=test_size, random_state=split_seed, stratify=strat)
        except ValueError:
            pool_idx, test_idx = train_test_split(idx, test_size=test_size, random_state=split_seed, stratify=None)
    order = np.random.default_rng(split_seed).permutation(pool_idx)  # nested few-shot ordering
    x_te, y_te = x_target_t[test_idx], y_target_full[test_idx]

    for budget in budgets:
        if budget != 0 and degenerate:
            continue  # too few target samples for a held-out train/test split
        cal_x, cal_y = x_val_t, y_val  # source val for zero-shot / few-shot (no target peeking)
        tune_internal = False
        if budget == 0:
            sub_seed = seed
            x_tr, y_tr = x_source_t, y_source
            n_tr = len(x_source_t)
        elif budget == TARGET_ID_UPPER_BOUND:
            sub_seed = _budget_seed(seed, budget)
            # oracle: train on the WHOLE 80% pool (not 64%). HP is tuned and the model REFIT on
            # the full pool via an internal target val, so it stays source-free.
            x_tr, y_tr = x_target_t[order], y_target_full[order]
            cal_x, cal_y = None, None
            tune_internal = True
            n_tr = len(order)
        else:
            sub_seed = _budget_seed(seed, budget)
            k = min(len(order), _target_budget_count(budget, len(order)))
            few = order[:k]  # nested prefix of the fixed ordering
            x_tr = np.concatenate([x_source_t, x_target_t[few]])
            y_tr = np.concatenate([y_source, y_target_full[few]])
            n_tr = len(x_tr)

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
            result = fit_score(x_tr, y_tr, x_te, y_te, sub_seed, cal_x, cal_y, tune_internal)
            score_fitted = None
            if len(result) == 4:
                scores, extra, per_sample, score_fitted = result
            elif len(result) == 3:
                scores, extra, per_sample = result
            else:
                scores, extra = result
                per_sample = None
        perf.set_identity(None)
        current_groups = groups_full[test_idx] if groups_full is not None else None
        scores = {**scores, **_worst_group_scores(per_sample, current_groups)}

        # Every budget is scored on the fixed held-out 20% (the matched set for the few-shot curve
        # and the inherent-difficulty decomposition).
        rows.append({
            **meta, "budget_type": "target", "label_budget": budget, "evaluation_split": "held_out",
            "seed": seed, "n_train_sub": n_tr, "n_test": len(y_te), **extra, **scores,
        })
        _append_prediction_rows(
            predictions, meta=meta, seed=seed, budget_type="target", label_budget=budget,
            n_train_sub=n_tr, sample_ids=sample_ids_full[test_idx], groups_test=current_groups,
            per_sample=per_sample, evaluation_split="held_out",
        )

        # Budget 0 ALSO emits a FULL-target zero-shot anchor (train source only, evaluate on the
        # WHOLE target domain): this is the PRIMARY deployment OOD estimand (compute_deltas reads
        # it), restoring the full-domain estimate the fixed-20% split would otherwise shrink.
        if budget == 0:
            full_idx = np.arange(len(y_target_full))
            if score_fitted is not None:
                scores_f, per_sample_f = score_fitted(x_target_t, y_target_full)
                extra_f = extra
            else:
                res_full = fit_score(x_source_t, y_source, x_target_t, y_target_full, sub_seed, cal_x, cal_y)
                if len(res_full) == 3:
                    scores_f, extra_f, per_sample_f = res_full
                else:
                    scores_f, extra_f = res_full
                    per_sample_f = None
            groups_all = groups_full if groups_full is not None else None
            scores_f = {**scores_f, **_worst_group_scores(per_sample_f, groups_all)}
            rows.append({
                **meta, "budget_type": "target", "label_budget": budget, "evaluation_split": "full",
                "seed": seed, "n_train_sub": len(x_source_t), "n_test": len(y_target_full),
                **extra_f, **scores_f,
            })
            _append_prediction_rows(
                predictions, meta=meta, seed=seed, budget_type="target", label_budget=budget,
                n_train_sub=len(x_source_t), sample_ids=sample_ids_full[full_idx], groups_test=groups_all,
                per_sample=per_sample_f, evaluation_split="full",
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

    def fit_score(x_tr, y_tr, x_te, y_te, probe_seed, x_cal=None, y_cal=None, tune_internal=False):
        clf, threshold, n_fit, n_cal, probe_meta = fit_probe_with_calibration(
            x_tr, y_tr, probe_seed, x_cal=x_cal, y_cal=y_cal, family=family, tune_internal=tune_internal
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
    budgets: list[int | float] = ALL_TARGET_BUDGETS,
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

    def fit_score(x_tr, y_tr, x_te, y_te, probe_seed, x_cal=None, y_cal=None, tune_internal=False):
        clf, threshold, n_fit, n_cal, probe_meta = fit_probe_with_calibration(
            x_tr, y_tr, probe_seed, x_cal=x_cal, y_cal=y_cal, family=family, tune_internal=tune_internal
        )
        extra = {
            "n_probe_fit": n_fit,
            "n_probe_calibration": n_cal,
            "threshold_source": probe_meta["calibration_source"],
            "threshold": threshold,
            **probe_meta,
        }

        def score_fitted(x_eval, y_eval):
            if transform is not None and hasattr(transform, "adapt_test_features"):
                x_eval = transform.adapt_test_features(clf, x_eval)
            return score_binary(clf, threshold, x_eval, y_eval, return_per_sample=True)

        scores, per_sample = score_fitted(x_te, y_te)
        return scores, extra, per_sample, score_fitted

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

    def fit_score(x_tr, y_tr, x_te, y_te, probe_seed, x_cal=None, y_cal=None, tune_internal=False):
        clf, probe_meta = fit_probe_multiclass(
            x_tr, y_tr, probe_seed, x_val=x_cal, y_val=y_cal, family=family, tune_internal=tune_internal
        )
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
    budgets: list[int | float] = ALL_TARGET_BUDGETS,
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

    def fit_score(x_tr, y_tr, x_te, y_te, probe_seed, x_cal=None, y_cal=None, tune_internal=False):
        clf, probe_meta = fit_probe_multiclass(
            x_tr, y_tr, probe_seed, x_val=x_cal, y_val=y_cal, family=family, tune_internal=tune_internal
        )

        def score_fitted(x_eval, y_eval):
            if transform is not None and hasattr(transform, "adapt_test_features"):
                x_eval = transform.adapt_test_features(clf, x_eval)
            return score_multiclass(clf, x_eval, y_eval, return_per_sample=True)

        scores, per_sample = score_fitted(x_te, y_te)
        return scores, probe_meta, per_sample, score_fitted

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
    y_train: np.ndarray,
    y_val: np.ndarray,
    seed: int,
    *,
    eval_streams: dict[str, Any],
    transform: FeatureTransform | None = None,
    budgets: list[float] = SOURCE_BUDGETS,
    meta: dict[str, Any] | None = None,
    family: str = "logistic",
) -> None:
    """PASTIS-R source-fraction sweep, scored on the FULL evaluation fold (every pixel).

    The probe is fit on a capped training pixel SAMPLE (``x_train``), with the val fold doubling as
    the HP-selection set (``x_val``; no test peeking), then scored by streaming every valid pixel of
    each evaluation fold. ``eval_streams`` maps an evaluation-split name (``validation``/``test``) to
    a zero-arg callable returning a fresh ``(features, labels)`` tile iterator
    (``cacheutils.iter_dense_tiles``). Full-fold scoring yields the exact fold mIoU and a credible
    worst-tile mIoU — not the noisy estimate a capped pixel sample gives.
    """
    meta = dict(meta or {})
    x_train = _apply(transform, x_train)
    x_val = _apply(transform, x_val)
    eval_classes = np.arange(19, dtype=np.int64)
    for budget in budgets:
        sub_seed = _budget_seed(seed, budget)
        sub = subset_indices(y_train, _source_fit_budget(budget), sub_seed, stratify=True)
        clf, probe_meta = fit_probe_multiclass(
            x_train[sub], y_train[sub], sub_seed, x_val=x_val, y_val=y_val, family=family
        )
        for split_name, tiles in eval_streams.items():
            rows.append({
                **meta,
                "evaluation_split": split_name,
                "budget_type": "source",
                "label_budget": budget,
                "seed": seed,
                "n_train_sub": int(len(sub)),
                **probe_meta,
                **score_segmentation_streamed(clf, tiles(), eval_classes, transform=transform),
            })


def run_probes_segmentation_target(
    rows: list[dict[str, Any]],
    x_source: np.ndarray,
    y_source: np.ndarray,
    seed: int,
    *,
    target_patches: Any,
    sample_target: Any,
    stream_target: Any,
    x_val: np.ndarray,
    y_val: np.ndarray,
    transform: FeatureTransform | None = None,
    budgets: list[int | float] = ALL_TARGET_BUDGETS,
    meta: dict[str, Any] | None = None,
    family: str = "logistic",
) -> None:
    """Patch-level dense few-shot / oracle curve for PASTIS-R, scored on the FULL held-out target.

    The target-label unit is the ORIGINAL 128x128 PATCH. ONE fixed 80/20 split over
    ``target_patches`` is drawn (fixed nested ordering of the 80% pool, fixed 20% test); every budget
    trains a probe and is scored on ALL pixels of the held-out 20% patches (streamed). Splitting by
    PATCH keeps a patch's four 64x64 child tiles together (no cross-tile leakage), and full-fold
    scoring avoids the sparse-sample noise a capped pixel sample would give.

    ``sample_target(patch_ids, seed)`` returns a capped TRAINING pixel sample restricted to those
    patches (used for the few-shot additions and the oracle pool); ``stream_target(patch_ids)`` yields
    full ``(features, labels)`` tiles for SCORING. Budget ``0`` is zero-shot (source only); few-shot
    adds the first ``k`` pool patches' sampled pixels to the source pool; ``TARGET_ID_UPPER_BOUND``
    (-1) is the target-ID oracle (trains on the 80% pool, HP tuned + refit internally — source-free).
    ``x_val``/``y_val`` (the source val fold) selects the probe HP for zero-shot / few-shot.
    """
    meta = dict(meta or {})
    x_source = _apply(transform, x_source)
    eval_classes = np.arange(19, dtype=np.int64)
    patches = np.array(sorted({int(p) for p in target_patches}))
    degenerate = len(patches) < 2
    split_rng = np.random.default_rng(_budget_seed(seed, 0.5))
    perm = split_rng.permutation(patches)
    n_test_patches = max(1, int(round(0.2 * len(patches)))) if not degenerate else len(patches)
    test_patches = set(perm[:n_test_patches].tolist())
    pool_order = [int(p) for p in perm.tolist() if p not in test_patches]  # nested 80% pool ordering

    for budget in budgets:
        if budget != 0 and (degenerate or not pool_order):
            continue
        sub_seed = _budget_seed(seed, budget)
        cal_x, cal_y, tune_internal = x_val, y_val, False  # source val for zero-shot / few-shot
        if budget == 0:
            x_tr, y_tr = x_source, y_source
        elif budget == TARGET_ID_UPPER_BOUND:
            xo, yo = sample_target(set(pool_order), sub_seed)[:2]  # capped sample of the 80% pool
            x_tr, y_tr = _apply(transform, xo), yo
            cal_x, cal_y, tune_internal = None, None, True  # source-free: tune + refit on internal pool split
        else:
            k = min(len(pool_order), _target_budget_count(budget, len(pool_order)))
            xf, yf = sample_target(set(pool_order[:k]), sub_seed)[:2]  # nested prefix of the pool
            x_tr = np.concatenate([x_source, _apply(transform, xf)])
            y_tr = np.concatenate([y_source, yf])
        clf, probe_meta = fit_probe_multiclass(
            x_tr, y_tr, sub_seed, x_val=cal_x, y_val=cal_y, family=family, tune_internal=tune_internal
        )
        rows.append({
            **meta,
            "evaluation_split": "test",
            "budget_type": "target",
            "label_budget": budget,
            "seed": seed,
            "n_train_sub": int(len(y_tr)),
            **probe_meta,
            **score_segmentation_streamed(clf, stream_target(test_patches), eval_classes, transform=transform),
        })


# --------------------------------------------------------------------------- #
# Pair-level execution
# --------------------------------------------------------------------------- #


def load_benchmark(benchmark_name: str):
    return importlib.import_module(f"evals.benchmarks.{benchmark_name}")


def _effective_n_jobs(embeddings) -> int:
    """Concurrent probe fits: use 85 percent of available cores, bounded by memory."""
    cpu_count = os.cpu_count() or 1
    requested = max(1, int(0.85 * cpu_count))
    arrays = embeddings.values() if hasattr(embeddings, "values") else [embeddings]
    max_bytes = max((np.asarray(arr).nbytes for arr in arrays), default=0)
    budget = int(os.environ.get("PROBE_COPY_BUDGET_BYTES", str(8_000_000_000)))
    if max_bytes > 0:
        requested = max(1, min(requested, budget // max_bytes))
    return requested


def build_methods(label_kind: str, seed: int):
    """name -> (cls_or_none, kwargs). Only plain ERM for now."""
    return {"erm": (None, {})}


def _value_counts(values: np.ndarray) -> dict[str, int]:
    """Stable JSON-friendly counts for class/domain labels."""
    arr = np.asarray(values)
    if arr.size == 0:
        return {}
    labels, counts = np.unique(arr.astype(str), return_counts=True)
    return {str(label): int(count) for label, count in zip(labels, counts, strict=True)}


def _split_manifest_entry(
    *,
    model_name: str,
    benchmark_name: str,
    seed: int,
    split_regime: str,
    domain_basis: str,
    holdout: str,
    train: np.ndarray,
    val: np.ndarray,
    test: np.ndarray,
    domains: np.ndarray,
    labels: np.ndarray,
) -> dict[str, Any]:
    """Describe one tabular split with domain and class composition."""

    def part(prefix: str, idx: np.ndarray) -> dict[str, Any]:
        idx = np.asarray(idx, dtype=np.int64)
        return {
            f"n_{prefix}": int(len(idx)),
            f"{prefix}_domains": sorted({str(v) for v in np.asarray(domains)[idx].tolist()}),
            f"{prefix}_domain_counts": _value_counts(np.asarray(domains)[idx]),
            f"{prefix}_class_counts": _value_counts(np.asarray(labels)[idx]),
        }

    row: dict[str, Any] = {
        "model": model_name,
        "benchmark": benchmark_name,
        "seed": int(seed),
        "split_regime": split_regime,
        "domain_basis": domain_basis,
        "holdout": str(holdout),
    }
    row.update(part("train", train))
    row.update(part("val", val))
    row.update(part("test", test))
    return row


def _dense_fold_stats(emb_dir, folds: set[int]) -> dict[str, Any]:
    """Exact dense label/domain stats for cached PASTIS fold partitions."""
    class_counts: dict[str, int] = {}
    domain_counts: dict[str, int] = {}
    n_tiles = 0
    patches: set[int] = set()
    for fold in sorted(folds):
        fold_dir = emb_dir / f"fold_{int(fold)}"
        for label_path in sorted(fold_dir.glob("*.labels.npy")):
            labels = np.asarray(np.load(label_path, mmap_mode="r"), dtype=np.int64)
            n_tiles += 1
            patch_id = int(label_path.name.split("_", 1)[0])
            patches.add(patch_id)
            domain_counts[str(int(fold))] = domain_counts.get(str(int(fold)), 0) + int(len(labels))
            for label, count in _value_counts(labels).items():
                class_counts[label] = class_counts.get(label, 0) + int(count)
    return {
        "n": int(sum(domain_counts.values())),
        "n_tiles": int(n_tiles),
        "n_patches": int(len(patches)),
        "domains": [str(int(fold)) for fold in sorted(folds)],
        "domain_counts": domain_counts,
        "class_counts": class_counts,
    }


def _segmentation_split_manifest_entry(
    *,
    model_name: str,
    benchmark_name: str,
    seed: int,
    split_regime: str,
    holdout: str,
    train_folds: set[int],
    val_folds: set[int],
    test_folds: set[int],
    emb_dir,
) -> dict[str, Any]:
    """Describe one dense fold split with exact cached-pixel class counts."""

    def add_part(row: dict[str, Any], prefix: str, stats: dict[str, Any]) -> None:
        row[f"n_{prefix}"] = stats["n"]
        row[f"n_{prefix}_tiles"] = stats["n_tiles"]
        row[f"n_{prefix}_patches"] = stats["n_patches"]
        row[f"{prefix}_domains"] = stats["domains"]
        row[f"{prefix}_domain_counts"] = stats["domain_counts"]
        row[f"{prefix}_class_counts"] = stats["class_counts"]

    row: dict[str, Any] = {
        "model": model_name,
        "benchmark": benchmark_name,
        "seed": int(seed),
        "split_regime": split_regime,
        "domain_basis": "geography",
        "holdout": str(holdout),
    }
    add_part(row, "train", _dense_fold_stats(emb_dir, train_folds))
    add_part(row, "val", _dense_fold_stats(emb_dir, val_folds))
    add_part(row, "test", _dense_fold_stats(emb_dir, test_folds))
    return row


def _write_split_manifest(results_dir, rows: list[dict[str, Any]]) -> None:
    """Write the split audit artifact beside the probe outputs."""
    IOU.write_json(results_dir / "split_manifest.json", {"splits": rows})


def _id_source_budget(source_budgets: list[float | int]) -> float | int:
    """Prefer the explicit full-source anchor when choosing the ID row for deltas."""
    for budget in source_budgets:
        if abs(float(budget) - 1.0) < 1e-9:
            return budget
    for budget in source_budgets:
        if abs(float(budget)) < 1e-9:
            return budget
    return max(source_budgets)


def _probe_cell(
    probe_fn,
    emb,
    train,
    val,
    test,
    y,
    groups,
    cls,
    kwargs,
    uses_target,
    meta,
    seed,
    family="logistic",
    budgets=None,
) -> tuple[list[dict], list[dict]]:
    """Fit the optional method transform and run the source-budget probe."""
    x_tr, x_cond_te = emb[train], emb[test]
    y_tr, y_te, g_tr = y[train], y[test], groups[train]
    x_val = emb[val] if len(val) else None
    y_val = y[val] if len(val) else None
    mname = meta.get("method", "?")
    identity = {k: meta[k] for k in ("seed", "holdout", "method") if k in meta}
    transform = None
    if cls is not None:
        x_paired = x_cond_te if uses_target else None
        transform = cls(**kwargs)
        with perf.measure(f"method.fit/{mname}", identity=identity, n_samples=len(train), n_features=x_tr.shape[1]):
            transform.fit(x_tr, y_tr, g_tr, x_paired=x_paired)
    rows: list[dict] = []
    perf.set_identity(identity)
    with perf.measure(
        f"probe.run/{meta.get('benchmark', '?')}/{mname}",
        n_samples_train=len(train),
        n_samples_test=len(test),
        n_features=x_tr.shape[1],
    ):
        preds: list[dict] = []
        probe_fn(
            rows,
            x_tr,
            x_cond_te,
            y_tr,
            y_te,
            seed,
            transform=transform,
            meta=meta,
            groups_train=g_tr,
            predictions=preds,
            sample_ids_test=np.asarray(test),
            groups_test=np.asarray(groups)[test],
            x_val=x_val,
            y_val=y_val,
            family=family,
            **({} if budgets is None else {"budgets": budgets}),
        )
    perf.set_identity(None)
    return rows, preds


def _probe_cell_target(
    probe_fn,
    emb,
    train,
    val,
    test,
    y,
    groups,
    cls,
    kwargs,
    uses_target,
    meta,
    seed,
    family="logistic",
    budgets=None,
) -> tuple[list[dict], list[dict]]:
    """Target-budget variant using the full target pool."""
    x_source_tr, x_target_full = emb[train], emb[test]
    y_source_tr, y_target_full = y[train], y[test]
    g_source_tr = groups[train]
    x_val = emb[val] if len(val) else None
    y_val = y[val] if len(val) else None
    mname = meta.get("method", "?")
    identity = {k: meta[k] for k in ("seed", "holdout", "method") if k in meta}
    transform = None
    if cls is not None:
        x_paired = x_target_full if uses_target else None
        transform = cls(**kwargs)
        with perf.measure(
            f"method.fit/{mname}", identity=identity, n_samples=len(train), n_features=x_source_tr.shape[1]
        ):
            transform.fit(x_source_tr, y_source_tr, g_source_tr, x_paired=x_paired)
    rows: list[dict] = []
    perf.set_identity(identity)
    with perf.measure(
        f"probe.target/{meta.get('benchmark', '?')}/{mname}",
        n_samples_source=len(train),
        n_samples_target=len(test),
        n_features=x_source_tr.shape[1],
    ):
        preds: list[dict] = []
        probe_fn(
            rows,
            x_source_tr,
            x_target_full,
            y_source_tr,
            y_target_full,
            seed,
            transform=transform,
            meta=meta,
            groups_source=g_source_tr,
            predictions=preds,
            sample_ids_target=np.asarray(test),
            groups_target=np.asarray(groups)[test],
            x_val=x_val,
            y_val=y_val,
            family=family,
            **({} if budgets is None else {"budgets": budgets}),
        )
    perf.set_identity(None)
    return rows, preds


def _run_segmentation_pair(
    benchmark_name,
    model_name,
    seeds,
    max_samples,
    max_dense_pixels,
    split_regimes,
    run_stages,
    active_probes,
    budget_regimes,
    overwrite_mode,
    enc_kwargs,
) -> None:
    """Run dense PASTIS-R execution over fold-based regimes."""
    stages = runstate.validate_run_stages(run_stages)
    gen_embeddings = "gen_embeddings" in stages
    probing = "probing" in stages
    source_budgets, target_budgets = _budget_lists(budget_regimes)
    bench_mod = load_benchmark(benchmark_name)
    bench_kwargs = dict(max_samples=max_samples, shuffle=True, seed=0)
    tag = cacheutils.bench_tag(bench_mod.BENCHMARK, bench_kwargs)
    perf.reset()
    bench = cacheutils.cached_bench(bench_mod.BENCHMARK, tag, **bench_kwargs)
    if gen_embeddings:
        cacheutils.extract_dense_and_cache(
            bench,
            bench_mod.BENCHMARK,
            model_name,
            tag,
            overwrite=overwrite_mode,
            **enc_kwargs,
        )
    emb_dir = (
        cacheutils.require_dense_cache(bench, bench_mod.BENCHMARK, model_name, tag, enc_kwargs.get("weights_path"))
        if probing
        else cacheutils.dense_embedding_cache_dir(bench, bench_mod.BENCHMARK, model_name, tag, enc_kwargs.get("weights_path"))
    )
    results_dir = cacheutils.OUTPUT_DIR / "results" / model_name / benchmark_name
    if not probing:
        n_events = perf.write_log(results_dir / "perf.jsonl")
        print(f"  embedding stage complete; perf: {n_events} events logged", flush=True)
        return
    signature = runstate.run_signature(
        model_name,
        tag,
        split_regimes,
        seeds,
        enc_kwargs,
        active_probes=active_probes,
        budget_regimes=budget_regimes,
        max_samples=max_samples,
        max_dense_pixels=max_dense_pixels,
    )
    runstate.check_run_signature(results_dir, signature, overwrite_mode=overwrite_mode)
    rows_path = results_dir / "probe_results.jsonl"
    if overwrite_mode:
        for path in (
            rows_path,
            results_dir / "probe_results.csv",
            results_dir / "summary.csv",
            results_dir / "deltas.csv",
            results_dir / "split_manifest.json",
            results_dir / "run_signature.txt",
        ):
            if path.exists():
                path.unlink()
    runstate.publish_run_signature(results_dir, signature)
    rows = IOU.read_jsonl(rows_path)
    fam_fields = ("seed", "method", "split_regime", "holdout", "probe_family")

    def _fam_key(r):
        return tuple(r.get(k) for k in fam_fields)

    present_by_family: dict[tuple, set] = {}
    for r in rows:
        present_by_family.setdefault(_fam_key(r), set()).add(
            (r.get("budget_type"), r.get("label_budget"), r.get("evaluation_split"))
        )
    expected_source = {("source", b, s) for b in source_budgets for s in ("validation", "test")}
    expected_target = {("target", b, "test") for b in target_budgets}

    def _expected(regime):
        exp = set(expected_source)
        if regime == "geographic_ood":
            exp |= expected_target
        return exp

    regime_idx = fam_fields.index("split_regime")
    done_families = {
        k for k, seen in present_by_family.items() if _expected(k[regime_idx]).issubset(seen)
    }
    incomplete = set(present_by_family) - done_families
    if incomplete:
        rows = [r for r in rows if _fam_key(r) not in incomplete]
        rows_path.unlink(missing_ok=True)
        IOU.append_jsonl(rows_path, rows)

    supported = getattr(bench_mod, "SPLIT_REGIMES", ["random_id"])
    regimes = [r for r in supported if r in split_regimes]
    fold_configs = list(regime_base.segmentation_fold_configs(bench_mod, regimes, overwrite_mode=overwrite_mode))
    _write_split_manifest(
        results_dir,
        [
            _segmentation_split_manifest_entry(
                model_name=model_name,
                benchmark_name=benchmark_name,
                seed=seed,
                split_regime=split_regime,
                holdout=holdout,
                train_folds=set(train_folds),
                val_folds=set(val_folds),
                test_folds=set(test_folds),
                emb_dir=emb_dir,
            )
            for seed in seeds
            for split_regime, holdout, train_folds, val_folds, test_folds in fold_configs
        ],
    )
    for seed in seeds:
        for method_name, (cls, kwargs) in build_methods(bench_mod.LABEL_KIND, seed).items():
            for split_regime, holdout, train_folds, val_folds, test_folds in fold_configs:
                families_to_run = [
                    f for f in active_probes
                    if (seed, method_name, split_regime, holdout, f) not in done_families
                ]
                if not families_to_run:
                    continue
                x_train, y_train, groups_train, _, _ = cacheutils.load_dense_samples(
                    emb_dir, train_folds, max_dense_pixels, seed
                )
                x_val, y_val, _, _, _ = cacheutils.load_dense_samples(
                    emb_dir, val_folds, max_dense_pixels, seed + 10_000
                )
                transform = None
                if cls is not None:
                    transform = cls(**kwargs)
                    transform.fit(x_train, y_train, groups_train, x_paired=None)
                for family in families_to_run:
                    cell_rows: list[dict] = []
                    seg_meta = {
                        "model": model_name,
                        "benchmark": bench_mod.BENCHMARK,
                        "method": method_name,
                        "split_regime": split_regime,
                        "domain_basis": "geography",
                        "holdout": holdout,
                        "probe_family": family,
                    }
                    run_probes_segmentation(
                        cell_rows,
                        x_train,
                        x_val,
                        y_train,
                        y_val,
                        seed,
                        eval_streams={
                            "validation": lambda vf=val_folds: cacheutils.iter_dense_tiles(emb_dir, vf),
                            "test": lambda tf=test_folds: cacheutils.iter_dense_tiles(emb_dir, tf),
                        },
                        transform=transform,
                        budgets=source_budgets,
                        meta=seg_meta,
                        family=family,
                    )
                    if split_regime == "geographic_ood":
                        run_probes_segmentation_target(
                            cell_rows,
                            x_train,
                            y_train,
                            seed,
                            target_patches=cacheutils.dense_fold_patches(emb_dir, test_folds),
                            sample_target=lambda pids, sd, tf=test_folds: cacheutils.load_dense_samples(
                                emb_dir, tf, max_dense_pixels, sd, patch_ids=pids
                            ),
                            stream_target=lambda pids, tf=test_folds: cacheutils.iter_dense_tiles(
                                emb_dir, tf, patch_ids=pids
                            ),
                            x_val=x_val,
                            y_val=y_val,
                            transform=transform,
                            budgets=target_budgets,
                            meta=seg_meta,
                            family=family,
                        )
                    IOU.append_jsonl(rows_path, cell_rows)
                    rows.extend(cell_rows)
    IOU.write_csv(results_dir / "probe_results.csv", rows)
    summary = IOU.summarize_rows(
        rows,
        keys=[
            "model",
            "method",
            "probe_family",
            "split_regime",
            "holdout",
            "evaluation_split",
            "budget_type",
            "label_budget",
        ],
        metrics=METRICS_SEGMENTATION,
    )
    IOU.write_csv(results_dir / "summary.csv", summary)
    IOU.write_json(results_dir / "metric_roles.json", METRIC_ROLES["segmentation"])
    deltas = IOU.compute_deltas(rows, METRICS_SEGMENTATION, id_source_budget=_id_source_budget(source_budgets))
    IOU.write_csv(results_dir / "deltas.csv", deltas)
    declared = set(compat.input_modalities(model_name))
    available = {"s2", "s1", "time"}
    IOU.write_json(results_dir / "model_inputs.json", {
        "model": model_name, "benchmark": benchmark_name, "s2_only_mode": False,
        "declared_modalities": sorted(declared),
        "available_modalities": sorted(available),
        "effective_modalities": sorted(declared & available),
    })
    perf.write_log(results_dir / "perf.jsonl")


def _run_tabular_pair(
    benchmark_name,
    model_name,
    seeds,
    max_samples,
    max_dense_pixels,
    split_regimes,
    run_stages,
    active_probes,
    budget_regimes,
    overwrite_mode,
    enc_kwargs,
) -> None:
    stages = runstate.validate_run_stages(run_stages)
    gen_embeddings = "gen_embeddings" in stages
    probing = "probing" in stages
    source_budgets, target_budgets = _budget_lists(budget_regimes)
    bench_mod = load_benchmark(benchmark_name)
    probe_fn_src, metrics = {
        "binary": (run_probes, METRICS_BINARY),
        "multiclass": (run_probes_multiclass, METRICS_MULTICLASS),
    }[bench_mod.LABEL_KIND]
    probe_fn_tgt, _ = {
        "binary": (run_probes_target, METRICS_BINARY),
        "multiclass": (run_probes_multiclass_target, METRICS_MULTICLASS),
    }[bench_mod.LABEL_KIND]
    holdouts = bench_mod.HOLDOUTS
    supported = getattr(bench_mod, "SPLIT_REGIMES", split_regimes)
    split_regimes = [r for r in split_regimes if r in supported]

    bench_kwargs = dict(max_samples=max_samples, shuffle=True, seed=0)
    tag = cacheutils.bench_tag(bench_mod.BENCHMARK, bench_kwargs)
    perf.reset()
    bench = cacheutils.cached_bench(bench_mod.BENCHMARK, tag, **bench_kwargs)
    s2_only = os.environ.get("RB_S2_ONLY", "").strip().lower() not in ("", "0", "false", "no")
    suffix = "__s2only" if s2_only else ""
    if s2_only:
        bench = bench.s2_only()
    emb_tag = tag + suffix
    y, _native_groups = bench_mod.make_targets(bench)
    if gen_embeddings:
        emb = cacheutils.extract_and_cache(
            bench, bench_mod.BENCHMARK, model_name, emb_tag, overwrite=overwrite_mode, **enc_kwargs
        )
    else:
        emb = cacheutils.load_cached_embeddings(
            bench, bench_mod.BENCHMARK, model_name, emb_tag, enc_kwargs.get("weights_path")
        )

    results_dir = cacheutils.OUTPUT_DIR / "results" / model_name / (benchmark_name + suffix)
    if not probing:
        n_events = perf.write_log(results_dir / "perf.jsonl")
        print(f"  embedding stage complete; perf: {n_events} events logged", flush=True)
        return
    signature = runstate.run_signature(
        model_name,
        emb_tag,
        split_regimes,
        seeds,
        enc_kwargs,
        active_probes=active_probes,
        budget_regimes=budget_regimes,
        max_samples=max_samples,
        max_dense_pixels=max_dense_pixels,
    )
    runstate.check_run_signature(results_dir, signature, overwrite_mode=overwrite_mode)
    rows_path = results_dir / "probe_results.jsonl"
    preds_path = results_dir / "predictions.jsonl"

    if overwrite_mode:
        for p in [
            rows_path,
            preds_path,
            results_dir / "probe_results.csv",
            results_dir / "summary.csv",
            results_dir / "deltas.csv",
            results_dir / "split_manifest.json",
            results_dir / "run_signature.txt",
        ]:
            if p.exists():
                p.unlink()
    runstate.publish_run_signature(results_dir, signature)

    rows = IOU.read_jsonl(rows_path)
    done = {
        (
            r.get("seed"),
            r.get("split_regime"),
            r.get("holdout"),
            r.get("method"),
            r.get("probe_family"),
            r.get("budget_type"),
            r.get("label_budget"),
            r.get("evaluation_split"),
        )
        for r in rows
    }

    jobs = []

    def uses_target_flag(cls):
        return getattr(cls, "USES_TARGET", False)

    def _scopes(budget_type, b):
        if budget_type == "target":
            return ("full", "held_out") if b == 0 else ("held_out",)
        return (None,)

    def _missing(base, budget_type, expected):
        return [
            b for b in expected
            if not all((*base, budget_type, b, sc) in done for sc in _scopes(budget_type, b))
        ]

    rerun_keys: set = set()
    split_specs: list[tuple] = []
    split_manifest: list[dict[str, Any]] = []

    for seed in seeds:
        for split_regime in split_regimes:
            for split_label, train, val, test, groups, has_target, domain_basis in regime_base.iter_splits(
                split_regime,
                bench,
                y,
                holdouts,
                seed,
                overwrite_mode=overwrite_mode,
                val_group=getattr(bench_mod, "VAL_HOLDOUT", None),
            ):
                split_specs.append((seed, split_regime, split_label, train, val, test, groups, has_target, domain_basis))
                split_manifest.append(
                    _split_manifest_entry(
                        model_name=model_name,
                        benchmark_name=benchmark_name,
                        seed=seed,
                        split_regime=split_regime,
                        domain_basis=domain_basis,
                        holdout=split_label,
                        train=train,
                        val=val,
                        test=test,
                        domains=groups,
                        labels=y,
                    )
                )
    _write_split_manifest(results_dir, split_manifest)

    for seed in seeds:
        seed_split_specs = [spec for spec in split_specs if spec[0] == seed]
        for mname, (cls, kwargs) in build_methods(bench_mod.LABEL_KIND, seed).items():
            for _, split_regime, split_label, train, val, test, groups, has_target, domain_basis in seed_split_specs:
                for family in active_probes:
                    meta = {
                        "model": model_name,
                        "benchmark": bench_mod.BENCHMARK,
                        "method": mname,
                        "split_regime": split_regime,
                        "domain_basis": domain_basis,
                        "holdout": split_label,
                        "probe_family": family,
                    }
                    base = (seed, split_regime, split_label, mname, family)
                    if has_target:
                        todo = _missing(base, "target", target_budgets)
                        if todo:
                            rerun_keys.update((*base, "target", b) for b in todo)
                            jobs.append(
                                delayed(_probe_cell_target)(
                                    probe_fn_tgt,
                                    emb,
                                    train,
                                    val,
                                    test,
                                    y,
                                    groups,
                                    cls,
                                    kwargs,
                                    uses_target_flag(cls),
                                    {**meta, "budget_type": "target"},
                                    seed,
                                    family,
                                    todo,
                                )
                            )
                    todo_src = _missing(base, "source", source_budgets)
                    if todo_src:
                        rerun_keys.update((*base, "source", b) for b in todo_src)
                        jobs.append(
                            delayed(_probe_cell)(
                                probe_fn_src,
                                emb,
                                train,
                                val,
                                test,
                                y,
                                groups,
                                cls,
                                kwargs,
                                uses_target_flag(cls),
                                {**meta, "budget_type": "source"},
                                seed,
                                family,
                                todo_src,
                            )
                        )

    rows = runstate.prune_partial_budgets(rows, rows_path, preds_path, rerun_keys)

    if jobs:
        n_jobs = _effective_n_jobs(emb)
        print(f"  probe jobs={len(jobs)} n_jobs={n_jobs}", flush=True)
        for cell_rows, cell_preds in Parallel(n_jobs=n_jobs, return_as="generator", prefer="threads")(jobs):
            if cell_preds:
                IOU.append_jsonl(preds_path, cell_preds)
            IOU.append_jsonl(rows_path, cell_rows)
            rows.extend(cell_rows)

    IOU.write_csv(results_dir / "probe_results.csv", rows)
    summary = IOU.summarize_rows(
        rows,
        keys=["model", "method", "probe_family", "split_regime", "domain_basis", "budget_type",
              "label_budget", "evaluation_split"],
        metrics=metrics,
    )
    IOU.write_csv(results_dir / "summary.csv", summary)
    IOU.write_json(results_dir / "metric_roles.json", METRIC_ROLES[bench_mod.LABEL_KIND])

    deltas = IOU.compute_deltas(
        rows, metrics, predictions=IOU.read_jsonl(preds_path), id_source_budget=_id_source_budget(source_budgets)
    )
    IOU.write_csv(results_dir / "deltas.csv", deltas)

    from evals import confounds
    axes = {"geography": np.asarray(bench.groups), "class": np.asarray(y)}
    if getattr(bench, "years", None) is not None:
        axes["year"] = np.asarray(bench.years)
    try:
        from dataio.koppen import koppen_main_group
        axes["climate"] = koppen_main_group(np.asarray(bench.latlon))
    except Exception:
        pass
    IOU.write_json(results_dir / "domain_confounds.json", confounds.domain_confound_report(axes))

    declared = set(compat.input_modalities(model_name))
    available = bench.available_modalities()
    IOU.write_json(results_dir / "model_inputs.json", {
        "model": model_name, "benchmark": benchmark_name, "s2_only_mode": s2_only,
        "declared_modalities": sorted(declared),
        "available_modalities": sorted(available),
        "effective_modalities": sorted(declared & available),
    })

    perf_path = results_dir / "perf.jsonl"
    n_events = perf.write_log(perf_path)
    print(f"  perf: {n_events} events logged to {perf_path}", flush=True)


def run_pair(
    *,
    benchmark_name,
    model_name,
    seeds,
    max_samples,
    max_dense_pixels,
    split_regimes,
    run_stages,
    active_probes,
    budget_regimes,
    overwrite_mode,
    enc_kwargs,
) -> None:
    """Run one configured model/benchmark pair."""
    bench_mod = load_benchmark(benchmark_name)
    if bench_mod.LABEL_KIND == "segmentation":
        _run_segmentation_pair(
            benchmark_name,
            model_name,
            seeds,
            max_samples,
            max_dense_pixels,
            split_regimes,
            run_stages,
            active_probes,
            budget_regimes,
            overwrite_mode,
            enc_kwargs,
        )
        return
    _run_tabular_pair(
        benchmark_name,
        model_name,
        seeds,
        max_samples,
        max_dense_pixels,
        split_regimes,
        run_stages,
        active_probes,
        budget_regimes,
        overwrite_mode,
        enc_kwargs,
    )
