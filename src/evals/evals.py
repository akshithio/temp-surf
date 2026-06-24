"""Benchmark-agnostic evaluation protocol: constants, sweep orchestration.

Scope:
  * shared protocol constants (holdouts, budgets, metrics)
  * budget-sweep orchestration (``_sweep_budgets``, ``_sweep_target_budgets``)
  * per-benchmark public runners (``run_probes*``)
"""

from __future__ import annotations

import importlib
import os
from typing import Any

import numpy as np

from evals.probes import (  # noqa: F401
    FeatureTransform,
    _apply,
    expected_calibration_error,
    fit_probe_multiclass,
    fit_probe_with_calibration,
    score_binary,
    score_multiclass,
)
from evals.regimes import base as regime_base

# Split-regime constructors now live with their regimes (evals/regimes/<regime>.py);
# re-exported here so existing callers (EV.make_splits, ...) keep working.
from evals.regimes.geographic_ood import make_strict_holdout_splits  # noqa: F401
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


subset_indices = perf.subset_indices
_target_budget_count = perf._target_budget_count
_budget_seed = perf._budget_seed
_sweep_budgets = perf._sweep_budgets
_sweep_target_budgets = perf._sweep_target_budgets


_value_counts = regime_base._value_counts
_split_manifest_entry = regime_base._split_manifest_entry
_dense_fold_stats = regime_base._dense_fold_stats
_segmentation_split_manifest_entry = regime_base._segmentation_split_manifest_entry
_write_split_manifest = regime_base._write_split_manifest

def _validate_source_budgets(budgets: list[float | int]) -> None:
    bad = [budget for budget in budgets if float(budget) <= 0.0 or float(budget) > 1.0]
    if bad:
        raise ValueError(f"source budgets must be fractions in (0, 1]; invalid: {bad}")


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
    _validate_source_budgets(source)
    if not any(float(b) == 0.0 for b in target):
        raise ValueError("target budgets must include 0 for the strict source-only OOD anchor.")
    return source, target


# --------------------------------------------------------------------------- #
# Protocol constants
# --------------------------------------------------------------------------- #
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
# Target-budget sweep: sample N target labels for training, rest stays as test
# --------------------------------------------------------------------------- #


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
    _validate_source_budgets(budgets)

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
        x_val=x_val, y_val=y_val, target_id_budget=TARGET_ID_UPPER_BOUND,
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
    _validate_source_budgets(budgets)

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
        x_val=x_val, y_val=y_val, target_id_budget=TARGET_ID_UPPER_BOUND,
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
    bench = load_benchmark("pastis")
    bench.run_probes_segmentation(
        rows, x_train, x_val, y_train, y_val, seed, eval_streams=eval_streams, transform=transform,
        budgets=budgets, meta=meta, family=family,
    )


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
    bench = load_benchmark("pastis")
    bench.run_probes_segmentation_target(
        rows, x_source, y_source, seed, target_patches=target_patches, sample_target=sample_target,
        stream_target=stream_target, x_val=x_val, y_val=y_val, transform=transform, budgets=budgets,
        meta=meta, family=family, target_id_budget=TARGET_ID_UPPER_BOUND,
    )

# --------------------------------------------------------------------------- #
# Pair-level execution
# --------------------------------------------------------------------------- #

def load_benchmark(benchmark_name: str):
    return importlib.import_module(f"evals.benchmarks.{benchmark_name}")


def build_methods(label_kind: str, seed: int):
    """name -> (cls_or_none, kwargs). Only plain ERM for now."""
    return {"erm": (None, {})}


def _id_source_budget(source_budgets: list[float | int]) -> float | int:
    """Prefer the explicit full-source anchor when choosing the ID row for deltas."""
    for budget in source_budgets:
        if abs(float(budget) - 1.0) < 1e-9:
            return budget
    return max(source_budgets)

