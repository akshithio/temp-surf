"""Shared, task-agnostic evaluation protocol for frozen-embedding robustness work.

This project is embedding-centric: encoders in ``src/models/*`` extract frozen
embeddings (one matrix per stress condition), robustness methods in
``src/methods/*`` are fitted feature transforms, and the per-task drivers in
``src/evals/tasks/*`` call the primitives here. Nothing in this module touches a
model, a raw dataset, or a corruption op -- it operates only on already-extracted
embedding matrices.

Scope:
  * split construction (random upper-bound split + strict geographic holdout)
  * sparse-label budget sub-sampling
  * the calibrated binary linear probe with an optional fitted feature-transform hook
  * shared protocol constants (holdouts, conditions, budgets, metrics)

Out of scope (lives elsewhere now):
  * embedding extraction + stress corruption .......... src/models/*
  * CropHarvest / EuroCropsML loading ................. src/io/get_input.py
  * multiclass / regression scoring ................... src/evals/tasks/*
"""

from __future__ import annotations

import os
import warnings
from typing import Any, Protocol

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression, Ridge

from utils import perf
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

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
    ("clean", "none", 0.0),
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
# to filter conditions at startup. "clean" is always included.
CONDITION_AXES: dict[str, set[str]] = {
    "clean": set(),
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

    "clean" (no axes) is always included since it is the unstressed baseline.
    A compound condition like ``s2_off_tdrop50`` requires both ``sensorial``
    and ``temporal`` to be active.
    """
    axes_set = set(active_axes)
    return [c for c in conditions if CONDITION_AXES.get(c[0], set()).issubset(axes_set)]

# Sparse-label probe budgets.
#
# SOURCE_BUDGETS  — fraction of available source-pool labels (secondary diagnostic).
#                   Answers: "Does more source-region training data help geographic
#                   generalization?"
# TARGET_BUDGETS — absolute count of target-region labels used for training (main
#                   experiment). 0  = strict geographic holdout (zero-shot transfer);
#                   5–50 = few-shot target adaptation.
SOURCE_BUDGETS: list[float] = [0.01, 0.05, 0.10, 0.25, 1.00]
TARGET_BUDGETS: list[int] = [0, 5, 10, 25, 50]

# Reported metrics per task family. calibrated_* use a source-validation threshold
# rather than 0.5 -- default-0.5 F1 misrepresents transfer under distribution shift.
METRICS: list[str] = [
    "f1",
    "auc",
    "balanced_accuracy",
    "calibrated_f1",
    "calibrated_balanced_accuracy",
    "ece",
]
METRICS_BINARY: list[str] = METRICS
METRICS_MULTICLASS: list[str] = [
    "macro_f1",
    "weighted_f1",
    "balanced_accuracy",
    "accuracy",
    "macro_auc",
]
METRICS_REGRESSION: list[str] = [
    "rmse",
    "mae",
    "r2",
    "pearson_r",
    "spearman_r",
]

PROBE_SOLVER = "liblinear"  # binary calibrated probe solver
PROBE_MULTICLASS_SOLVER = "lbfgs"  # multinomial logistic for crop-type
PROBE_MAX_ITER = 20000
PROBE_TOL = 1e-5
RIDGE_ALPHA = 1.0  # regression probe (phenology proxy)


def condition_names() -> list[str]:
    return [name for name, _, _ in CONDITIONS]


# --------------------------------------------------------------------------- #
# Feature-transform hook
# --------------------------------------------------------------------------- #


class FeatureTransform(Protocol):
    """A fitted robustness method: a pure map on the frozen embedding space.

    Methods (src/methods/*) implement their own fitting (using invariant pairs,
    unlabeled target features, group labels, ...). By the time a transform reaches
    ``run_probes`` it is already fitted; the probe just applies ``transform`` to
    train and test features identically. ``None`` means the ERM baseline.
    """

    def transform(self, x: np.ndarray) -> np.ndarray: ...


def _apply(transform: FeatureTransform | None, x: np.ndarray) -> np.ndarray:
    return x if transform is None else transform.transform(x)


# --------------------------------------------------------------------------- #
# Splits
# --------------------------------------------------------------------------- #


def make_splits(y: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """80/10/10 stratified train/val/test on the full pool (the easy upper bound)."""
    idx = np.arange(len(y))
    train_val, test = train_test_split(idx, test_size=0.10, random_state=seed, stratify=y)
    train, val = train_test_split(
        train_val,
        test_size=0.1111111111,
        random_state=seed + 1,
        stratify=y[train_val],
    )
    return np.sort(train), np.sort(val), np.sort(test)


def make_strict_holdout_splits(
    y: np.ndarray,
    groups: np.ndarray,
    heldout_group: str,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Strict geographic holdout: the entire heldout group is the test set.

    Returns (train, val, test, train_val). The probe never sees the target region.
    Strict *SSL* exclusion (the encoder also never pretrained on this region) is
    enforced upstream at extraction time, not here.
    """
    idx = np.arange(len(y))
    test = idx[groups == heldout_group]
    train_val = idx[groups != heldout_group]
    if len(test) == 0:
        raise ValueError(f"No samples found for strict holdout group: {heldout_group}")
    if len(np.unique(y[test])) < 2:
        raise ValueError(f"Strict holdout group is one-class: {heldout_group}")
    if len(np.unique(y[train_val])) < 2:
        raise ValueError(f"Strict holdout training pool is one-class after excluding: {heldout_group}")
    try:
        train, val = train_test_split(train_val, test_size=0.10, random_state=seed, stratify=y[train_val])
    except ValueError:
        # multiclass pools can have singleton classes that break stratification
        train, val = train_test_split(train_val, test_size=0.10, random_state=seed, stratify=None)
    return np.sort(train), np.sort(val), np.sort(test), np.sort(train_val)


def subset_indices(y: np.ndarray, budget: float, seed: int, stratify: bool = True) -> np.ndarray:
    """Sub-sample of indices for a sparse-label budget.

    Stratified for classification; ``stratify=False`` for regression (continuous
    targets). Falls back to non-stratified if a class is too small to stratify at
    the requested budget (expected for multiclass at tiny budgets -- unseen-class
    drops are part of the EuroCropsML transfer story).
    """
    idx = np.arange(len(y))
    if budget >= 1.0:
        return idx
    # Use an absolute count floored at 2, so tiny budgets on small pools (e.g. SICKLE)
    # don't round to a train_size of 0. For large pools this matches budget*N.
    k = min(len(idx) - 1, max(2, int(round(budget * len(idx)))))
    strat = y if stratify else None
    try:
        sub, _ = train_test_split(idx, train_size=k, random_state=seed, stratify=strat)
    except ValueError:
        sub, _ = train_test_split(idx, train_size=k, random_state=seed, stratify=None)
    return np.sort(sub)


# --------------------------------------------------------------------------- #
# Calibrated binary probe
# --------------------------------------------------------------------------- #


def best_f1_threshold(y_true: np.ndarray, prob: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return 0.5
    candidates = np.unique(prob)
    if len(candidates) > 256:
        candidates = np.quantile(prob, np.linspace(0.01, 0.99, 199))
    best_threshold = 0.5
    best_score = -1.0
    for threshold in candidates:
        pred = (prob >= threshold).astype(np.int64)
        score = float(f1_score(y_true, pred, zero_division=0))
        if score > best_score:
            best_score = score
            best_threshold = float(threshold)
    return best_threshold


def fit_probe_with_calibration(
    x_train: np.ndarray,
    y_train: np.ndarray,
    seed: int,
) -> tuple[Any, float, int, int, dict[str, Any]]:
    """Fit a standardized logistic probe and a source-validation F1 threshold."""
    idx = np.arange(len(y_train))
    if len(idx) >= 20 and len(np.unique(y_train)) == 2 and min(np.bincount(y_train)) >= 4:
        fit_idx, cal_idx = train_test_split(
            idx,
            test_size=0.20,
            random_state=seed,
            stratify=y_train,
        )
    else:
        fit_idx = idx
        cal_idx = idx
    n_classes = len(np.unique(y_train))
    with perf.measure(
        "probe.fit/binary",
        n_samples=len(fit_idx), n_features=x_train.shape[1], n_classes=n_classes,
    ):
        clf = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                max_iter=PROBE_MAX_ITER,
                class_weight="balanced",
                solver=PROBE_SOLVER,
                tol=PROBE_TOL,
                random_state=seed,
            ),
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            clf.fit(x_train[fit_idx], y_train[fit_idx])
    convergence_warnings = [warning for warning in caught if issubclass(warning.category, ConvergenceWarning)]
    logistic = clf.named_steps["logisticregression"]
    n_iter = int(np.max(logistic.n_iter_)) if hasattr(logistic, "n_iter_") else -1
    probe_meta = {
        "probe_solver": PROBE_SOLVER,
        "probe_max_iter": PROBE_MAX_ITER,
        "probe_tol": PROBE_TOL,
        "probe_n_iter": n_iter,
        "probe_converged": int(len(convergence_warnings) == 0),
        "probe_convergence_warnings": len(convergence_warnings),
        "probe_warning_message": str(convergence_warnings[0].message) if convergence_warnings else "",
    }
    cal_prob = clf.predict_proba(x_train[cal_idx])[:, 1]
    threshold = best_f1_threshold(y_train[cal_idx], cal_prob)
    return clf, threshold, int(len(fit_idx)), int(len(cal_idx)), probe_meta


def expected_calibration_error(y_true: np.ndarray, prob: np.ndarray, n_bins: int = 10) -> float:
    """ECE: mean absolute gap between accuracy and confidence per bin, weighted by bin size.

    .. math::
        ECE = \\sum_{b} \\frac{|B_b|}{n} \\, |\\text{acc}(B_b) - \\text{conf}(B_b)|
    """
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_indices = np.digitize(prob, bin_edges[1:-1])
    ece = 0.0
    for b in range(n_bins):
        in_bin = bin_indices == b
        if not in_bin.any():
            continue
        acc = y_true[in_bin].mean()
        conf = prob[in_bin].mean()
        ece += in_bin.sum() * abs(acc - conf)
    return float(ece / len(y_true))


def score_binary(clf: Any, threshold: float, x_test: np.ndarray, y_test: np.ndarray) -> dict[str, float]:
    """Default-threshold and calibrated-threshold binary metrics."""
    with perf.measure("probe.score/binary", n_samples=len(y_test), n_features=x_test.shape[1]):
        pred = clf.predict(x_test)
        prob = clf.predict_proba(x_test)[:, 1]
        calibrated_pred = (prob >= threshold).astype(np.int64)
    scores = {
        "f1": float(f1_score(y_test, pred, zero_division=0)),
        "auc": float(roc_auc_score(y_test, prob)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, pred)),
        "calibrated_f1": float(f1_score(y_test, calibrated_pred, zero_division=0)),
        "calibrated_balanced_accuracy": float(balanced_accuracy_score(y_test, calibrated_pred)),
        "ece": expected_calibration_error(y_test, prob),
    }
    n_classes = len(np.unique(y_test))
    perf.log_static("probe.macs/binary", macs=x_test.shape[1] * n_classes, n_samples=len(y_test))
    return scores


# --------------------------------------------------------------------------- #
# Multiclass probe (crop-type, e.g. EuroCropsML)
# --------------------------------------------------------------------------- #


def fit_probe_multiclass(x_train: np.ndarray, y_train: np.ndarray, seed: int) -> tuple[Any, dict[str, Any]]:
    """Fit a standardized multinomial logistic probe (no threshold calibration)."""
    with perf.measure(
        "probe.fit/multiclass",
        n_samples=len(y_train), n_features=x_train.shape[1], n_classes=len(np.unique(y_train)),
    ):
        clf = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                max_iter=PROBE_MAX_ITER,
                class_weight="balanced",
                solver=PROBE_MULTICLASS_SOLVER,
                tol=PROBE_TOL,
                random_state=seed,
            ),
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            clf.fit(x_train, y_train)
    convergence_warnings = [w for w in caught if issubclass(w.category, ConvergenceWarning)]
    logistic = clf.named_steps["logisticregression"]
    n_iter = int(np.max(logistic.n_iter_)) if hasattr(logistic, "n_iter_") else -1
    probe_meta = {
        "probe_solver": PROBE_MULTICLASS_SOLVER,
        "probe_max_iter": PROBE_MAX_ITER,
        "probe_n_iter": n_iter,
        "probe_converged": int(len(convergence_warnings) == 0),
        "probe_convergence_warnings": len(convergence_warnings),
        "n_classes_train": int(len(logistic.classes_)),
    }
    return clf, probe_meta


def score_multiclass(clf: Any, x_test: np.ndarray, y_test: np.ndarray) -> dict[str, float]:
    with perf.measure("probe.score/multiclass", n_samples=len(y_test), n_features=x_test.shape[1]):
        pred = clf.predict(x_test)
        try:
            proba = clf.predict_proba(x_test)
            macro_auc = float(
                roc_auc_score(y_test, proba, multi_class="ovr", average="macro", labels=clf.classes_)
            )
        except (ValueError, AttributeError):
            macro_auc = float("nan")
    n_classes = len(getattr(clf, "classes_", []))
    perf.log_static("probe.macs/multiclass", macs=x_test.shape[1] * max(n_classes, 1), n_samples=len(y_test))
    return {
        "macro_f1": float(f1_score(y_test, pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_test, pred, average="weighted", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, pred)),
        "accuracy": float(accuracy_score(y_test, pred)),
        "macro_auc": macro_auc,
    }


# --------------------------------------------------------------------------- #
# Regression probe (phenology proxy, e.g. NDVI peak timing)
# --------------------------------------------------------------------------- #


def fit_probe_regression(x_train: np.ndarray, y_train: np.ndarray, seed: int) -> tuple[Any, dict[str, Any]]:  # noqa: ARG001
    """Fit a standardized ridge regression probe."""
    with perf.measure("probe.fit/regression", n_samples=len(y_train), n_features=x_train.shape[1]):
        clf = make_pipeline(StandardScaler(), Ridge(alpha=RIDGE_ALPHA))
        clf.fit(x_train, y_train)
    return clf, {"probe_solver": "ridge", "ridge_alpha": RIDGE_ALPHA}


def _rank(a: np.ndarray) -> np.ndarray:
    return np.argsort(np.argsort(a)).astype(np.float64)


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if len(a) < 2 or a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def score_regression(clf: Any, x_test: np.ndarray, y_test: np.ndarray) -> dict[str, float]:
    with perf.measure("probe.score/regression", n_samples=len(y_test), n_features=x_test.shape[1]):
        pred = clf.predict(x_test)
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_test, pred))),
        "mae": float(mean_absolute_error(y_test, pred)),
        "r2": float(r2_score(y_test, pred)),
        "pearson_r": _corr(y_test, pred),
        "spearman_r": _corr(_rank(np.asarray(y_test)), _rank(np.asarray(pred))),
    }


# --------------------------------------------------------------------------- #
# Shared budget sweep + per-task runners
# --------------------------------------------------------------------------- #


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
) -> None:
    """Apply the fitted transform, then sweep budgets, appending one row each.

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
        scores, extra = fit_score(
            x_train[sub], y_train[sub], x_test, y_test, seed + int(round(budget * 1000)) + 17
        )
        rows.append(
            {
                **meta,
                "label_budget": budget,
                "seed": seed,
                "n_train_sub": int(len(sub)),
                "n_test": int(len(y_test)),
                **extra,
                **scores,
            }
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

        # Tag every sub-event (fit, score, macs) with this budget's identity
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
            scores, extra = fit_score(x_tr, y_tr, x_te, y_te, sub_seed)
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
            scores, extra = fit_score(
                x_train[sub], y_train[sub], x_test, y_test, seed + int(round(budget * 1000)) + 17
            )
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
) -> None:
    """Binary calibrated-probe budget sweep (source-fraction budgets).

    The optional ``transform`` is an already-fitted robustness method applied
    identically to train and test features before probing (``None`` = ERM). The
    ``meta`` dict (encoder, method, holdout, condition, train_regime, ...) is
    merged into every row so the comparison tables are self-describing.
    """

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
        return score_binary(clf, threshold, x_te, y_te), extra

    _sweep_budgets(
        rows, x_train, x_test, y_train, y_test, seed, fit_score,
        transform=transform, budgets=budgets, meta=meta, stratify=True, groups_train=groups_train,
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
) -> None:
    """Binary calibrated-probe *target-budget* sweep.

    Budget = 0 → strict geographic holdout (train on source only, test on all
    target).  Budget > 0 → sample that many *target* labels for training and
    keep the remainder as the test set (few-shot target adaptation).
    """

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
        return score_binary(clf, threshold, x_te, y_te), extra

    _sweep_target_budgets(
        rows, x_source, x_target_full, y_source, y_target_full, seed, fit_score,
        transform=transform, budgets=budgets, meta=meta, stratify=True, groups_source=groups_source,
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
) -> None:
    """Multiclass logistic-probe source-budget sweep (crop-type classification)."""

    def fit_score(x_tr: np.ndarray, y_tr: np.ndarray, x_te: np.ndarray, y_te: np.ndarray, probe_seed: int):
        clf, probe_meta = fit_probe_multiclass(x_tr, y_tr, probe_seed)
        if transform is not None and hasattr(transform, "adapt_test_features"):
            x_te = transform.adapt_test_features(clf, x_te)
        return score_multiclass(clf, x_te, y_te), probe_meta

    _sweep_budgets(
        rows, x_train, x_test, y_train, y_test, seed, fit_score,
        transform=transform, budgets=budgets, meta=meta, stratify=True, groups_train=groups_train,
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
) -> None:
    """Multiclass target-budget sweep."""

    def fit_score(x_tr: np.ndarray, y_tr: np.ndarray, x_te: np.ndarray, y_te: np.ndarray, probe_seed: int):
        clf, probe_meta = fit_probe_multiclass(x_tr, y_tr, probe_seed)
        if transform is not None and hasattr(transform, "adapt_test_features"):
            x_te = transform.adapt_test_features(clf, x_te)
        return score_multiclass(clf, x_te, y_te), probe_meta

    _sweep_target_budgets(
        rows, x_source, x_target_full, y_source, y_target_full, seed, fit_score,
        transform=transform, budgets=budgets, meta=meta, stratify=True, groups_source=groups_source,
    )


def run_probes_regression(
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
) -> None:
    """Ridge-probe source-budget sweep (phenology-proxy regression). Targets are
    continuous, so budget sub-sampling is not stratified."""

    def fit_score(x_tr: np.ndarray, y_tr: np.ndarray, x_te: np.ndarray, y_te: np.ndarray, probe_seed: int):
        clf, probe_meta = fit_probe_regression(x_tr, y_tr, probe_seed)
        return score_regression(clf, x_te, y_te), probe_meta

    _sweep_budgets(
        rows, x_train, x_test, y_train, y_test, seed, fit_score,
        transform=transform, budgets=budgets, meta=meta, stratify=False, groups_train=groups_train,
    )


def run_probes_regression_target(
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
) -> None:
    """Ridge-probe target-budget sweep (phenology-proxy regression)."""

    def fit_score(x_tr: np.ndarray, y_tr: np.ndarray, x_te: np.ndarray, y_te: np.ndarray, probe_seed: int):
        clf, probe_meta = fit_probe_regression(x_tr, y_tr, probe_seed)
        return score_regression(clf, x_te, y_te), probe_meta

    _sweep_target_budgets(
        rows, x_source, x_target_full, y_source, y_target_full, seed, fit_score,
        transform=transform, budgets=budgets, meta=meta, stratify=False, groups_source=groups_source,
    )
