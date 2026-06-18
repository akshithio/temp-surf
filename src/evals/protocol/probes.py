"""Probe fitting, scoring, and calibration primitives."""

from __future__ import annotations

import warnings
from typing import Any, Protocol

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    log_loss,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from utils import perfutils as perf

# --------------------------------------------------------------------------- #
# Probe hyper-parameters
# --------------------------------------------------------------------------- #

PROBE_SOLVER = "liblinear"
PROBE_MULTICLASS_SOLVER = "lbfgs"
PROBE_MAX_ITER = 20000
PROBE_TOL = 1e-5

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
    """ECE: mean absolute gap between accuracy and confidence per bin, weighted by bin size."""
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


def score_binary(
    clf: Any, threshold: float, x_test: np.ndarray, y_test: np.ndarray, return_per_sample: bool = False
) -> dict[str, float] | tuple[dict[str, float], dict[str, np.ndarray]]:
    """Default-threshold and calibrated-threshold binary metrics.

    Probability-quality metrics (Brier, NLL/log-loss) complement ECE: ECE is
    bin-sensitive and noisy on small holdouts, while Brier and NLL are proper
    scoring rules that score the full predictive distribution. If
    ``return_per_sample`` is set, also returns the per-sample prediction arrays
    (y_true / prob / pred_default / pred_calibrated) for artifact logging.
    """
    with perf.measure("probe.score/binary", n_samples=len(y_test), n_features=x_test.shape[1]):
        pred = clf.predict(x_test)
        prob = clf.predict_proba(x_test)[:, 1]
        calibrated_pred = (prob >= threshold).astype(np.int64)
    two_class = len(np.unique(y_test)) == 2  # roc_auc / log_loss need both classes present
    test_optimal_threshold = best_f1_threshold(y_test, prob)
    calibrated_pred_target_optimal = (prob >= test_optimal_threshold).astype(np.int64)
    scores = {
        "f1": float(f1_score(y_test, pred, zero_division=0)),
        "auc": float(roc_auc_score(y_test, prob)) if two_class else float("nan"),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, pred)),
        "calibrated_f1": float(f1_score(y_test, calibrated_pred, zero_division=0)),
        "calibrated_balanced_accuracy": float(balanced_accuracy_score(y_test, calibrated_pred)),
        "calibrated_f1_target_optimal": float(f1_score(y_test, calibrated_pred_target_optimal, zero_division=0)),
        "optimal_threshold_test": float(test_optimal_threshold),
        "ece": expected_calibration_error(y_test, prob),
        "brier": float(brier_score_loss(y_test, prob)),
        "nll": float(log_loss(y_test, prob, labels=[0, 1])),
        "test_pos_rate": float(np.mean(y_test)),  # test-set prevalence -> no-skill floor for F1/accuracy
    }
    n_classes = len(np.unique(y_test))
    perf.log_static("probe.macs/binary", macs=x_test.shape[1] * n_classes, n_samples=len(y_test))
    if return_per_sample:
        per_sample = {
            "y_true": np.asarray(y_test, dtype=np.int64),
            "prob": prob.astype(np.float64),
            "pred_default": np.asarray(pred, dtype=np.int64),
            "pred_calibrated": calibrated_pred,
        }
        return scores, per_sample
    return scores


# --------------------------------------------------------------------------- #
# Multiclass probe
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


def score_multiclass(
    clf: Any, x_test: np.ndarray, y_test: np.ndarray, return_per_sample: bool = False
) -> dict[str, float] | tuple[dict[str, float], dict[str, np.ndarray]]:
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
    _vals, _counts = np.unique(y_test, return_counts=True)
    with warnings.catch_warnings():
        # Expected for multiclass at small label budgets: the probe can predict a class
        # absent from this test split. The metric handles it; the warning is just noise.
        warnings.filterwarnings("ignore", message="y_pred contains classes not in y_true")
        scores = {
            "macro_f1": float(f1_score(y_test, pred, average="macro", zero_division=0)),
            "weighted_f1": float(f1_score(y_test, pred, average="weighted", zero_division=0)),
            "balanced_accuracy": float(balanced_accuracy_score(y_test, pred)),
            "accuracy": float(accuracy_score(y_test, pred)),
            "macro_auc": macro_auc,
            "test_n_classes": int(len(_vals)),  # for the chance/no-skill floor
            "test_majority_rate": float(_counts.max() / len(y_test)),
        }
    if return_per_sample:
        per_sample = {
            "y_true": np.asarray(y_test, dtype=np.int64),
            "pred": np.asarray(pred, dtype=np.int64),
            "classes": np.asarray(getattr(clf, "classes_", []), dtype=np.int64),
            "proba": np.asarray(proba, dtype=np.float64) if "proba" in locals() else np.zeros((len(y_test), 0)),
        }
        return scores, per_sample
    return scores


# --------------------------------------------------------------------------- #
# Segmentation scoring (per-pixel) — PASTIS-style crop mapping
# --------------------------------------------------------------------------- #


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
    """Per-pixel multiclass scoring with segmentation metrics (the PASTIS linear-probe protocol).

    ``x_test`` is the flattened valid-pixel feature matrix and ``y_test`` the per-pixel class
    labels (background / void already dropped upstream). mIoU is averaged over ``eval_classes``
    (default = the probe's trained classes), excluding classes absent from ``y_test`` (NaN) --
    matching the standard frozen-encoder linear-probe mIoU report (Galileo Table 17 / OlmoEarth).
    """
    with perf.measure("probe.score/segmentation", n_samples=len(y_test), n_features=x_test.shape[1]):
        pred = clf.predict(x_test)
    classes = np.asarray(
        eval_classes if eval_classes is not None else getattr(clf, "classes_", np.unique(y_test))
    )
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
