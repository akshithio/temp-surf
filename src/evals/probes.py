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
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
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
# Probe families (the probe-capacity ablation axis)
# --------------------------------------------------------------------------- #
# The primary protocol is `logistic` (the canonical frozen-feature linear probe).
# `mlp` (higher-capacity head) and `knn` (probe-free, OlmoEarth-style) exist to test
# whether an observed geographic gap is caused by the *linear probe* or by the
# *encoder*: if the gap persists under mlp/knn it is an encoder property; if it
# vanishes under mlp it is a "linearly inaccessible from source-only" finding.
PROBE_FAMILIES: list[str] = ["logistic", "mlp", "knn"]

# Per-family hyperparameter grid, each selected on the regime's val set. Same SIZE (5)
# across families so no family gets a larger search budget.
# Set a grid to a single value to disable that family's tuning.
PROBE_GRIDS: dict[str, list[float]] = {
    "logistic": [0.01, 0.1, 1.0, 10.0, 100.0],  # C (inverse L2 strength)
    "mlp": [1e-4, 1e-3, 1e-2, 1e-1, 1.0],        # alpha (L2), fixed 2-layer arch below
    "knn": [5, 10, 20, 50, 100],                  # n_neighbors (cosine distance)
}
PROBE_DEFAULT_HP: dict[str, float] = {"logistic": 1.0, "mlp": 1e-3, "knn": 20}
PROBE_C_GRID: list[float] = PROBE_GRIDS["logistic"]  # backward-compat alias

MLP_HIDDEN: tuple[int, ...] = (256, 128)  # 2-layer head (≈ TESSERA's PASTIS MLP capacity)
PROBE_MLP_MAX_ITER: int = 500


def _build_probe(family: str, hp: float, *, solver: str, seed: int, n_fit: int):
    """Build a standardized probe pipeline for ``family`` at hyperparameter ``hp``.

    Only ``logistic`` uses ``solver``/``class_weight``; ``mlp`` and ``knn`` are class-
    balanced upstream by resampling (see :func:`_class_balanced_resample`), so all three
    train on balanced data. ``knn`` neighbors are clamped to the training size.
    """
    if family == "logistic":
        clf = LogisticRegression(
            C=hp, max_iter=PROBE_MAX_ITER, class_weight="balanced", solver=solver, tol=PROBE_TOL, random_state=seed,
        )
    elif family == "mlp":
        clf = MLPClassifier(hidden_layer_sizes=MLP_HIDDEN, alpha=hp, max_iter=PROBE_MLP_MAX_ITER, random_state=seed)
    elif family == "knn":
        clf = KNeighborsClassifier(n_neighbors=max(1, min(int(hp), n_fit - 1)), metric="cosine", weights="distance")
    else:
        raise ValueError(f"Unknown probe family {family!r}; known: {PROBE_FAMILIES}")
    return make_pipeline(StandardScaler(), clf)


def _class_balanced_resample(x: np.ndarray, y: np.ndarray, seed: int):
    """Resample ``(x, y)`` to a class-balanced set of ~the same total size.

    Used for families with no ``class_weight`` option (``mlp``, ``knn``) so EVERY probe
    family trains on class-balanced data: ``logistic`` balances via
    ``class_weight="balanced"`` (the paper-faithful config), ``mlp``/``knn`` via this
    resample. Bounded to the original size (each class drawn ``len(y) // n_classes``
    times, with replacement only when oversampling), so many-class crop-type does not
    blow up. Deterministic in ``seed`` so the grid sweep is stable.
    """
    rng = np.random.default_rng(seed)
    classes = np.unique(y)
    per_class = max(1, len(y) // len(classes))
    parts = []
    for c in classes:
        c_idx = np.where(y == c)[0]
        parts.append(rng.choice(c_idx, size=per_class, replace=per_class > len(c_idx)))
    sel = np.concatenate(parts)
    rng.shuffle(sel)
    return x[sel], y[sel]


def _fit_probe(x: np.ndarray, y: np.ndarray, *, family: str, hp: float, solver: str, seed: int):
    """Fit a probe of ``family`` at ``hp``; return ``(clf, n_iter, convergence_warnings)``.

    ``mlp``/``knn`` (no ``class_weight``) are fit on a class-balanced resample so every
    family handles imbalance equivalently; ``logistic`` uses ``class_weight="balanced"``.
    """
    if family in ("mlp", "knn"):
        x, y = _class_balanced_resample(x, y, seed)
    clf = _build_probe(family, hp, solver=solver, seed=seed, n_fit=len(y))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        clf.fit(x, y)
    convergence_warnings = [w for w in caught if issubclass(w.category, ConvergenceWarning)]
    estimator = clf.steps[-1][1]
    n_iter = int(np.max(estimator.n_iter_)) if hasattr(estimator, "n_iter_") else -1
    return clf, n_iter, convergence_warnings

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
    x_cal: np.ndarray | None = None,
    y_cal: np.ndarray | None = None,
    family: str = "logistic",
) -> tuple[Any, float, int, int, dict[str, Any]]:
    """Fit a binary probe of ``family`` and an F1 threshold on a validation set.

    The hyperparameter (``C`` for logistic, ``alpha`` for mlp, ``n_neighbors`` for knn)
    is swept over ``PROBE_GRIDS[family]`` and selected by val AUC; the probe is fit on
    all of ``x_train`` and the threshold calibrated on ``x_cal``. With no usable val it
    falls back to the legacy 80/20 internal split at the family's default hyperparameter.
    """
    grid = PROBE_GRIDS[family]
    use_external = (
        x_cal is not None and len(x_cal) > 0 and y_cal is not None and len(np.unique(y_cal)) == 2
    )
    with perf.measure(
        f"probe.fit/binary/{family}",
        n_samples=len(y_train), n_features=x_train.shape[1], n_classes=len(np.unique(y_train)),
    ):
        if use_external:
            # Fit on ALL of train at each grid point, pick by val AUC (threshold-free),
            # then calibrate the threshold on the same val.
            best = None
            for hp in grid:
                clf, n_iter, cw = _fit_probe(x_train, y_train, family=family, hp=hp, solver=PROBE_SOLVER, seed=seed)
                val_auc = float(roc_auc_score(y_cal, clf.predict_proba(x_cal)[:, 1]))
                if best is None or val_auc > best[0]:
                    best = (val_auc, hp, clf, n_iter, cw)
            _, chosen_hp, clf, n_iter, convergence_warnings = best
            cal_x, cal_y = x_cal, y_cal
            n_fit, n_cal = len(y_train), len(y_cal)
            calibration_source = "regime_val"
            grid_size = len(grid)
        else:
            idx = np.arange(len(y_train))
            if len(idx) >= 20 and len(np.unique(y_train)) == 2 and min(np.bincount(y_train)) >= 4:
                fit_idx, cal_idx = train_test_split(idx, test_size=0.20, random_state=seed, stratify=y_train)
            else:
                fit_idx = cal_idx = idx
            chosen_hp = PROBE_DEFAULT_HP[family]
            clf, n_iter, convergence_warnings = _fit_probe(
                x_train[fit_idx], y_train[fit_idx], family=family, hp=chosen_hp, solver=PROBE_SOLVER, seed=seed
            )
            cal_x, cal_y = x_train[cal_idx], y_train[cal_idx]
            n_fit, n_cal = len(fit_idx), len(cal_idx)
            calibration_source = "train_subsplit"
            grid_size = 1
    probe_meta = {
        "probe_family": family,
        "probe_solver": PROBE_SOLVER if family == "logistic" else family,
        "probe_max_iter": PROBE_MAX_ITER,
        "probe_tol": PROBE_TOL,
        "probe_n_iter": n_iter,
        "probe_converged": int(len(convergence_warnings) == 0),
        "probe_convergence_warnings": len(convergence_warnings),
        "probe_warning_message": str(convergence_warnings[0].message) if convergence_warnings else "",
        "calibration_source": calibration_source,
        "probe_hp": chosen_hp,
        "probe_C": chosen_hp if family == "logistic" else float("nan"),
        "probe_grid_size": grid_size,
    }
    cal_prob = clf.predict_proba(cal_x)[:, 1]
    threshold = best_f1_threshold(cal_y, cal_prob)
    return clf, threshold, int(n_fit), int(n_cal), probe_meta


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


def fit_probe_multiclass(
    x_train: np.ndarray,
    y_train: np.ndarray,
    seed: int,
    x_val: np.ndarray | None = None,
    y_val: np.ndarray | None = None,
    family: str = "logistic",
) -> tuple[Any, dict[str, Any]]:
    """Fit a multiclass probe of ``family`` (no threshold calibration).

    When a validation set ``(x_val, y_val)`` is supplied, the family's hyperparameter is
    swept over ``PROBE_GRIDS[family]`` and selected by val balanced accuracy (the same
    equal-budget grid used by the binary probe). Otherwise the default is used.
    """
    grid = PROBE_GRIDS[family]
    use_val = x_val is not None and len(x_val) > 0 and y_val is not None and len(y_val) > 0
    with perf.measure(
        f"probe.fit/multiclass/{family}",
        n_samples=len(y_train), n_features=x_train.shape[1], n_classes=len(np.unique(y_train)),
    ):
        if use_val:
            best = None
            for hp in grid:
                clf, n_iter, cw = _fit_probe(
                    x_train, y_train, family=family, hp=hp, solver=PROBE_MULTICLASS_SOLVER, seed=seed
                )
                val_score = float(balanced_accuracy_score(y_val, clf.predict(x_val)))
                if best is None or val_score > best[0]:
                    best = (val_score, hp, clf, n_iter, cw)
            _, chosen_hp, clf, n_iter, convergence_warnings = best
            grid_size = len(grid)
        else:
            chosen_hp = PROBE_DEFAULT_HP[family]
            clf, n_iter, convergence_warnings = _fit_probe(
                x_train, y_train, family=family, hp=chosen_hp, solver=PROBE_MULTICLASS_SOLVER, seed=seed
            )
            grid_size = 1
    probe_meta = {
        "probe_family": family,
        "probe_solver": PROBE_MULTICLASS_SOLVER if family == "logistic" else family,
        "probe_max_iter": PROBE_MAX_ITER,
        "probe_n_iter": n_iter,
        "probe_converged": int(len(convergence_warnings) == 0),
        "probe_convergence_warnings": len(convergence_warnings),
        "n_classes_train": int(len(clf.classes_)),
        "probe_hp": chosen_hp,
        "probe_C": chosen_hp if family == "logistic" else float("nan"),
        "probe_grid_size": grid_size,
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
    matching the standard frozen-model linear-probe mIoU report (Galileo Table 17 / OlmoEarth).
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


def score_segmentation_per_tile(
    clf: Any,
    x_test: np.ndarray,
    y_test: np.ndarray,
    tile_ids: np.ndarray,
    *,
    eval_classes: np.ndarray | None = None,
) -> dict[str, float]:
    """Score segmentation per tile and return aggregated tile-level metrics.

    Reports ``mean_per_tile_miou`` (average of per-tile mIoUs) and
    ``worst_tile_miou`` (minimum per-tile mIoU) alongside the number of tiles
    scored (``n_tiles_scored``).  Tiles with zero valid pixels after void
    removal are excluded.
    """
    classes = np.asarray(
        eval_classes if eval_classes is not None else getattr(clf, "classes_", np.unique(y_test))
    )
    pred = clf.predict(x_test)
    unique_tiles = np.unique(tile_ids)
    tile_mious: list[float] = []
    for tid in unique_tiles:
        mask = tile_ids == tid
        if mask.sum() == 0:
            continue
        yt, pt = y_test[mask], pred[mask]
        ious = per_class_iou(yt, pt, classes)
        present = [v for v in ious.values() if not np.isnan(v)]
        if present:
            tile_mious.append(float(np.mean(present)))
    if not tile_mious:
        return {"mean_per_tile_miou": float("nan"), "worst_tile_miou": float("nan"), "n_tiles_scored": 0}
    arr = np.asarray(tile_mious)
    return {
        "mean_per_tile_miou": float(np.mean(arr)),
        "worst_tile_miou": float(np.min(arr)),
        "n_tiles_scored": len(arr),
    }
