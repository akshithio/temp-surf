"""Probe families, fitting, scoring, and calibration.

Three probe families are provided: logistic regression, MLP, and kNN.
Each has its own hyper-parameter grid and build function. The module
also exposes binary scoring (with F1-threshold calibration) and
multiclass scoring (with shared-vs-unseen-class decomposition).
"""

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

# ── Logistic regression probe ────────────────────────────────────────────────

LINEAR_SOLVER = "liblinear"
LINEAR_MULTICLASS_SOLVER = "lbfgs"
LINEAR_MAX_ITER = 20_000
LINEAR_TOL = 1e-5
LINEAR_GRID: list[float] = [0.01, 0.1, 1.0, 10.0, 100.0]
LINEAR_DEFAULT_HP = 1.0


def _build_logistic(hp: float, *, solver: str, seed: int, n_fit: int) -> Any:
    del n_fit
    clf = LogisticRegression(
        C=hp,
        max_iter=LINEAR_MAX_ITER,
        class_weight="balanced",
        solver=solver,
        tol=LINEAR_TOL,
        random_state=seed,
    )
    return make_pipeline(StandardScaler(), clf)


# ── MLP probe ────────────────────────────────────────────────────────────────

MLP_HIDDEN: tuple[int, ...] = (256, 128)
MLP_MAX_ITER = 500
MLP_GRID: list[float] = [1e-4, 1e-3, 1e-2, 1e-1, 1.0]
MLP_DEFAULT_HP = 1e-3


def _build_mlp(hp: float, *, solver: str, seed: int, n_fit: int) -> Any:
    del solver, n_fit
    clf = MLPClassifier(hidden_layer_sizes=MLP_HIDDEN, alpha=hp, max_iter=MLP_MAX_ITER, random_state=seed)
    return make_pipeline(StandardScaler(), clf)


# ── KNN probe ────────────────────────────────────────────────────────────────

KNN_GRID: list[float] = [5, 10, 20, 50, 100]
KNN_DEFAULT_HP = 20


def _build_knn(hp: float, *, solver: str, seed: int, n_fit: int) -> Any:
    del solver, seed
    clf = KNeighborsClassifier(
        n_neighbors=max(1, min(int(hp), n_fit - 1)),
        metric="cosine",
        weights="distance",
    )
    return make_pipeline(StandardScaler(), clf)


# ── Probe dispatch ───────────────────────────────────────────────────────────

PROBE_FAMILIES: list[str] = ["logistic", "mlp", "knn"]
PROBE_MAX_ITER_BY_FAMILY: dict[str, int] = {"logistic": LINEAR_MAX_ITER, "mlp": MLP_MAX_ITER, "knn": -1}
PROBE_GRIDS: dict[str, list[float]] = {
    "logistic": LINEAR_GRID,
    "mlp": MLP_GRID,
    "knn": KNN_GRID,
}
PROBE_DEFAULT_HP: dict[str, float] = {
    "logistic": LINEAR_DEFAULT_HP,
    "mlp": MLP_DEFAULT_HP,
    "knn": KNN_DEFAULT_HP,
}
PROBE_C_GRID: list[float] = PROBE_GRIDS["logistic"]


def _build_probe(family: str, hp: float, *, solver: str, seed: int, n_fit: int) -> Any:
    if family == "logistic":
        return _build_logistic(hp, solver=solver, seed=seed, n_fit=n_fit)
    if family == "mlp":
        return _build_mlp(hp, solver=solver, seed=seed, n_fit=n_fit)
    if family == "knn":
        return _build_knn(hp, solver=solver, seed=seed, n_fit=n_fit)
    raise ValueError(f"Unknown probe family {family!r}; known: {PROBE_FAMILIES}")


# ── Shared helpers ───────────────────────────────────────────────────────────


def _class_balanced_resample(x: np.ndarray, y: np.ndarray, seed: int):
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


# ── Feature-transform protocol ───────────────────────────────────────────────


class FeatureTransform(Protocol):
    """Feature-space transform hook."""

    def transform(self, x: np.ndarray) -> np.ndarray: ...


def _apply(transform: FeatureTransform | None, x: np.ndarray) -> np.ndarray:
    return x if transform is None else transform.transform(x)


# ── Binary probe fitting and scoring ─────────────────────────────────────────


_F1_THRESHOLD_WARNED = False


def best_f1_threshold(y_true: np.ndarray, prob: np.ndarray) -> float:
    global _F1_THRESHOLD_WARNED
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
    if best_score <= 0.0:
        if not _F1_THRESHOLD_WARNED:
            print("   !! F1 threshold calibration is degenerate; using threshold=0.5", flush=True)
            _F1_THRESHOLD_WARNED = True
        return 0.5
    return best_threshold


def fit_probe_with_calibration(
    x_train: np.ndarray,
    y_train: np.ndarray,
    seed: int,
    x_cal: np.ndarray | None = None,
    y_cal: np.ndarray | None = None,
    family: str = "logistic",
    tune_internal: bool = False,
) -> tuple[Any, float, int, int, dict[str, Any]]:
    """Fit a binary probe and calibrate an F1 threshold."""
    grid = PROBE_GRIDS[family]
    use_external = (
        x_cal is not None and len(x_cal) > 0 and y_cal is not None and len(np.unique(y_cal)) == 2
    )
    with perf.measure(
        f"probe.fit/binary/{family}",
        n_samples=len(y_train), n_features=x_train.shape[1], n_classes=len(np.unique(y_train)),
    ):
        cal_prob = None
        if not use_external and tune_internal and len(y_train) >= 20 and len(np.unique(y_train)) == 2 \
                and min(np.bincount(y_train)) >= 4:
            idx = np.arange(len(y_train))
            fit_idx, cal_idx = train_test_split(idx, test_size=0.20, random_state=seed, stratify=y_train)
            best = None
            for hp in grid:
                clf_i, _, _ = _fit_probe(
                    x_train[fit_idx], y_train[fit_idx], family=family, hp=hp, solver=LINEAR_SOLVER, seed=seed
                )
                try:
                    score = float(roc_auc_score(y_train[cal_idx], clf_i.predict_proba(x_train[cal_idx])[:, 1]))
                except ValueError:
                    score = 0.0
                if best is None or score > best[0]:
                    best = (score, hp, clf_i)
            chosen_hp, clf_oof = best[1], best[2]
            clf, n_iter, convergence_warnings = _fit_probe(
                x_train, y_train, family=family, hp=chosen_hp, solver=LINEAR_SOLVER, seed=seed
            )
            cal_x, cal_y = x_train[cal_idx], y_train[cal_idx]
            cal_prob = clf_oof.predict_proba(cal_x)[:, 1]
            n_fit, n_cal = len(y_train), len(cal_idx)
            calibration_source = "target_internal_tuned_oof"
            grid_size = len(grid)
        elif use_external:
            best = None
            for hp in grid:
                clf, n_iter, cw = _fit_probe(x_train, y_train, family=family, hp=hp, solver=LINEAR_SOLVER, seed=seed)
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
                x_train[fit_idx], y_train[fit_idx], family=family, hp=chosen_hp, solver=LINEAR_SOLVER, seed=seed
            )
            cal_x, cal_y = x_train[cal_idx], y_train[cal_idx]
            n_fit, n_cal = len(fit_idx), len(cal_idx)
            calibration_source = "train_subsplit"
            grid_size = 1
    probe_meta = {
        "probe_family": family,
        "probe_solver": LINEAR_SOLVER if family == "logistic" else family,
        "probe_max_iter": PROBE_MAX_ITER_BY_FAMILY[family],
        "probe_tol": LINEAR_TOL if family == "logistic" else float("nan"),
        "probe_n_iter": n_iter,
        "probe_converged": int(len(convergence_warnings) == 0),
        "probe_convergence_warnings": len(convergence_warnings),
        "probe_warning_message": str(convergence_warnings[0].message) if convergence_warnings else "",
        "calibration_source": calibration_source,
        "probe_hp": chosen_hp,
        "probe_C": chosen_hp if family == "logistic" else float("nan"),
        "probe_grid_size": grid_size,
    }
    if cal_prob is None:
        cal_prob = clf.predict_proba(cal_x)[:, 1]
    threshold = best_f1_threshold(cal_y, cal_prob)
    return clf, threshold, int(n_fit), int(n_cal), probe_meta


def expected_calibration_error(y_true: np.ndarray, prob: np.ndarray, n_bins: int = 10) -> float:
    """Expected calibration error."""
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
    """Binary probe metrics."""
    with perf.measure("probe.score/binary", n_samples=len(y_test), n_features=x_test.shape[1]):
        pred = clf.predict(x_test)
        prob = clf.predict_proba(x_test)[:, 1]
        calibrated_pred = (prob >= threshold).astype(np.int64)
    two_class = len(np.unique(y_test)) == 2
    test_optimal_threshold = best_f1_threshold(y_test, prob)
    calibrated_pred_target_optimal = (prob >= test_optimal_threshold).astype(np.int64)
    scores = {
        "f1": float(f1_score(y_test, pred, zero_division=0)),
        "auc": float(roc_auc_score(y_test, prob)) if two_class else float("nan"),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, pred)),
        "calibrated_f1": float(f1_score(y_test, calibrated_pred, zero_division=0)),
        "calibrated_balanced_accuracy": float(balanced_accuracy_score(y_test, calibrated_pred)),
        "diagnostic_calibrated_f1_target_optimal": float(f1_score(y_test, calibrated_pred_target_optimal, zero_division=0)),
        "diagnostic_optimal_threshold_test": float(test_optimal_threshold),
        "ece": expected_calibration_error(y_test, prob),
        "brier": float(brier_score_loss(y_test, prob)),
        "nll": float(log_loss(y_test, prob, labels=[0, 1])),
        "test_pos_rate": float(np.mean(y_test)),
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


# ── Multiclass probe fitting and scoring ─────────────────────────────────────


def fit_probe_multiclass(
    x_train: np.ndarray,
    y_train: np.ndarray,
    seed: int,
    x_val: np.ndarray | None = None,
    y_val: np.ndarray | None = None,
    family: str = "logistic",
    tune_internal: bool = False,
) -> tuple[Any, dict[str, Any]]:
    """Fit a multiclass probe."""
    grid = PROBE_GRIDS[family]
    use_val = x_val is not None and len(x_val) > 0 and y_val is not None and len(y_val) > 0
    with perf.measure(
        f"probe.fit/multiclass/{family}",
        n_samples=len(y_train), n_features=x_train.shape[1], n_classes=len(np.unique(y_train)),
    ):
        if not use_val and tune_internal and len(y_train) >= 20 and len(np.unique(y_train)) >= 2:
            idx = np.arange(len(y_train))
            try:
                fit_idx, cal_idx = train_test_split(idx, test_size=0.20, random_state=seed, stratify=y_train)
            except ValueError:
                fit_idx, cal_idx = train_test_split(idx, test_size=0.20, random_state=seed, stratify=None)
            best = None
            for hp in grid:
                clf_i, _, _ = _fit_probe(
                    x_train[fit_idx], y_train[fit_idx], family=family, hp=hp, solver=LINEAR_MULTICLASS_SOLVER, seed=seed
                )
                score = float(balanced_accuracy_score(y_train[cal_idx], clf_i.predict(x_train[cal_idx])))
                if best is None or score > best[0]:
                    best = (score, hp)
            chosen_hp = best[1]
            clf, n_iter, convergence_warnings = _fit_probe(
                x_train, y_train, family=family, hp=chosen_hp, solver=LINEAR_MULTICLASS_SOLVER, seed=seed
            )
            grid_size = len(grid)
        elif use_val:
            best = None
            for hp in grid:
                clf, n_iter, cw = _fit_probe(
                    x_train, y_train, family=family, hp=hp, solver=LINEAR_MULTICLASS_SOLVER, seed=seed
                )
                val_score = float(balanced_accuracy_score(y_val, clf.predict(x_val)))
                if best is None or val_score > best[0]:
                    best = (val_score, hp, clf, n_iter, cw)
            _, chosen_hp, clf, n_iter, convergence_warnings = best
            grid_size = len(grid)
        else:
            chosen_hp = PROBE_DEFAULT_HP[family]
            clf, n_iter, convergence_warnings = _fit_probe(
                x_train, y_train, family=family, hp=chosen_hp, solver=LINEAR_MULTICLASS_SOLVER, seed=seed
            )
            grid_size = 1
    probe_meta = {
        "probe_family": family,
        "probe_solver": LINEAR_MULTICLASS_SOLVER if family == "logistic" else family,
        "probe_max_iter": PROBE_MAX_ITER_BY_FAMILY[family],
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
    seen = np.asarray(getattr(clf, "classes_", []))
    seen_mask = np.isin(y_test, seen) if seen.size else np.zeros(len(y_test), dtype=bool)
    n_unseen = int(len(set(np.unique(y_test).tolist()) - set(seen.tolist())))
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="y_pred contains classes not in y_true")
        scores = {
            "macro_f1": float(f1_score(y_test, pred, average="macro", zero_division=0)),
            "weighted_f1": float(f1_score(y_test, pred, average="weighted", zero_division=0)),
            "balanced_accuracy": float(balanced_accuracy_score(y_test, pred)),
            "accuracy": float(accuracy_score(y_test, pred)),
            "macro_auc": macro_auc,
            "test_n_classes": int(len(_vals)),
            "test_majority_rate": float(_counts.max() / len(y_test)),
            "n_classes_seen": int(seen.size),
            "n_classes_unseen": n_unseen,
            "unseen_prevalence": float(1.0 - seen_mask.mean()) if len(y_test) else float("nan"),
            "shared_macro_f1": (
                float(f1_score(y_test[seen_mask], pred[seen_mask], average="macro", zero_division=0))
                if seen_mask.any() else float("nan")
            ),
            "shared_balanced_accuracy": (
                float(balanced_accuracy_score(y_test[seen_mask], pred[seen_mask])) if seen_mask.any() else float("nan")
            ),
            "shared_accuracy": (
                float(accuracy_score(y_test[seen_mask], pred[seen_mask])) if seen_mask.any() else float("nan")
            ),
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
