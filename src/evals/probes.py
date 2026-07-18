"""Probe family builders, fitting, and calibration.

Three probe families are provided: logistic regression, MLP, and kNN.
Each has its own hyper-parameter grid and build function.
"""

from __future__ import annotations

import os
import warnings
from typing import Any

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    balanced_accuracy_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from evals.metrics import (
    best_f1_threshold,
)
from evals.metrics import (
    expected_calibration_error as expected_calibration_error,
)
from evals.metrics import (
    score_binary as score_binary,
)
from evals.metrics import (
    score_multiclass as score_multiclass,
)
from utils import perfutils as perf

# ── Logistic regression probe ────────────────────────────────────────────────

LINEAR_SOLVER = "liblinear"
LINEAR_MULTICLASS_SOLVER = "lbfgs"
LINEAR_MAX_ITER = 20_000
LINEAR_TOL = 1e-5
def _probe_tuning_enabled() -> bool:
    """Whether to sweep the probe's hyperparameter grid (RB_PROBE_TUNING).

    RB_METHOD_TUNING was the old spelling. It is refused rather than honoured or ignored: honouring
    it would keep a name that describes machinery that no longer exists, and ignoring it would
    silently change the probe grid out from under a launcher that still sets it -- which would move
    published numbers with no error and no signature change.
    """
    if os.environ.get("RB_METHOD_TUNING", "").strip():
        raise RuntimeError(
            "RB_METHOD_TUNING is no longer supported: post-hoc adaptation has been removed and it "
            "never tuned a method -- it sizes the PROBE hyperparameter grid. Use RB_PROBE_TUNING=1. "
            "Refusing rather than ignoring, because silently dropping it would change the probe "
            "grid, and therefore the numbers, without any error."
        )
    return os.environ.get("RB_PROBE_TUNING", "").strip().lower() in ("1", "true", "yes")


# TRACTABILITY: the 5-value C-sweep costs 5x probe fits per cell. Collapsed to a single C=1.0 by
# default on full EuroCropsML (706k); set RB_PROBE_TUNING=1 to restore the full grid.
# (Formerly RB_METHOD_TUNING -- it never tuned a "method", it sizes the PROBE's hyperparameter
# grid, and that name became actively misleading once post-hoc adaptation was removed. Supplying
# the old name is a hard error rather than a silent no-op: see _probe_tuning_enabled.)
LINEAR_GRID: list[float] = (
    [0.01, 0.1, 1.0, 10.0, 100.0]
    if _probe_tuning_enabled()
    else [1.0]
)
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
MLP_DEFAULT_HP = 1e-3
# TRACTABILITY (mirrors LINEAR_GRID above): the probe-capacity ablation is a *sensitivity* check, not
# a headline protocol, so by default we collapse the five-alpha MLP grid to the single default alpha.
# Combined with the RB_PROBE_CAP training-size cap, this is what makes the capped MLP tractable on the
# full tabular benchmarks. Set RB_PROBE_TUNING=1 to restore the full five-value sweep.
MLP_GRID: list[float] = (
    [1e-4, 1e-3, 1e-2, 1e-1, 1.0]
    if _probe_tuning_enabled()
    else [MLP_DEFAULT_HP]
)


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
    """Uniform class-balanced resample for the MLP/kNN families (ordinary probe machinery)."""
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


#: ERM fits probes on unweighted samples, so these are constants. They are emitted directly rather
#: than computed, because the weighting machinery is gone but all 14,773 canonical rows carry these
#: columns -- dropping them would change the artifact schema.
ERM_WEIGHT_METADATA: dict[str, Any] = {
    "probe_sample_weighted": 0,
    "probe_weight_min": float("nan"),
    "probe_weight_max": float("nan"),
    "probe_weight_std": float("nan"),
    "probe_weight_ess": float("nan"),
}


def _fit_probe(
    x: np.ndarray,
    y: np.ndarray,
    *,
    family: str,
    hp: float,
    solver: str,
    seed: int,
):
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


# ── Binary probe fitting ─────────────────────────────────────────────────────


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
                    x_train[fit_idx],
                    y_train[fit_idx],
                    family=family,
                    hp=hp,
                    solver=LINEAR_SOLVER,
                    seed=seed,
                )
                try:
                    score = float(roc_auc_score(y_train[cal_idx], clf_i.predict_proba(x_train[cal_idx])[:, 1]))
                except ValueError:
                    score = 0.0
                if best is None or score > best[0]:
                    best = (score, hp, clf_i)
            chosen_hp, clf_oof = best[1], best[2]
            clf, n_iter, convergence_warnings = _fit_probe(
                x_train,
                y_train,
                family=family,
                hp=chosen_hp,
                solver=LINEAR_SOLVER,
                seed=seed,
            )
            cal_x, cal_y = x_train[cal_idx], y_train[cal_idx]
            cal_prob = clf_oof.predict_proba(cal_x)[:, 1]
            n_fit, n_cal = len(y_train), len(cal_idx)
            calibration_source = "target_internal_tuned_oof"
            grid_size = len(grid)
        elif use_external:
            best = None
            for hp in grid:
                clf, n_iter, cw = _fit_probe(
                    x_train,
                    y_train,
                    family=family,
                    hp=hp,
                    solver=LINEAR_SOLVER,
                    seed=seed,
                    )
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
                x_train[fit_idx],
                y_train[fit_idx],
                family=family,
                hp=chosen_hp,
                solver=LINEAR_SOLVER,
                seed=seed,
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
        **ERM_WEIGHT_METADATA,
    }
    if cal_prob is None:
        cal_prob = clf.predict_proba(cal_x)[:, 1]
    threshold = best_f1_threshold(cal_y, cal_prob)
    return clf, threshold, int(n_fit), int(n_cal), probe_meta


# ── Multiclass probe fitting ─────────────────────────────────────────────────


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
                    x_train[fit_idx],
                    y_train[fit_idx],
                    family=family,
                    hp=hp,
                    solver=LINEAR_MULTICLASS_SOLVER,
                    seed=seed,
                )
                score = float(balanced_accuracy_score(y_train[cal_idx], clf_i.predict(x_train[cal_idx])))
                if best is None or score > best[0]:
                    best = (score, hp)
            chosen_hp = best[1]
            clf, n_iter, convergence_warnings = _fit_probe(
                x_train,
                y_train,
                family=family,
                hp=chosen_hp,
                solver=LINEAR_MULTICLASS_SOLVER,
                seed=seed,
            )
            grid_size = len(grid)
        elif use_val:
            best = None
            for hp in grid:
                clf, n_iter, cw = _fit_probe(
                    x_train,
                    y_train,
                    family=family,
                    hp=hp,
                    solver=LINEAR_MULTICLASS_SOLVER,
                    seed=seed,
                    )
                val_score = float(balanced_accuracy_score(y_val, clf.predict(x_val)))
                if best is None or val_score > best[0]:
                    best = (val_score, hp, clf, n_iter, cw)
            _, chosen_hp, clf, n_iter, convergence_warnings = best
            grid_size = len(grid)
        else:
            chosen_hp = PROBE_DEFAULT_HP[family]
            clf, n_iter, convergence_warnings = _fit_probe(
                x_train,
                y_train,
                family=family,
                hp=chosen_hp,
                solver=LINEAR_MULTICLASS_SOLVER,
                seed=seed,
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
        **ERM_WEIGHT_METADATA,
    }
    return clf, probe_meta
