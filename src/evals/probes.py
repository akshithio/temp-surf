"""Probe family builders, fitting, and calibration.

Three probe families are provided: logistic regression, MLP, and kNN.
Each has its own hyper-parameter grid and build function.
"""

from __future__ import annotations

import hashlib
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
# Whether to sweep the probe's hyperparameter grid. False = the collapsed single-point grid (the
# default run). Set from the committed ``PROBE_TUNING`` constant in main.py -- no env override.
PROBE_TUNING: bool = False


# TRACTABILITY: the 5-value C-sweep costs 5x probe fits per cell. Collapsed to a single C=1.0 by
# default on full EuroCropsML (706k); set PROBE_TUNING=True in main.py to restore the full grid.
LINEAR_GRID: list[float] = (
    [0.01, 0.1, 1.0, 10.0, 100.0]
    if PROBE_TUNING
    else [1.0]
)
LINEAR_DEFAULT_HP = 1.0


def _build_logistic(hp: float, *, solver: str, seed: int, n_fit: int) -> Any:
    del n_fit
    clf = LogisticRegression(
        C=hp,
        max_iter=LINEAR_MAX_ITER,
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
# Combined with the PROBE_CAP training-size cap, this is what makes the capped MLP tractable on the
# full tabular benchmarks. Set PROBE_TUNING=True in main.py to restore the full five-value sweep.
MLP_GRID: list[float] = (
    [1e-4, 1e-3, 1e-2, 1e-1, 1.0]
    if PROBE_TUNING
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


def configure(*, tuning: bool) -> None:
    """Set PROBE_TUNING from main.py's committed constant and rebuild the grids accordingly.

    The grids are module-level, so a plain assignment to PROBE_TUNING after import would leave them
    stale; this rebuilds them. Probe functions read the grids at call time, so this takes effect for
    all subsequent fits. Called once at startup from main.py -- there is no env override.
    """
    global PROBE_TUNING, LINEAR_GRID, MLP_GRID, PROBE_GRIDS, PROBE_C_GRID
    PROBE_TUNING = tuning
    LINEAR_GRID = [0.01, 0.1, 1.0, 10.0, 100.0] if tuning else [1.0]
    MLP_GRID = [1e-4, 1e-3, 1e-2, 1e-1, 1.0] if tuning else [MLP_DEFAULT_HP]
    PROBE_GRIDS = {"logistic": LINEAR_GRID, "mlp": MLP_GRID, "knn": KNN_GRID}
    PROBE_C_GRID = PROBE_GRIDS["logistic"]


def _build_probe(family: str, hp: float, *, solver: str, seed: int, n_fit: int) -> Any:
    if family == "logistic":
        return _build_logistic(hp, solver=solver, seed=seed, n_fit=n_fit)
    if family == "mlp":
        return _build_mlp(hp, solver=solver, seed=seed, n_fit=n_fit)
    if family == "knn":
        return _build_knn(hp, solver=solver, seed=seed, n_fit=n_fit)
    raise ValueError(f"Unknown probe family {family!r}; known: {PROBE_FAMILIES}")


# ── Shared helpers ───────────────────────────────────────────────────────────


#: Families whose sklearn estimator accepts ``sample_weight`` at fit time. kNN does not, so it simply
#: trains on the same selected rows unweighted -- it never trains on a DIFFERENT set.
_WEIGHTED_FAMILIES = frozenset({"logistic", "mlp"})


def class_balance_weights(y: np.ndarray) -> np.ndarray:
    """``w_i = n / (C * n_{y_i})`` -- inverse class frequency, computed once and applied identically by
    every probe family that supports weights.

    This REPLACES the old class-balanced resample. Resampling changed WHICH rows the MLP/kNN families
    trained on, so a probe-family comparison confounded model capacity with sample identity: logistic
    and MLP were not scored on the same experiment. Weighting leaves the selected row set byte-identical
    across families and moves the imbalance correction into the loss, which is why logistic's
    ``class_weight="balanced"`` was also removed -- applying both would correct the imbalance twice."""
    ya = np.asarray(y)
    classes, counts = np.unique(ya, return_counts=True)
    n, c = int(len(ya)), int(len(classes))
    per = {cl: n / (c * int(cnt)) for cl, cnt in zip(classes.tolist(), counts.tolist(), strict=True)}
    return np.asarray([per[v] for v in ya.tolist()], dtype=np.float64)


def selected_set_digest(x: np.ndarray, y: np.ndarray) -> str:
    """Content digest of the EXACT rows a probe was handed, so a test (and a reader) can prove that two
    families consumed an identical training set rather than merely equal-sized ones."""
    h = hashlib.sha256()
    xa = np.ascontiguousarray(x)
    ya = np.ascontiguousarray(y)
    h.update(str(xa.shape).encode())
    h.update(memoryview(xa).cast("B"))
    h.update(memoryview(ya).cast("B"))
    return h.hexdigest()[:16]


def weight_metadata(w: np.ndarray | None, x: np.ndarray, y: np.ndarray) -> dict[str, Any]:
    """Per-fit weight statistics + the selected-set digest, recorded on every probe row."""
    digest = selected_set_digest(x, y)
    if w is None:
        return {**ERM_WEIGHT_METADATA, "probe_selected_set_digest": digest}
    wa = np.asarray(w, dtype=np.float64)
    ess = float((wa.sum() ** 2) / float(np.square(wa).sum())) if wa.size else float("nan")
    return {
        "probe_sample_weighted": 1,
        "probe_weight_min": float(wa.min()) if wa.size else float("nan"),
        "probe_weight_max": float(wa.max()) if wa.size else float("nan"),
        "probe_weight_std": float(wa.std()) if wa.size else float("nan"),
        "probe_weight_ess": ess,
        "probe_selected_set_digest": digest,
    }


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
    sample_weight: np.ndarray | None = None,
):
    clf = _build_probe(family, hp, solver=solver, seed=seed, n_fit=len(y))
    # Same rows for every family; imbalance is handled by weights where the estimator supports them.
    w = sample_weight
    if w is None and family in _WEIGHTED_FAMILIES:
        w = class_balance_weights(y)
    if w is not None and family not in _WEIGHTED_FAMILIES:
        w = None
    fit_kw = {} if w is None else {f"{clf.steps[-1][0]}__sample_weight": np.asarray(w, dtype=np.float64)}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        clf.fit(x, y, **fit_kw)
    # Published on the fitted pipeline so the caller records the ACTUAL final-fit weights and row set.
    clf._probe_fit_meta = weight_metadata(w, x, y)
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
        **getattr(clf, "_probe_fit_meta", ERM_WEIGHT_METADATA),
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
        **getattr(clf, "_probe_fit_meta", ERM_WEIGHT_METADATA),
    }
    return clf, probe_meta
