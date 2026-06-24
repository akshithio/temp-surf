"""Performance tracking instrument for the robustness pipeline.

Records wall-clock, user-CPU, system-CPU time, MAC estimates, data
dimensions, and GPU utilization at every stage.  Use as a context manager::

    with measure("encode/baseline", n_samples=128, n_features=128):
        ...

Every event is tagged with a thread-local **identity** (seed, holdout,
method, budget, budget_type) so nested ``perf.measure`` calls
inside fit/score/sweep functions automatically know which cell they belong
to.  Callers set identity at the sweep boundary::

    perf.set_identity({"seed": 42, "holdout": "togo", ...})
    # all nested measure() calls inherit this identity

The logger is thread-safe so parallel probe workers can each record their
own timings.  All events are accumulated per-(model, benchmark) and flushed
to a JSONL file at the end of each run.
"""

from __future__ import annotations

import json
import os
import resource
import threading
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from threading import Lock
from typing import Any

import numpy as np
from sklearn.model_selection import train_test_split

from evals.regimes import base as regime_base

try:
    import pynvml

    pynvml.nvmlInit()
    _NVML_OK = True
except Exception:
    _NVML_OK = False


@dataclass
class PerfEvent:
    name: str
    wall_s: float
    user_s: float
    sys_s: float
    macs: int | None = None
    n_samples: int | None = None
    n_features: int | None = None
    n_classes: int | None = None
    identity: dict[str, Any] | None = None
    gpu_util: float | None = None
    gpu_mem_mb: float | None = None
    extras: dict[str, Any] | None = None


_EVENTS: list[PerfEvent] = []
_LOCK = Lock()


# --------------------------------------------------------------------------- #
# Thread-local identity context — all measure() calls in this thread
# automatically get tagged, avoiding plumbing through every closure.
# --------------------------------------------------------------------------- #

_tls = threading.local()


def set_identity(identity: dict[str, Any] | None) -> None:
    _tls.identity = identity


def get_identity() -> dict[str, Any] | None:
    return getattr(_tls, "identity", None)


# --------------------------------------------------------------------------- #
# GPU snapshot (best-effort, returns None when unavailable)
# --------------------------------------------------------------------------- #


def _gpu_snapshot() -> tuple[float | None, float | None]:
    if not _NVML_OK:
        return None, None
    try:
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
        mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
        return float(util.gpu), float(mem.used / 1024**2)
    except Exception:
        return None, None


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #


def reset() -> None:
    with _LOCK:
        _EVENTS.clear()


@contextmanager
def measure(name: str, identity: dict[str, Any] | None = None, **extras: Any):
    start_wall = time.perf_counter()
    start_ru = resource.getrusage(resource.RUSAGE_SELF)
    gpu_util_start, gpu_mem_start = _gpu_snapshot()
    try:
        yield
    finally:
        wall = time.perf_counter() - start_wall
        end_ru = resource.getrusage(resource.RUSAGE_SELF)
        gpu_util_end, gpu_mem_end = _gpu_snapshot()
        with _LOCK:
            _EVENTS.append(PerfEvent(
                name=name,
                wall_s=round(wall, 4),
                user_s=round(end_ru.ru_utime - start_ru.ru_utime, 4),
                sys_s=round(end_ru.ru_stime - start_ru.ru_stime, 4),
                identity=identity or get_identity(),
                gpu_util=_avg(gpu_util_start, gpu_util_end),
                gpu_mem_mb=_avg(gpu_mem_start, gpu_mem_end),
                extras=extras or None,
            ))


def _avg(a: float | None, b: float | None) -> float | None:
    if a is None or b is None:
        return None
    return round((a + b) / 2, 1)


def log_static(
    name: str,
    *,
    macs: int | None = None,
    n_samples: int | None = None,
    n_features: int | None = None,
    n_classes: int | None = None,
    identity: dict[str, Any] | None = None,
    **extras: Any,
) -> None:
    """Record a dimension / MAC annotation (zero-duration event)."""
    with _LOCK:
        _EVENTS.append(PerfEvent(
            name=name,
            wall_s=0.0, user_s=0.0, sys_s=0.0,
            macs=macs, n_samples=n_samples, n_features=n_features,
            n_classes=n_classes,
            identity=identity or get_identity(),
            extras=extras or None,
        ))


def write_log(path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    with _LOCK:
        events = list(_EVENTS)
    with open(path, "w") as f:
        for ev in events:
            f.write(json.dumps(asdict(ev), default=str) + "\n")
    return len(events)

def _apply_transform(transform: Any | None, x: np.ndarray | None) -> np.ndarray | None:
    return x if transform is None or x is None else transform.transform(x)

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


def _target_budget_count(budget: float | int, pool_size: int) -> int:
    """Target budget values in (0, 1) are fractions; values >=1 are absolute counts."""
    b = float(budget)
    if 0.0 < b < 1.0:
        return max(1, int(round(b * pool_size)))
    return max(1, int(budget))


def _budget_seed(seed: int, budget: float) -> int:
    """Non-negative per-budget seed. ``abs`` guards the target-ID-upper-bound budget
    (-1), where ``seed + round(budget*1000)`` would go negative and crash RNG/sklearn."""
    return abs(seed + int(round(budget * 1000)))


def _sweep_budgets(
    rows: list[dict[str, Any]],
    x_train: np.ndarray,
    x_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    seed: int,
    fit_score: Any,
    *,
    transform: Any | None,
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
    x_train = _apply_transform(transform, x_train)
    x_test = _apply_transform(transform, x_test)
    x_val = _apply_transform(transform, x_val) if x_val is not None and len(x_val) else None
    for budget in budgets:
        sub_seed = _budget_seed(seed, budget)
        fit_budget = float(budget)
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
            "evaluation_split": "test",
        }
        set_identity(identity)
        with measure(f"probe.sweep.source/{meta.get('benchmark', '?')}/{meta.get('method', '?')}",
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
        set_identity(None)
        scores = {**scores, **regime_base._worst_group_scores(per_sample, groups_test)}

        rows.append(
            {
                **meta,
                "budget_type": "source",
                "label_budget": budget,
                "evaluation_split": "test",
                "seed": seed,
                "n_train_sub": int(len(sub)),
                "n_test": int(len(y_test)),
                **extra,
                **scores,
            }
        )
        regime_base._append_prediction_rows(
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


def _sweep_target_budgets(
    rows: list[dict[str, Any]],
    x_source: np.ndarray,
    x_target_full: np.ndarray,
    y_source: np.ndarray,
    y_target_full: np.ndarray,
    seed: int,
    fit_score: Any,
    *,
    transform: Any | None,
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
    target_id_budget: float | int = -1,
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
    x_source_t = _apply_transform(transform, x_source)
    x_target_t = _apply_transform(transform, x_target_full)
    x_val_t = _apply_transform(transform, x_val) if x_val is not None and len(x_val) else None
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
        elif budget == target_id_budget:
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
        set_identity(identity)
        with measure(f"probe.sweep.target/{meta.get('benchmark', '?')}/{meta.get('method', '?')}",
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
        set_identity(None)
        current_groups = groups_full[test_idx] if groups_full is not None else None
        scores = {**scores, **regime_base._worst_group_scores(per_sample, current_groups)}

        # Every budget is scored on the fixed held-out 20% (the matched set for the few-shot curve
        # and the inherent-difficulty decomposition).
        rows.append({
            **meta, "budget_type": "target", "label_budget": budget, "evaluation_split": "held_out",
            "seed": seed, "n_train_sub": n_tr, "n_test": len(y_te), **extra, **scores,
        })
        regime_base._append_prediction_rows(
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
            scores_f = {**scores_f, **regime_base._worst_group_scores(per_sample_f, groups_all)}
            rows.append({
                **meta, "budget_type": "target", "label_budget": budget, "evaluation_split": "full",
                "seed": seed, "n_train_sub": len(x_source_t), "n_test": len(y_target_full),
                **extra_f, **scores_f,
            })
            regime_base._append_prediction_rows(
                predictions, meta=meta, seed=seed, budget_type="target", label_budget=budget,
                n_train_sub=len(x_source_t), sample_ids=sample_ids_full[full_idx], groups_test=groups_all,
                per_sample=per_sample_f, evaluation_split="full",
            )


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
