"""Performance tracking and budget-sweep helpers."""

from __future__ import annotations

import hashlib
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

#: Budget cells that were skipped after a degenerate fit. A skipped cell writes NO row, so without
#: this the only trace is one `[skip]` line in a multi-hour log while the run still exits 0. The
#: completeness check in artifacts.write_run_complete independently catches the missing cell; this
#: records WHY, so the failure is actionable rather than just "a cell is absent". Appended from
#: joblib threads -- list.append is atomic under the GIL.
CELL_FAILURES: list[dict[str, Any]] = []


def clear_cell_failures() -> None:
    CELL_FAILURES.clear()
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

def _maybe_subset_groups(groups: np.ndarray | None, idx: np.ndarray) -> np.ndarray | None:
    return None if groups is None else np.asarray(groups)[idx]


#: Constant per-row metadata the sweeps used to get from the method-variant selector. ERM has one
#: variant and never tunes, so every value is fixed -- but all 14,773 canonical rows carry these
#: columns, so they are emitted verbatim rather than dropped with the selector. Tabular only: the
#: dense path never wrote them.
ERM_TUNING_METADATA: dict[str, Any] = {
    "method_tuned": 0,
    "method_n_variants": 1,
    "method_selected_variant": "default",
    "method_selected_kwargs": "{}",
    "method_selection_scope": "none",
    "method_selection_metric": "",
    "method_selection_score": float("nan"),
}


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


# ── Probe-capacity training-size cap (PROBE_CAP, set from main.py) ───────────────────────────
# Caps probe TRAINING size so the (256,128) MLP is tractable on the full tabular benchmarks, for the
# probe-capacity *sensitivity* ablation. Properties, all required for a defensible comparison:
#   * ENCODER-INDEPENDENT: the sampling seed is keyed only on the cell identity (benchmark/seed/region/
#     route/budget), never on the model, so every encoder trains on the IDENTICAL sampled rows.
#   * Class-stratified, and group-balanced across source geographic groups where feasible.
#   * Never touches validation or test sets, and (few-shot route) never drops a target-k example --
#     only the source portion is subsampled to meet the cap.
#   * ALL THREE families (logistic, MLP, kNN) share this fixed-budget candidate pool: a common
#     controlled training budget, so benchmark size cannot confound the probe-family comparison.
#     It is NOT a per-family exception. (kNN/MLP then class-balance *within* the pool, per
#     ``_fit_probe``, so the families share the candidate pool but not the exact resampled set --
#     describe it as a shared candidate pool, not identical training rows.)
_CAP_FAMILIES: frozenset[str] = frozenset({"logistic", "mlp", "knn"})


# Absolute source-head training-size cap for the probe-capacity check. None = uncapped (the default
# run). Set from the committed ``PROBE_CAP`` constant in main.py -- there is no env override.
PROBE_CAP: int | None = None


def _cap_seed(seed: int, budget_type: str, budget: float | int, meta: dict[str, Any] | None) -> int:
    """Deterministic ENCODER-INDEPENDENT sampling seed: keyed only on the cell identity so all models
    (and both the capped MLP and the capped logistic) share the same sampled rows for a given cell."""
    meta = meta or {}
    key = "|".join(str(v) for v in (
        meta.get("benchmark"), seed, meta.get("split_regime"), meta.get("holdout"), budget_type, budget,
    ))
    return int(hashlib.sha1(key.encode()).hexdigest()[:12], 16)


def _cap_stratified_indices(y: np.ndarray, groups: np.ndarray | None, cap: int, seed: int) -> np.ndarray:
    """Sorted positions into ``y`` of a class-stratified (group-balanced where feasible) sample of size
    ``min(cap, len(y))``. Deterministic in ``(y, groups, cap, seed)`` only."""
    n = len(y)
    if cap >= n:
        return np.arange(n)
    rng = np.random.default_rng(seed)
    y = np.asarray(y)
    classes, class_counts = np.unique(y, return_counts=True)
    # Per-class quota proportional to class frequency, summing to exactly ``cap`` (largest-remainder).
    exact = class_counts.astype(np.float64) / n * cap
    quota = np.floor(exact).astype(int)
    deficit = cap - int(quota.sum())
    if deficit > 0:
        quota[np.argsort(-(exact - quota))[:deficit]] += 1
    selected: list[int] = []
    for cls, q in zip(classes, quota, strict=True):
        if q <= 0:
            continue
        cls_pos = np.where(y == cls)[0]
        if q >= len(cls_pos):
            selected.extend(cls_pos.tolist())
            continue
        if groups is None:
            selected.extend(rng.choice(cls_pos, size=int(q), replace=False).tolist())
            continue
        # Group-balance within the class: round-robin across the groups present in this class, drawing
        # one at a time so groups stay as even as feasible (a small group contributes all it has; the
        # others absorb the remainder). Group order + within-group order are RNG-permuted for determinism.
        g = np.asarray(groups)[cls_pos]
        uniq = np.unique(g)
        uniq = uniq[rng.permutation(len(uniq))]
        pools = {gg: rng.permutation(cls_pos[g == gg]).tolist() for gg in uniq}
        active = [gg for gg in uniq if pools[gg]]
        remaining = int(q)
        while remaining > 0 and active:
            for gg in list(active):
                if remaining == 0:
                    break
                selected.append(pools[gg].pop())
                remaining -= 1
                if not pools[gg]:
                    active.remove(gg)
    return np.sort(np.asarray(selected, dtype=int))


def _cap_row_meta(cap: int | None, family: str, n_precap: int, y_post: np.ndarray | None) -> dict[str, Any]:
    """Result-row metadata: the cap, pre/post training sizes, and per-class post-cap counts."""
    capped = cap is not None and family in _CAP_FAMILIES and y_post is not None
    if capped:
        classes, counts = np.unique(np.asarray(y_post), return_counts=True)
        class_counts = ";".join(f"{int(c)}:{int(k)}" for c, k in zip(classes, counts, strict=True))
        n_post = int(len(y_post))
    else:
        class_counts = ""
        n_post = int(n_precap)
    return {
        "probe_cap": int(cap) if cap is not None else 0,
        "probe_capped": int(bool(capped)),
        "n_train_precap": int(n_precap),
        "n_train_postcap": n_post,
        "probe_cap_class_counts": class_counts,
    }


def _record_cell_failure(meta: dict, budget: Any, budget_type: str, exc: Exception) -> None:
    """Record a skipped budget cell and say so on stdout."""
    CELL_FAILURES.append({
        "seed": meta.get("seed"),
        "split_regime": meta.get("split_regime"),
        "holdout": meta.get("holdout"),
        "method": meta.get("method"),
        "probe_family": meta.get("probe_family"),
        "budget_type": budget_type,
        "label_budget": budget,
        "reason": f"{type(exc).__name__}: {exc}",
    })
    print(
        f"   [skip] cell failed ({exc}) method={meta.get('method', '?')} budget={budget}",
        flush=True,
    )


def _sweep_budgets(
    rows: list[dict[str, Any]],
    x_train: np.ndarray,
    x_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    seed: int,
    fit_score: Any,
    *,
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
    extra_evals: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]] | None = None,
) -> None:
    """Sweep source-fraction budgets, appending one
    row each.

    ``fit_score(x_tr_sub, y_tr_sub, x_test, y_test, probe_seed, x_cal, y_cal) ->
    (scores, extra)`` is the only benchmark-specific piece; everything else
    (budget sub-sampling, metadata, bookkeeping) is shared.
    ``x_val``/``y_val`` is the regime's held-out validation set, used by the binary
    probe to calibrate its threshold (ignored by the multiclass probe).
    """
    meta = dict(meta or {})
    eval_sets = {name: (x, y, sample_ids, groups) for name, (x, y, sample_ids, groups) in (extra_evals or {}).items() if len(y)}
    for budget in budgets:
        sub_seed = _budget_seed(seed, budget)
        fit_budget = float(budget)
        identity = {
            "seed": seed,
            "holdout": meta.get("holdout"),
            "method": meta.get("method"),
            "budget_type": "source",
            "label_budget": budget,
            "evaluation_split": "test",
        }
        sub = subset_indices(y_train, fit_budget, sub_seed, stratify=stratify)

        # Probe-capacity cap: subsample the (already budget-selected) SOURCE train set to PROBE_CAP.
        # Encoder-independent + class-stratified + group-balanced; evaluation sets are untouched.
        cap = PROBE_CAP
        n_precap = len(sub)
        if cap is not None and family in _CAP_FAMILIES and n_precap > cap:
            sub = sub[_cap_stratified_indices(
                y_train[sub], _maybe_subset_groups(groups_train, sub), cap, _cap_seed(seed, "source", budget, meta),
            )]
        cap_info = _cap_row_meta(cap, family, n_precap, y_train[sub] if cap is not None and family in _CAP_FAMILIES else None)

        tuning_meta = dict(ERM_TUNING_METADATA)
        # ERM: the probe fits the frozen embeddings as they are. No transform is fitted, the
        # eval sets are used unmodified, and the training data is the budget subsample itself.
        x_test_t = x_test
        eval_sets_t = eval_sets
        set_identity(identity)
        x_fit, y_fit = x_train[sub], y_train[sub]
        cal_x_fit, cal_y_fit = (x_val, y_val) if x_val is not None and len(x_val) else (None, None)
        try:
            with measure(f"probe.sweep.source/{meta.get('benchmark', '?')}/{meta.get('method', '?')}",
                              n_train=len(y_fit), n_test=len(y_test)):
                result = fit_score(
                    x_fit, y_fit, x_test_t, y_test, seed + int(round(budget * 1000)) + 17,
                    cal_x_fit, cal_y_fit,
                )
        except ValueError as exc:
            # A degenerate (empty) train/eval array for one budget cell; skip it rather than
            # aborting the pair -- but RECORD it: the cell now has no row, and a silently absent
            # cell is the failure this accumulator exists to make loud.
            set_identity(None)
            _record_cell_failure(meta, budget, "source", exc)
            continue
        score_fitted = None
        if len(result) == 4:
            scores, extra_meta, per_sample, score_fitted = result
        elif len(result) == 3:
            scores, extra_meta, per_sample = result
        else:
            scores, extra_meta = result
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
                "n_train_sub": int(len(y_fit)),
                "n_test": int(len(y_test)),
                **tuning_meta,
                **extra_meta,
                **cap_info,
                **scores,
            }
        )
        regime_base._append_prediction_rows(
            predictions,
            meta=meta,
            seed=seed,
            budget_type="source",
            label_budget=budget,
            n_train_sub=len(y_fit),
            sample_ids=np.asarray(sample_ids_test if sample_ids_test is not None else np.arange(len(y_test))),
            groups_test=groups_test,
            per_sample=per_sample,
        )
        if score_fitted is None:
            continue
        for split_name, (x_eval, y_eval, sample_ids_eval, groups_eval) in eval_sets_t.items():
            scores_e, per_sample_e = score_fitted(x_eval, y_eval)
            scores_e = {**scores_e, **regime_base._worst_group_scores(per_sample_e, groups_eval)}
            rows.append({
                **meta,
                "budget_type": "source",
                "label_budget": budget,
                "evaluation_split": split_name,
                "seed": seed,
                "n_train_sub": int(len(y_fit)),
                "n_test": int(len(y_eval)),
                **tuning_meta,
                **extra_meta,
                **cap_info,
                **scores_e,
            })
            regime_base._append_prediction_rows(
                predictions,
                meta=meta,
                seed=seed,
                budget_type="source",
                label_budget=budget,
                n_train_sub=len(y_fit),
                sample_ids=np.asarray(sample_ids_eval if sample_ids_eval is not None else np.arange(len(y_eval))),
                groups_test=groups_eval,
                per_sample=per_sample_e,
                evaluation_split=split_name,
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
    x_te_raw, y_te = x_target_full[test_idx], y_target_full[test_idx]

    for budget in budgets:
        if budget != 0 and degenerate:
            continue  # too few target samples for a held-out train/test split
        cal_x_raw, cal_y = x_val, y_val  # source val for zero-shot / few-shot (no target peeking)
        tune_internal = False
        if budget == 0:
            sub_seed = seed
            x_tr_raw, y_tr = x_source, y_source
            groups_tr = groups_source
            n_source_train = len(y_source)
        elif budget == target_id_budget:
            sub_seed = _budget_seed(seed, budget)
            x_tr_raw, y_tr = x_target_full[order], y_target_full[order]
            groups_tr = groups_full[order] if groups_full is not None else None
            cal_x_raw, cal_y = None, None
            tune_internal = True
            n_source_train = 0
        else:
            sub_seed = _budget_seed(seed, budget)
            k = min(len(order), _target_budget_count(budget, len(order)))
            few = order[:k]
            x_tr_raw = np.concatenate([x_source, x_target_full[few]])
            y_tr = np.concatenate([y_source, y_target_full[few]])
            if groups_source is not None and groups_full is not None:
                groups_tr = np.concatenate([np.asarray(groups_source), groups_full[few]])
            else:
                groups_tr = None
            n_source_train = len(y_source)

        # Probe-capacity cap (route-aware): budget 0 caps the source-only train; the oracle caps the
        # target pool; few-shot KEEPS all k target labels and caps only the source head to PROBE_CAP.
        cap = PROBE_CAP
        n_precap = len(y_tr)
        cap_active = cap is not None and family in _CAP_FAMILIES and n_precap > cap
        if cap_active:
            cseed = _cap_seed(seed, "target", budget, meta)
            if budget == 0 or budget == target_id_budget:
                sel = _cap_stratified_indices(y_tr, groups_tr, cap, cseed)
                x_tr_raw, y_tr = x_tr_raw[sel], y_tr[sel]
                groups_tr = _maybe_subset_groups(groups_tr, sel)
                if budget == 0:
                    n_source_train = len(y_tr)
            else:
                n_target_k = n_precap - n_source_train  # the k target few-shot labels (kept in full)
                src_sel = _cap_stratified_indices(
                    y_tr[:n_source_train],
                    _maybe_subset_groups(groups_tr, np.arange(n_source_train)) if groups_tr is not None else None,
                    max(0, cap - n_target_k),
                    cseed,
                )
                keep = np.concatenate([src_sel, np.arange(n_source_train, n_precap)])
                x_tr_raw, y_tr = x_tr_raw[keep], y_tr[keep]
                groups_tr = _maybe_subset_groups(groups_tr, keep)
                n_source_train = int(len(src_sel))
        cap_info = _cap_row_meta(cap, family, n_precap, y_tr if cap_active else None)

        identity = {
            "seed": seed,
            "holdout": meta.get("holdout"),
            "method": meta.get("method"),
            "budget_type": "target",
            "label_budget": budget,
        }
        tuning_meta = dict(ERM_TUNING_METADATA)
        # ERM: probe the frozen embeddings directly. x_tr_raw already carries the source head plus
        # the k target shots the budget selected; no transform is fitted and nothing is reweighted.
        x_fit, y_fit = x_tr_raw, y_tr
        x_te = x_te_raw
        cal_x_fit, cal_y_fit = (
            (cal_x_raw, cal_y) if cal_x_raw is not None and len(cal_x_raw) else (None, None)
        )
        n_tr = len(x_fit)

        set_identity(identity)
        try:
            with measure(f"probe.sweep.target/{meta.get('benchmark', '?')}/{meta.get('method', '?')}",
                              n_train=n_tr, n_test=len(y_te)):
                result = fit_score(
                    x_fit,
                    y_fit,
                    x_te,
                    y_te,
                    sub_seed,
                    cal_x_fit,
                    cal_y_fit,
                    tune_internal,
                )
        except ValueError as exc:
            # As above: skip the single degenerate cell, but record it so the missing row has a
            # documented cause instead of appearing as an unexplained gap.
            set_identity(None)
            _record_cell_failure(meta, budget, "target", exc)
            continue
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
            "seed": seed, "n_train_sub": n_tr, "n_test": len(y_te), **tuning_meta, **cap_info, **extra, **scores,
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
            x_target_t = x_target_full
            if score_fitted is not None:
                scores_f, per_sample_f = score_fitted(x_target_t, y_target_full)
                extra_f = extra
            else:
                res_full = fit_score(
                    x_fit,
                    y_fit,
                    x_target_t,
                    y_target_full,
                    sub_seed,
                    cal_x_fit,
                    cal_y_fit,
                    False,
                )
                if len(res_full) == 3:
                    scores_f, extra_f, per_sample_f = res_full
                else:
                    scores_f, extra_f = res_full
                    per_sample_f = None
            groups_all = groups_full if groups_full is not None else None
            scores_f = {**scores_f, **regime_base._worst_group_scores(per_sample_f, groups_all)}
            rows.append({
                **meta, "budget_type": "target", "label_budget": budget, "evaluation_split": "full",
                "seed": seed, "n_train_sub": n_tr, "n_test": len(y_target_full),
                **tuning_meta,
                **cap_info,
                **extra_f, **scores_f,
            })
            regime_base._append_prediction_rows(
                predictions, meta=meta, seed=seed, budget_type="target", label_budget=budget,
                n_train_sub=n_tr, sample_ids=sample_ids_full[full_idx], groups_test=groups_all,
                per_sample=per_sample_f, evaluation_split="full",
            )


def _cpu_capacity() -> int:
    counts = [os.cpu_count() or 1]
    try:
        counts.append(len(os.sched_getaffinity(0)))
    except Exception:
        pass
    for name in ("LOKY_MAX_CPU_COUNT", "SLURM_CPUS_PER_TASK", "SLURM_CPUS_ON_NODE"):
        value = os.environ.get(name)
        if value and value.isdigit():
            counts.append(int(value))
    return max(1, min(counts))


def _available_memory_bytes() -> int:
    try:
        with open("/proc/meminfo") as handle:
            for line in handle:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass
    return int(os.environ.get("PROBE_COPY_BUDGET_BYTES", str(8_000_000_000)))


def _effective_n_jobs(embeddings=None, *, job_bytes: int | None = None) -> int:
    """Concurrent probe fits: use available cores, bounded by measured free memory."""
    cpu_count = _cpu_capacity()
    requested = max(1, int(0.85 * cpu_count))
    tabular_probe = job_bytes is None and embeddings is not None
    if job_bytes is not None:
        max_bytes = int(job_bytes)
    elif embeddings is None:
        max_bytes = 0
    else:
        arrays = embeddings.values() if hasattr(embeddings, "values") else [embeddings]
        max_bytes = max((np.asarray(arr).nbytes for arr in arrays), default=0)
    if tabular_probe:
        max_bytes *= 3
    explicit_budget = os.environ.get("PROBE_COPY_BUDGET_BYTES")
    memory_fraction = 0.65 if job_bytes is not None else 0.50
    budget = int(explicit_budget) if explicit_budget else int(memory_fraction * _available_memory_bytes())
    if max_bytes > 0 and budget > 0:
        requested = max(1, min(requested, budget // max_bytes))
    return requested
