"""Stage 2 -- the geographic label-access sweep (fixed-budget allocation + additive appendix) and its
calibration routing.

Route construction, the strictly-nested allocation prefixes, half-up target rounding, the controlled
budget cap (applied to the BUDGET, so it lands identically on every probe family), calibration routing,
target-test non-leakage, order validation, and same-budget failure distinguishability are checked with a
recording fake ``fit_score`` (so the assertions are exact and encoder-free); one integration test drives
real logistic probes to confirm the calibration-source routing end to end. Unit ids live in feature 0."""

from __future__ import annotations

import numpy as np
import pytest

from evals import split_artifacts as SA
from utils import perfutils as perf

PERCENTS = SA.ALLOCATION_PERCENTS      # (0, 25, 50, 75, 100) -- integer percent
COUNTS = SA.LABEL_ACCESS_COUNTS        # (5, 10, 25, 50) -- absolute additive counts
BUDGET = 40                            # B_d; divides every percent exactly (rounding gets its own test)
S_POOL, T_POOL, N_TEST = 60, 55, 10

#: route_specs emission order -> (route, label_budget), used to map recorded fit calls back to routes.
ROUTE_ORDER = [
    *[(SA.ROUTE_FIXED_BUDGET_ALLOCATION, f) for f in PERCENTS],
    *[(SA.ROUTE_SOURCE_PLUS_TARGET, k) for k in COUNTS],
    (SA.ROUTE_TARGET_ONLY_FULL, 0),
    (SA.ROUTE_SOURCE_PLUS_TARGET_FULL, 0),
]
N_ROUTES = len(ROUTE_ORDER)
ALLOC_CALLS = [i for i, (r, _) in enumerate(ROUTE_ORDER) if r == SA.ROUTE_FIXED_BUDGET_ALLOCATION]
ADDITIVE_CALLS = [i for i, (r, _) in enumerate(ROUTE_ORDER) if r == SA.ROUTE_SOURCE_PLUS_TARGET]
TARGET_FULL_CALL = ROUTE_ORDER.index((SA.ROUTE_TARGET_ONLY_FULL, 0))
COMBINED_FULL_CALL = ROUTE_ORDER.index((SA.ROUTE_SOURCE_PLUS_TARGET_FULL, 0))
#: the source-free, route-internal calibration set: the WHOLE allocation curve plus target_only_full.
INTERNAL_CALLS = [*ALLOC_CALLS, TARGET_FULL_CALL]
#: the source-validated set: the additive appendix plus the full combined reference.
SOURCE_VAL_CALLS = [*ADDITIVE_CALLS, COMBINED_FULL_CALL]

SOURCE_IDS = set(range(100, 100 + S_POOL))
POOL_IDS = set(range(200, 200 + T_POOL))
TEST_IDS = set(range(300, 300 + N_TEST))
VAL_IDS = set(range(400, 410))


def _ids(x):
    return {int(v) for v in np.asarray(x)[:, 0].tolist()}


def _src_ids(train):
    return train & SOURCE_IDS


def _tgt_ids(train):
    return train & POOL_IDS


def _make(S=S_POOL, P=T_POOL, T=N_TEST):
    rng = np.random.default_rng(0)
    def block(lo, n):
        return np.column_stack([np.arange(lo, lo + n), rng.normal(size=n)]), np.array([i % 2 for i in range(n)])
    xs, ys = block(100, S)
    xp, yp = block(200, P)
    xt, yt = block(300, T)
    xv, yv = block(400, 10)
    return xs, ys, xp, yp, xt, yt, xv, yv


def _orders(S=S_POOL, P=T_POOL, seed=1):
    """The ONE frozen source order + the target-pool order. There is no second source draw any more:
    every allocation point slices a PREFIX of ``source_order``."""
    rng = np.random.default_rng(seed)
    return rng.permutation(S), rng.permutation(P)


class _Rec:
    def __init__(self):
        self.calls = []

    def __call__(self, x_tr, y_tr, x_te, y_te, probe_seed, x_cal=None, y_cal=None, tune_internal=False):
        self.calls.append({
            "train": _ids(x_tr), "n_train": int(len(y_tr)), "test": _ids(x_te),
            "cal": None if x_cal is None else _ids(x_cal), "tune_internal": tune_internal, "seed": probe_seed,
        })
        return {"f1": 1.0}, {"extra": 1}, None, None


class _RecSF(_Rec):
    """Like ``_Rec`` but ALSO returns a reusable ``score_fitted``. Under the retired contract that
    scorer produced an extra ``complete_target`` diagnostic row; under the current one it must be
    ignored entirely -- label access emits target_test rows and nothing else."""

    def __call__(self, x_tr, y_tr, x_te, y_te, probe_seed, x_cal=None, y_cal=None, tune_internal=False):
        scores, extra, _, _ = super().__call__(x_tr, y_tr, x_te, y_te, probe_seed, x_cal, y_cal, tune_internal)

        def score_fitted(x_eval, y_eval):
            return {"f1": 1.0}, None

        per_sample = score_fitted(x_te, y_te)[1]
        return scores, extra, per_sample, score_fitted


def _run(rec, S=S_POOL, P=T_POOL, T=N_TEST, budget=BUDGET, cap=None, percents=PERCENTS,
         counts=COUNTS, full_target_reference=True, full_combined_reference=True,
         family="logistic", meta_extra=None, source_order=None, target_order=None):
    xs, ys, xp, yp, xt, yt, xv, yv = _make(S, P, T)
    src, tgt = _orders(S, P)
    if source_order is not None:
        src = source_order
    if target_order is not None:
        tgt = target_order
    meta = {"benchmark": "b", "holdout": "R", "method": "erm", "probe_family": "logistic"}
    meta.update(meta_extra or {})
    rows: list[dict] = []
    perf._sweep_label_access_routes(
        rows, xs, ys, xp, yp, xt, yt, 7, rec,
        source_order=src, target_order=tgt, budget=budget,
        percents=percents, counts=counts, controlled_cap=cap,
        full_target_reference=full_target_reference, full_combined_reference=full_combined_reference,
        meta=meta, x_val=xv, y_val=yv, family=family,
    )
    return rows, src, tgt


def _by_route(rows):
    return {(r["label_access_route"], r["label_budget"]): r for r in rows}


# --- route set: exactly the configured fits, all distinct ------------------------------------------

def test_exactly_the_configured_fits_all_distinct():
    """Replaces the retired "13 distinct fits" lock: exactly the configured allocation + additive +
    reference fits, one row each, no duplicates and no aliased extras."""
    rec = _Rec()
    rows, *_ = _run(rec)
    assert len(rec.calls) == N_ROUTES and len(rows) == N_ROUTES
    routes = [(r["label_access_route"], r["label_budget"]) for r in rows]
    assert len(set(routes)) == N_ROUTES
    assert routes == ROUTE_ORDER
    # every configured allocation fraction and additive count is fit exactly once
    for f in PERCENTS:
        assert routes.count((SA.ROUTE_FIXED_BUDGET_ALLOCATION, f)) == 1
    for k in COUNTS:
        assert routes.count((SA.ROUTE_SOURCE_PLUS_TARGET, k)) == 1


def test_optional_references_are_config_driven():
    rec = _Rec()
    rows, *_ = _run(rec, full_target_reference=False, full_combined_reference=False)
    routes = {r["label_access_route"] for r in rows}
    assert routes == {SA.ROUTE_FIXED_BUDGET_ALLOCATION, SA.ROUTE_SOURCE_PLUS_TARGET}
    assert len(rows) == len(PERCENTS) + len(COUNTS)
    assert set(_by_route(rows)) == set(
        (route, b) for (route, b, _es) in SA.label_access_expected_rows(
            PERCENTS, COUNTS, full_target_reference=False, full_combined_reference=False)
    )


def test_no_retired_route_and_no_complete_target_row():
    """The retired routes are gone and label access emits NO ``complete_target`` row -- even when the
    probe hands back a reusable fitted scorer (which is what used to produce that diagnostic)."""
    rows, *_ = _run(_RecSF())
    assert len(rows) == N_ROUTES
    assert all(r["evaluation_split"] == SA.EVAL_TARGET_TEST for r in rows)
    assert not any(r["evaluation_split"] == SA.EVAL_COMPLETE_TARGET for r in rows)
    retired = {"source_only", "matched_source", "matched_target", "fixed_total_mixed"}
    assert not (retired & {r["label_access_route"] for r in rows})
    assert set(SA.LABEL_ACCESS_EVAL_SPLITS) == {SA.EVAL_TARGET_TEST}


# --- the frozen target_test never enters training, and scores every route -------------------------

def test_no_target_test_leakage():
    rec = _Rec()
    _run(rec)
    for c in rec.calls:
        assert not (c["train"] & TEST_IDS)          # no target_test unit in ANY training set
        assert c["test"] == TEST_IDS                # every route scored on the SAME frozen target_test
        if c["cal"] is not None:
            assert not (c["cal"] & TEST_IDS)        # calibration never touches target_test


# --- route construction ---------------------------------------------------------------------------

def test_route_construction_and_counts():
    rec = _Rec()
    rows, *_ = _run(rec)
    by = _by_route(rows)
    triple = lambda r: (r["n_source_labels"], r["n_target_labels"], r["n_total_labels"])  # noqa: E731
    for f in PERCENTS:
        k = SA.allocation_target_count(f, BUDGET)
        assert triple(by[(SA.ROUTE_FIXED_BUDGET_ALLOCATION, f)]) == (BUDGET - k, k, BUDGET)
    for k in COUNTS:
        # additive routes hold the COMPLETE source pool (never the budget) and add k target units
        assert triple(by[(SA.ROUTE_SOURCE_PLUS_TARGET, k)]) == (S_POOL, k, S_POOL + k)
    assert triple(by[(SA.ROUTE_TARGET_ONLY_FULL, 0)]) == (0, T_POOL, T_POOL)
    assert triple(by[(SA.ROUTE_SOURCE_PLUS_TARGET_FULL, 0)]) == (S_POOL, T_POOL, S_POOL + T_POOL)


def test_allocation_total_is_the_budget_at_every_fraction():
    """The whole fixed-budget claim: n_source + n_target == B at every fraction, B fixed across the
    curve."""
    rows, *_ = _run(_Rec())
    alloc = [r for r in rows if r["label_access_route"] == SA.ROUTE_FIXED_BUDGET_ALLOCATION]
    assert len(alloc) == len(PERCENTS)
    for r in alloc:
        assert r["n_source_labels"] + r["n_target_labels"] == BUDGET
        assert r["n_total_labels"] == BUDGET
        assert r["allocation_total_budget"] == BUDGET


def test_allocation_points_are_nested_prefixes_of_one_draw():
    """Replaces the retired "matched_source and matched_target consume separate orders" test. There is
    now ONE source draw: as f rises the source share is a shrinking PREFIX of ``source_order`` and the
    target share a growing prefix of ``target_order``, so the five points are strictly nested."""
    rec = _Rec()
    _rows, src, tgt = _run(rec)
    trains = [rec.calls[i]["train"] for i in ALLOC_CALLS]
    src_sets = [_src_ids(t) for t in trains]
    tgt_sets = [_tgt_ids(t) for t in trains]
    for i, f in enumerate(PERCENTS):
        k = SA.allocation_target_count(f, BUDGET)
        # exact prefixes of the SINGLE frozen draw -- not an independent per-point sample
        assert src_sets[i] == {int(100 + p) for p in src[: BUDGET - k]}
        assert tgt_sets[i] == {int(200 + p) for p in tgt[:k]}
        assert len(src_sets[i]) + len(tgt_sets[i]) == BUDGET
    for i in range(len(PERCENTS) - 1):
        assert src_sets[i + 1] <= src_sets[i]     # source share shrinks, strictly nested
        assert tgt_sets[i] <= tgt_sets[i + 1]     # target share grows, strictly nested
    assert src_sets[0] == {int(100 + p) for p in src[:BUDGET]} and tgt_sets[0] == set()
    assert src_sets[-1] == set() and len(tgt_sets[-1]) == BUDGET


def test_additive_routes_use_the_complete_source_pool():
    rec = _Rec()
    _rows, _src, tgt = _run(rec)
    for i, k in zip(ADDITIVE_CALLS, COUNTS, strict=True):
        train = rec.calls[i]["train"]
        assert _src_ids(train) == SOURCE_IDS                       # COMPLETE pool, never the budget
        assert _tgt_ids(train) == {int(200 + p) for p in tgt[:k]}   # prefix of the same target order
    assert rec.calls[COMBINED_FULL_CALL]["train"] == SOURCE_IDS | POOL_IDS
    assert rec.calls[TARGET_FULL_CALL]["train"] == POOL_IDS


@pytest.mark.parametrize(("budget", "expected"), [
    (40, [0, 10, 20, 30, 40]),
    (42, [0, 11, 21, 32, 42]),   # 10.5 -> 11 and 31.5 -> 32: HALF-UP, not banker's rounding
    (10, [0, 3, 5, 8, 10]),      # 2.5 -> 3 and 7.5 -> 8
    (7, [0, 2, 4, 5, 7]),        # 1.75 -> 2, 3.5 -> 4, 5.25 -> 5
])
def test_half_up_rounding_at_every_fraction(budget, expected):
    rows, *_ = _run(_Rec(), budget=budget)
    by = _by_route(rows)
    got = [by[(SA.ROUTE_FIXED_BUDGET_ALLOCATION, f)]["n_target_labels"] for f in PERCENTS]
    assert got == expected
    assert [SA.allocation_target_count(f, budget) for f in PERCENTS] == expected
    for f, k in zip(PERCENTS, expected, strict=True):
        r = by[(SA.ROUTE_FIXED_BUDGET_ALLOCATION, f)]
        assert (r["n_source_labels"], r["n_total_labels"]) == (budget - k, budget)


# --- calibration routing ---------------------------------------------------------------------------

def test_calibration_routing():
    """The WHOLE allocation curve plus target_only_full use the source-free route-internal procedure
    (f=100 is target-only and must not inherit source validation while its siblings do); the additive
    routes and the full combined reference keep the frozen source validation set."""
    rec = _Rec()
    _run(rec)
    for i in SOURCE_VAL_CALLS:
        assert rec.calls[i]["tune_internal"] is False and rec.calls[i]["cal"] == VAL_IDS
    for i in INTERNAL_CALLS:
        assert rec.calls[i]["tune_internal"] is True and rec.calls[i]["cal"] is None


def test_probe_seed_is_the_run_seed_no_derivation():
    rec = _Rec()
    _run(rec)
    assert all(c["seed"] == 7 for c in rec.calls)   # run seed is the probe seed; no derived seeds


def test_row_identity_and_budget_fields():
    rows, *_ = _run(_Rec())
    assert all(r["budget_type"] == "label_access" for r in rows)
    assert all(r["evaluation_split"] == SA.EVAL_TARGET_TEST for r in rows)
    assert all(r["label_budget_unit"] == "samples" for r in rows)
    assert all(r["seed"] == 7 for r in rows)
    for r in rows:
        assert r["allocation_total_budget"] == BUDGET
        assert r["benchmark_budget"] == BUDGET
        assert r["controlled_budget_cap"] == 0          # no cap configured
        assert r["n_source_pool"] == S_POOL and r["n_target_pool"] == T_POOL
        assert r["n_total_labels"] == r["n_source_labels"] + r["n_target_labels"]


def test_sweep_emits_exactly_the_expected_rows():
    rows, *_ = _run(_RecSF())
    emitted = {(r["label_access_route"], r["label_budget"], r["evaluation_split"]) for r in rows}
    assert emitted == set(SA.label_access_expected_rows(PERCENTS, COUNTS))


# --- the controlled cap applies to the BUDGET, before the mixture is built -------------------------

def test_controlled_cap_shrinks_the_realized_budget():
    cap = 30
    rows, src, tgt = _run(_Rec(), cap=cap)
    by = _by_route(rows)
    for f in PERCENTS:
        k = SA.allocation_target_count(f, cap)
        r = by[(SA.ROUTE_FIXED_BUDGET_ALLOCATION, f)]
        assert (r["n_source_labels"], r["n_target_labels"], r["n_total_labels"]) == (cap - k, k, cap)
    assert all(r["allocation_total_budget"] == cap for r in rows)
    assert all(r["controlled_budget_cap"] == cap for r in rows)
    assert all(r["benchmark_budget"] == BUDGET for r in rows)   # B_d recorded verbatim beside the cap
    # the cap touches the BUDGET only: the additive routes still hold the COMPLETE source pool.
    for k in COUNTS:
        assert by[(SA.ROUTE_SOURCE_PLUS_TARGET, k)]["n_source_labels"] == S_POOL
    assert by[(SA.ROUTE_SOURCE_PLUS_TARGET_FULL, 0)]["n_source_labels"] == S_POOL


def test_cap_above_the_budget_is_a_no_op():
    rows, *_ = _run(_Rec(), cap=BUDGET + 100)
    assert all(r["allocation_total_budget"] == BUDGET for r in rows)


def test_controlled_cap_lands_identically_on_every_probe_family():
    """The cap is applied to the budget, never after route construction, so logistic and MLP see the
    IDENTICAL mixture. A family-conditional cap would silently make the two families incomparable."""
    a, b = _Rec(), _Rec()
    rows_a, *_ = _run(a, cap=30, family="logistic")
    rows_b, *_ = _run(b, cap=30, family="mlp")
    assert [c["train"] for c in a.calls] == [c["train"] for c in b.calls]
    assert [c["n_train"] for c in a.calls] == [c["n_train"] for c in b.calls]
    fields = ("label_access_route", "label_budget", "n_source_labels", "n_target_labels",
              "n_total_labels", "allocation_total_budget", "controlled_budget_cap", "benchmark_budget")
    assert [{k: r[k] for k in fields} for r in rows_a] == [{k: r[k] for k in fields} for r in rows_b]


def test_probe_cap_does_not_touch_the_label_access_sweep(monkeypatch):
    """``perf.PROBE_CAP`` governs the source/target budget sweeps only. The label-access mixture is
    controlled by ``controlled_cap`` on the budget; a stray PROBE_CAP must not resize any route."""
    base = _Rec()
    rows_base, *_ = _run(base)
    monkeypatch.setattr(perf, "PROBE_CAP", 5)
    capped = _Rec()
    rows_capped, *_ = _run(capped)
    assert [c["train"] for c in base.calls] == [c["train"] for c in capped.calls]
    assert rows_base == rows_capped


# --- infeasibility is a hard failure, never a clamp -------------------------------------------------

def test_infeasible_budget_hard_fails():
    with pytest.raises(ValueError, match="infeasible"):
        _run(_Rec(), budget=T_POOL + 1)              # budget exceeds the target pool
    with pytest.raises(ValueError, match="infeasible"):
        _run(_Rec(), S=10, budget=40)                # budget exceeds the source pool
    with pytest.raises(ValueError, match="infeasible"):
        _run(_Rec(), cap=0)                          # realized budget 0
    with pytest.raises(ValueError, match="max additive count"):
        _run(_Rec(), P=10, budget=5)                 # max additive count 50 > target pool 10


# --- order-array validation before any route/fit ----------------------------------------------------

def _bad(order, **kw):
    return _run(_Rec(), **kw, source_order=order)


def test_order_rejects_non_1d():
    src, _tgt = _orders()
    with pytest.raises(ValueError, match="1-D"):
        _bad(src.reshape(-1, 1))


def test_order_rejects_non_integer():
    src, _tgt = _orders()
    with pytest.raises(ValueError, match="integer"):
        _bad(src.astype(float))


def test_order_rejects_wrong_length():
    src, _tgt = _orders()
    with pytest.raises(ValueError, match="expected exactly"):
        _bad(src[:-1])   # omission


def test_order_rejects_negative():
    src, _tgt = _orders()
    bad = src.copy()
    bad[0] = -1
    with pytest.raises(ValueError, match="negative"):
        _bad(bad)


def test_order_rejects_out_of_range():
    src, _tgt = _orders()
    bad = src.copy()
    bad[0] = S_POOL   # valid positions are 0..S-1
    with pytest.raises(ValueError, match="out-of-range"):
        _bad(bad)


def test_order_rejects_duplicate():
    src, _tgt = _orders()
    bad = src.copy()
    bad[1] = bad[0]   # duplicate (length still S)
    with pytest.raises(ValueError, match="duplicate"):
        _bad(bad)


def test_source_order_is_named_in_its_error():
    src, _tgt = _orders()
    with pytest.raises(ValueError, match="source_order"):
        _bad(src[:-1])


def test_target_order_is_validated():
    _src, tgt = _orders()
    with pytest.raises(ValueError, match="target_order"):
        _run(_Rec(), target_order=tgt[:-1])   # target omission


# --- failure records keep same-budget routes distinguishable ---------------------------------------

def test_same_budget_allocation_and_additive_failures_stay_distinguishable():
    """A fixed_budget_allocation(25) failure (25 PERCENT) and a source_plus_target(25) failure (25
    COUNT) share label_budget=25 and evaluation_split=target_test -- only the route + supervision counts
    separate them. The failure records must carry all of that, or the two collapse to one
    indistinguishable log line."""
    perf.clear_cell_failures()
    k_alloc = SA.allocation_target_count(25, BUDGET)         # 10 target of a 40 total
    boom_sets = {(BUDGET - k_alloc, k_alloc), (S_POOL, 25)}  # allocation@25% and additive@25

    def boom_on_the_two_25s(x_tr, y_tr, x_te, y_te, probe_seed, x_cal=None, y_cal=None, tune_internal=False):
        train = _ids(x_tr)
        if (len(_src_ids(train)), len(_tgt_ids(train))) in boom_sets:
            raise ValueError("degenerate 25 fit")
        return {"f1": 1.0}, {"extra": 1}, None, None

    rows, *_ = _run(boom_on_the_two_25s)

    assert len(rows) == N_ROUTES - 2                          # the two failures produced no row
    fails = [f for f in perf.CELL_FAILURES if f["budget_type"] == "label_access"]
    assert len(fails) == 2
    assert {f["label_budget"] for f in fails} == {25}
    by_route = {f["label_access_route"]: f for f in fails}
    assert set(by_route) == {SA.ROUTE_FIXED_BUDGET_ALLOCATION, SA.ROUTE_SOURCE_PLUS_TARGET}
    alloc, add = by_route[SA.ROUTE_FIXED_BUDGET_ALLOCATION], by_route[SA.ROUTE_SOURCE_PLUS_TARGET]
    assert alloc["evaluation_split"] == add["evaluation_split"] == SA.EVAL_TARGET_TEST
    # same numeric budget + split, but the supervision accounting distinguishes them
    assert (alloc["n_source_labels"], alloc["n_target_labels"], alloc["n_total_labels"]) == (
        BUDGET - k_alloc, k_alloc, BUDGET)
    assert (add["n_source_labels"], add["n_target_labels"], add["n_total_labels"]) == (
        S_POOL, 25, S_POOL + 25)
    assert all(f["label_budget_unit"] == "samples" for f in fails)
    assert all(f["allocation_total_budget"] == BUDGET and f["benchmark_budget"] == BUDGET for f in fails)
    assert all(f["n_source_pool"] == S_POOL and f["n_target_pool"] == T_POOL for f in fails)
    assert all("degenerate 25 fit" in f["reason"] for f in fails)
    assert alloc != add  # the records are not identical -- the whole point
    perf.clear_cell_failures()


# --- one real-probe integration: calibration source end to end --------------------------------------

def test_integration_real_logistic_calibration_source():
    from evals import evals as EV
    xs, ys, xp, yp, xt, yt, xv, yv = _make()
    d = lambda a: a[:, 1:]  # drop the id column; keep the noise feature  # noqa: E731
    src, tgt = _orders()
    rows: list[dict] = []
    EV.run_probes_label_access(
        rows, d(xs), d(xp), d(xt), ys, yp, yt, 0,
        budget=BUDGET, source_order=src, target_order=tgt,
        meta={"benchmark": "b", "holdout": "R", "method": "erm", "probe_family": "logistic"},
        x_val=d(xv), y_val=yv,
    )
    assert len(rows) == N_ROUTES
    assert all(r["evaluation_split"] == SA.EVAL_TARGET_TEST for r in rows)
    by = _by_route(rows)
    # source-calibrated routes use the external source_val; source-free routes do NOT.
    for k in COUNTS:
        assert by[(SA.ROUTE_SOURCE_PLUS_TARGET, k)]["calibration_source"] == "regime_val"
    assert by[(SA.ROUTE_SOURCE_PLUS_TARGET_FULL, 0)]["calibration_source"] == "regime_val"
    assert by[(SA.ROUTE_TARGET_ONLY_FULL, 0)]["calibration_source"] != "regime_val"
    for f in PERCENTS:
        assert by[(SA.ROUTE_FIXED_BUDGET_ALLOCATION, f)]["calibration_source"] != "regime_val"
