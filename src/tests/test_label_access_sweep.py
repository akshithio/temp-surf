"""Stage 2 -- the 13-route geographic label-access sweep and its calibration routing.

Route construction, exact fit reuse (source_only once), calibration routing, target-test
non-leakage, separate source orders, and infeasible-count hard failure are checked with a recording
fake ``fit_score`` (so the assertions are exact and encoder-free); one integration test drives real
logistic probes to confirm the calibration-source routing end to end. Unit ids live in feature 0."""

from __future__ import annotations

import numpy as np
import pytest

from evals import split_artifacts as SA
from utils import perfutils as perf

COUNTS = (5, 10, 25, 50)
# route_specs order (index -> route), used to map recorded fit calls back to routes.
ROUTE_ORDER = [
    SA.ROUTE_SOURCE_ONLY, *[SA.ROUTE_SOURCE_PLUS_TARGET] * 4, SA.ROUTE_TARGET_ONLY_FULL,
    SA.ROUTE_SOURCE_PLUS_TARGET_FULL, SA.ROUTE_MATCHED_SOURCE, SA.ROUTE_MATCHED_TARGET,
    *[SA.ROUTE_FIXED_TOTAL_MIXED] * 4,
]
SOURCE_VAL_CALLS = [0, 1, 2, 3, 4, 6, 9, 10, 11, 12]   # source_only, spt*4, spt_full, ftm*4
INTERNAL_CALLS = [5, 7, 8]                              # target_only_full, matched_source, matched_target


def _ids(x):
    return {int(v) for v in np.asarray(x)[:, 0].tolist()}


def _make(S=60, P=55, T=10):
    rng = np.random.default_rng(0)
    def block(lo, n):
        return np.column_stack([np.arange(lo, lo + n), rng.normal(size=n)]), np.array([i % 2 for i in range(n)])
    xs, ys = block(100, S)
    xp, yp = block(200, P)
    xt, yt = block(300, T)
    xv, yv = block(400, 10)
    return xs, ys, xp, yp, xt, yt, xv, yv


class _Rec:
    def __init__(self):
        self.calls = []

    def __call__(self, x_tr, y_tr, x_te, y_te, probe_seed, x_cal=None, y_cal=None, tune_internal=False):
        self.calls.append({
            "train": _ids(x_tr), "n_train": int(len(y_tr)), "test": _ids(x_te),
            "cal": None if x_cal is None else _ids(x_cal), "tune_internal": tune_internal, "seed": probe_seed,
        })
        return {"f1": 1.0}, {"extra": 1}, None, None   # score_fitted None -> no complete-target diagnostic


class _RecSF:
    """Like _Rec but returns a reusable score_fitted, so the source_only complete-target diagnostic is
    emitted -- and we can prove it reuses that scorer instead of refitting."""

    def __init__(self):
        self.fit_calls = 0
        self.score_calls = 0

    def __call__(self, x_tr, y_tr, x_te, y_te, probe_seed, x_cal=None, y_cal=None, tune_internal=False):
        self.fit_calls += 1

        def score_fitted(x_eval, y_eval):
            self.score_calls += 1
            return {"f1": 1.0}, None

        scores, per_sample = score_fitted(x_te, y_te)
        return scores, {"extra": 1}, per_sample, score_fitted


def _run(rec, S=60, P=55, T=10, meta_extra=None):
    xs, ys, xp, yp, xt, yt, xv, yv = _make(S, P, T)
    rng = np.random.default_rng(1)
    matched, fixed, target = rng.permutation(len(ys)), rng.permutation(len(ys)), rng.permutation(len(yp))
    meta = {"benchmark": "b", "holdout": "R", "method": "erm", "probe_family": "logistic"}
    meta.update(meta_extra or {})
    rows: list[dict] = []
    perf._sweep_label_access_routes(
        rows, xs, ys, xp, yp, xt, yt, 7, rec, counts=COUNTS,
        matched_source_order=matched, fixed_removal_order=fixed, target_order=target,
        meta=meta, x_val=xv, y_val=yv,
    )
    return rows, matched, fixed, target


def _run_orders(matched, fixed, target, S=60, P=55, T=10):
    xs, ys, xp, yp, xt, yt, xv, yv = _make(S, P, T)
    perf._sweep_label_access_routes(
        [], xs, ys, xp, yp, xt, yt, 7, _Rec(), counts=COUNTS,
        matched_source_order=matched, fixed_removal_order=fixed, target_order=target,
        meta={"benchmark": "b", "holdout": "R"}, x_val=xv, y_val=yv,
    )


def test_thirteen_distinct_fits_and_source_only_reuse():
    rec = _Rec()
    rows, *_ = _run(rec)
    assert len(rec.calls) == 13 and len(rows) == 13          # exactly 13 fits and 13 rows
    routes = [(r["label_access_route"], r["label_budget"]) for r in rows]
    assert len(set(routes)) == 13                            # all distinct
    assert routes.count((SA.ROUTE_SOURCE_ONLY, 0)) == 1      # source_only fit ONCE
    # no aliased additive/fixed k=0 rows (source_only IS that baseline, reused in Stage-5 contrasts)
    assert (SA.ROUTE_SOURCE_PLUS_TARGET, 0) not in routes
    assert (SA.ROUTE_FIXED_TOTAL_MIXED, 0) not in routes


def test_no_target_test_leakage():
    rec = _Rec()
    _run(rec)
    target_test = set(range(300, 310))
    for c in rec.calls:
        assert not (c["train"] & set(range(300, 400)))       # no target_test unit in ANY training set
        assert c["test"] == target_test                      # every route scored on the SAME target_test
        if c["cal"] is not None:
            assert not (c["cal"] & set(range(300, 400)))      # calibration never touches target_test


def test_route_construction_and_counts():
    rec = _Rec()
    rows, *_ = _run(rec)
    by = {(r["label_access_route"], r["label_budget"]): r for r in rows}
    S, P, m = 60, 55, 55
    triple = lambda r: (r["n_source_labels"], r["n_target_labels"], r["n_total_labels"])  # noqa: E731
    assert triple(by[(SA.ROUTE_SOURCE_ONLY, 0)]) == (S, 0, S)
    for k in COUNTS:
        assert triple(by[(SA.ROUTE_SOURCE_PLUS_TARGET, k)]) == (S, k, S + k)
        assert triple(by[(SA.ROUTE_FIXED_TOTAL_MIXED, k)]) == (S - k, k, S)   # total stays fixed at S
    assert triple(by[(SA.ROUTE_TARGET_ONLY_FULL, 0)]) == (0, P, P)
    assert triple(by[(SA.ROUTE_SOURCE_PLUS_TARGET_FULL, 0)]) == (S, P, S + P)
    assert triple(by[(SA.ROUTE_MATCHED_SOURCE, 0)]) == (m, 0, m)
    assert triple(by[(SA.ROUTE_MATCHED_TARGET, 0)]) == (0, m, m)


def test_matched_and_fixed_consume_separate_orders():
    rec = _Rec()
    _rows, matched, fixed, target = _run(rec)
    m = 55
    assert rec.calls[7]["train"] == {int(100 + p) for p in matched[:m]}   # matched_source <- matched order
    # fixed_total_mixed(5): drop first 5 of the FIXED-removal order, add first 5 target
    kept = {int(100 + p) for p in fixed[5:]}
    add = {int(200 + p) for p in target[:5]}
    assert rec.calls[9]["train"] == kept | add
    # the two source interventions genuinely use different orders (matched prefix != fixed-removal prefix)
    assert list(matched[:m]) != list(fixed[:m])


def test_calibration_routing():
    rec = _Rec()
    _run(rec)
    source_val = set(range(400, 410))
    for i in SOURCE_VAL_CALLS:
        assert rec.calls[i]["tune_internal"] is False and rec.calls[i]["cal"] == source_val
    for i in INTERNAL_CALLS:
        assert rec.calls[i]["tune_internal"] is True and rec.calls[i]["cal"] is None


def test_probe_seed_is_the_run_seed_no_derivation():
    rec = _Rec()
    _run(rec)
    assert all(c["seed"] == 7 for c in rec.calls)   # run seed is the probe seed; no derived seeds


def test_row_identity_fields():
    rec = _Rec()
    rows, *_ = _run(rec)
    assert all(r["budget_type"] == "label_access" for r in rows)
    assert all(r["evaluation_split"] == "target_test" for r in rows)
    assert all(r["label_budget_unit"] == "samples" for r in rows)
    assert all(r["seed"] == 7 for r in rows)


def test_infeasible_count_hard_fails():
    with pytest.raises(ValueError, match="infeasible"):
        _run(_Rec(), P=10)   # target pool 10 < max configured count 50 -- never clamped
    with pytest.raises(ValueError, match="infeasible"):
        _run(_Rec(), S=10)   # source pool 10 < 50


def test_integration_real_logistic_calibration_source():
    from evals import evals as EV
    xs, ys, xp, yp, xt, yt, xv, yv = _make()
    d = lambda a: a[:, 1:]  # drop the id column; keep the noise feature  # noqa: E731
    rng = np.random.default_rng(1)
    matched, fixed, target = rng.permutation(len(ys)), rng.permutation(len(ys)), rng.permutation(len(yp))
    rows: list[dict] = []
    EV.run_probes_label_access(
        rows, d(xs), d(xp), d(xt), ys, yp, yt, 0,
        matched_source_order=matched, fixed_removal_order=fixed, target_order=target,
        meta={"benchmark": "b", "holdout": "R", "method": "erm", "probe_family": "logistic"},
        x_val=d(xv), y_val=yv,
    )
    assert len(rows) == 14   # 13 route fits on target_test + 1 source_only complete-target diagnostic
    tt = [r for r in rows if r["evaluation_split"] == SA.EVAL_TARGET_TEST]
    assert len(tt) == 13 and sum(r["evaluation_split"] == SA.EVAL_COMPLETE_TARGET for r in rows) == 1
    by = {r["label_access_route"]: r for r in tt if r["label_budget"] == 0}
    # source-calibrated route uses the external source_val; source-free routes do NOT.
    assert by[SA.ROUTE_SOURCE_ONLY]["calibration_source"] == "regime_val"
    assert by[SA.ROUTE_MATCHED_SOURCE]["calibration_source"] != "regime_val"
    assert by[SA.ROUTE_TARGET_ONLY_FULL]["calibration_source"] != "regime_val"


def test_diagnostic_reuses_fitted_scorer_no_refit():
    rec = _RecSF()
    rows, *_ = _run(rec)
    assert rec.fit_calls == 13   # the diagnostic is NOT a 14th fit
    diag = [r for r in rows if r["evaluation_split"] == SA.EVAL_COMPLETE_TARGET]
    assert len(diag) == 1
    assert diag[0]["label_access_route"] == SA.ROUTE_SOURCE_ONLY and diag[0]["label_budget"] == 0
    assert len([r for r in rows if r["evaluation_split"] == SA.EVAL_TARGET_TEST]) == 13
    # score_fitted called once per route target_test eval (13) + once reused for the diagnostic = 14
    assert rec.score_calls == 14


def test_sweep_emits_exactly_the_expected_rows():
    rec = _RecSF()
    rows, *_ = _run(rec)
    emitted = {(r["label_access_route"], r["label_budget"], r["evaluation_split"]) for r in rows}
    assert emitted == set(SA.label_access_expected_rows())


# --- PROBE_CAP preserves the experiment: ONE shared source base pool of size B (repair 1) -----------

SOURCE_ID_RANGE = set(range(100, 160))   # source unit ids (S=60)


def test_cap_shared_base_preserves_route_geometry(monkeypatch):
    monkeypatch.setattr(perf, "PROBE_CAP", 55)   # B = min(60, 55) = 55 < S
    rec = _Rec()
    rows, *_ = _run(rec)
    B, m = 55, 55
    by = {(r["label_access_route"], r["label_budget"]): r for r in rows}
    assert all(r["n_source_precap"] == 60 and r["n_source_base"] == B and r["probe_capped"] == 1 for r in rows)
    assert by[(SA.ROUTE_SOURCE_ONLY, 0)]["n_source_labels"] == B
    for k in COUNTS:
        assert by[(SA.ROUTE_SOURCE_PLUS_TARGET, k)]["n_total_labels"] == B + k        # additive totals B+k
        assert by[(SA.ROUTE_FIXED_TOTAL_MIXED, k)]["n_total_labels"] == B             # fixed totals stay B
        assert by[(SA.ROUTE_FIXED_TOTAL_MIXED, k)]["n_source_labels"] == B - k
    assert by[(SA.ROUTE_SOURCE_PLUS_TARGET_FULL, 0)]["n_source_labels"] == B          # all B source retained
    assert by[(SA.ROUTE_MATCHED_SOURCE, 0)]["n_source_labels"] == m
    assert by[(SA.ROUTE_MATCHED_TARGET, 0)]["n_target_labels"] == m                   # matched routes equal


def test_cap_every_source_route_from_same_base(monkeypatch):
    monkeypatch.setattr(perf, "PROBE_CAP", 55)
    rec = _Rec()
    _run(rec)
    base = rec.calls[0]["train"]   # source_only trains on exactly the shared base pool
    assert len(base) == 55 and base.issubset(SOURCE_ID_RANGE)
    for i in [0, 1, 2, 3, 4, 6, 7, 9, 10, 11, 12]:   # every route with >= 1 source unit
        assert (rec.calls[i]["train"] & SOURCE_ID_RANGE).issubset(base)


def test_cap_is_model_independent_and_deterministic(monkeypatch):
    monkeypatch.setattr(perf, "PROBE_CAP", 55)
    a, b, c = _Rec(), _Rec(), _Rec()
    _run(a, meta_extra={"model": "encoderA"})
    _run(b, meta_extra={"model": "encoderB"})
    _run(c, meta_extra={"model": "encoderA"})
    assert a.calls[0]["train"] == b.calls[0]["train"]   # base pool independent of the model/encoder
    assert a.calls[0]["train"] == c.calls[0]["train"]   # and deterministic across repeats


def test_cap_below_max_count_hard_fails(monkeypatch):
    monkeypatch.setattr(perf, "PROBE_CAP", 49)   # B = 49 < max configured count 50
    with pytest.raises(ValueError, match="infeasible"):
        _run(_Rec())


# --- order-array validation before any route/fit (repair 2) ---------------------------------------

def _orders(S=60, P=55):
    rng = np.random.default_rng(3)
    return rng.permutation(S), rng.permutation(S), rng.permutation(P)


def test_order_rejects_non_1d():
    matched, fixed, target = _orders()
    with pytest.raises(ValueError, match="1-D"):
        _run_orders(matched.reshape(-1, 1), fixed, target)


def test_order_rejects_non_integer():
    matched, fixed, target = _orders()
    with pytest.raises(ValueError, match="integer"):
        _run_orders(matched.astype(float), fixed, target)


def test_order_rejects_wrong_length():
    matched, fixed, target = _orders()
    with pytest.raises(ValueError, match="expected exactly"):
        _run_orders(matched[:-1], fixed, target)   # omission


def test_order_rejects_negative():
    matched, fixed, target = _orders()
    bad = matched.copy()
    bad[0] = -1
    with pytest.raises(ValueError, match="negative"):
        _run_orders(bad, fixed, target)


def test_order_rejects_out_of_range():
    matched, fixed, target = _orders()
    bad = matched.copy()
    bad[0] = 60   # S=60 -> valid 0..59
    with pytest.raises(ValueError, match="out-of-range"):
        _run_orders(bad, fixed, target)


def test_order_rejects_duplicate():
    matched, fixed, target = _orders()
    bad = matched.copy()
    bad[1] = bad[0]   # duplicate (length still S)
    with pytest.raises(ValueError, match="duplicate"):
        _run_orders(bad, fixed, target)


def test_target_order_is_validated():
    matched, fixed, target = _orders()
    with pytest.raises(ValueError, match="target_order"):
        _run_orders(matched, fixed, target[:-1])   # target omission


# --- failure records keep same-budget routes distinguishable -----------------------------------
def test_same_budget_additive_and_fixed_total_failures_stay_distinguishable():
    """A source_plus_target(25) failure and a fixed_total_mixed(25) failure share label_budget=25 and
    evaluation_split=target_test -- only the route + supervision counts separate them. The failure
    records must carry all of that, or the two collapse to one indistinguishable log line."""
    perf.clear_cell_failures()

    def boom_on_25_target(x_tr, y_tr, x_te, y_te, probe_seed, x_cal=None, y_cal=None, tune_internal=False):
        # target units live in feature-0 range [200, 300); exactly source_plus_target(25) and
        # fixed_total_mixed(25) train on 25 target labels, so this raises for those two routes ONLY.
        n_target = sum(1 for i in _ids(x_tr) if 200 <= i < 300)
        if n_target == 25:
            raise ValueError("degenerate 25-target fit")
        return {"f1": 1.0}, {"extra": 1}, None, None

    rows, *_ = _run(boom_on_25_target)

    assert len(rows) == 11  # 13 routes minus the two that failed
    fails = [f for f in perf.CELL_FAILURES if f["budget_type"] == "label_access"]
    assert len(fails) == 2
    assert {f["label_budget"] for f in fails} == {25}
    by_route = {f["label_access_route"]: f for f in fails}
    assert set(by_route) == {SA.ROUTE_SOURCE_PLUS_TARGET, SA.ROUTE_FIXED_TOTAL_MIXED}
    spt, ftm = by_route[SA.ROUTE_SOURCE_PLUS_TARGET], by_route[SA.ROUTE_FIXED_TOTAL_MIXED]
    # same budget + split + target count, but the source/total supervision distinguishes them
    assert spt["evaluation_split"] == ftm["evaluation_split"] == SA.EVAL_TARGET_TEST
    assert spt["n_target_labels"] == ftm["n_target_labels"] == 25
    assert (spt["n_source_labels"], spt["n_total_labels"]) == (60, 85)   # additive: S + 25
    assert (ftm["n_source_labels"], ftm["n_total_labels"]) == (35, 60)   # fixed-total: total held at S
    assert all(f["label_budget_unit"] == "samples" for f in fails)
    assert all("degenerate 25-target fit" in f["reason"] for f in fails)
    assert spt != ftm  # the records are not identical -- the whole point
    perf.clear_cell_failures()
