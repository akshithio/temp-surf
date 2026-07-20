"""Stage-3 runtime dispatch + identity threading for the geographic label-access suite.

The sweep itself (route construction, nesting, half-up rounding, calibration routing, the controlled
budget cap, order validation) is covered by ``test_label_access_sweep.py``; the frozen artifact
(generation/validation/load) by ``test_label_access.py``; and the through-``main`` publish path by
``test_artifact_integrity.test_tabular_pair_consumes_geographic_few_shot_end_to_end``. This file pins the
piece between them -- ``runstate._probe_cell_label_access`` -- and the 9-field cell identity it must carry
so resume, completeness, and predictions can never confuse one route for another.
"""

from __future__ import annotations

import numpy as np
import pytest

from evals import evals as EV
from evals import split_artifacts as SA
from utils import artifacts, runstate

# One synthetic geographic_ood cell, in CURRENT row indices (disjoint partitions), sized so the whole
# suite is feasible: source S=60 and pool P=55 both clear the benchmark budget B_d and every additive
# count in LABEL_ACCESS_COUNTS.
TRAIN = np.arange(0, 60, dtype=np.int64)
VAL = np.arange(60, 70, dtype=np.int64)
POOL = np.arange(70, 125, dtype=np.int64)
TEST = np.arange(125, 137, dtype=np.int64)
N = 137
#: B_d, the benchmark-common fixed-budget allocation budget frozen at split generation.
BUDGET = 40

BASE_META = {
    "model": "raw", "benchmark": "cropharvest", "method": "erm",
    "split_regime": SA.LABEL_ACCESS_REGIME, "holdout": "kenya", "probe_family": "logistic",
    "target_role": "headline", "budget_type": "label_access",
}
BASE_KEY = (0, SA.LABEL_ACCESS_REGIME, "kenya", "erm", "logistic")  # (seed, regime, holdout, method, family)


def _emb_labels():
    """Perfectly class-separable embeddings + alternating labels, so every route's train set holds both
    classes and no cell degenerately skips."""
    y = np.array([i % 2 for i in range(N)], dtype=np.int64)
    emb = np.zeros((N, 4), dtype=np.float32)
    emb[y == 1] = 1.0
    return emb, y


def _orders():
    """The two frozen orders as CURRENT indices in ascending-rank order (as ``load_label_access``
    returns them): ONE nested source order + one target-pool order."""
    rng = np.random.default_rng(0)
    return rng.permutation(TRAIN), rng.permutation(POOL)


def _run(write_predictions=True, groups=None, budget=BUDGET):
    emb, y = _emb_labels()
    source_ranked, target_ranked = _orders()
    rows, preds = runstate._probe_cell_label_access(
        EV.run_probes_label_access, emb, TRAIN, VAL, POOL, TEST,
        source_ranked, target_ranked, budget, y, groups, dict(BASE_META), 0, "logistic",
        write_predictions=write_predictions,
    )
    return rows, preds


def _expected_keys():
    return {(*BASE_KEY, "label_access", b, es, route) for (route, b, es) in runstate.label_access_expected_rows()}


def _by_route(rows):
    return {(r["label_access_route"], r["label_budget"]): r for r in rows}


# --------------------------------------------------------------------------- #
# runtime: the cell dispatches the whole suite, mapping frozen orders -> positions
# --------------------------------------------------------------------------- #
def test_runtime_emits_full_suite_on_frozen_target_test():
    rows, _ = _run()
    expected = runstate.label_access_expected_rows()
    assert len(rows) == len(expected)
    assert all(r["evaluation_split"] == SA.EVAL_TARGET_TEST for r in rows)
    # every route is scored on the SAME frozen target_test -- never the target pool, never the region.
    assert {r["n_test"] for r in rows} == {len(TEST)}
    # the frozen orders were mapped to positions and consumed: label accounting matches the contract.
    by = _by_route(rows)
    for f in SA.ALLOCATION_PERCENTS:
        k = SA.allocation_target_count(f, BUDGET)
        r = by[(SA.ROUTE_FIXED_BUDGET_ALLOCATION, f)]
        assert (r["n_source_labels"], r["n_target_labels"], r["n_total_labels"]) == (BUDGET - k, k, BUDGET)
    for k in SA.LABEL_ACCESS_COUNTS:
        r = by[(SA.ROUTE_SOURCE_PLUS_TARGET, k)]
        assert (r["n_source_labels"], r["n_total_labels"]) == (len(TRAIN), len(TRAIN) + k)
    assert by[(SA.ROUTE_TARGET_ONLY_FULL, 0)]["n_target_labels"] == len(POOL)
    assert by[(SA.ROUTE_SOURCE_PLUS_TARGET_FULL, 0)]["n_total_labels"] == len(TRAIN) + len(POOL)
    assert all(r["benchmark_budget"] == BUDGET and r["allocation_total_budget"] == BUDGET for r in rows)
    assert all((r["n_source_pool"], r["n_target_pool"]) == (len(TRAIN), len(POOL)) for r in rows)


def test_runtime_emits_no_source_only_and_no_complete_target_row():
    """Label access no longer refits a source-only probe (the ordinary full-source E1 row is that fit)
    and no longer emits a complete-target diagnostic -- the canonical population is the frozen
    target_test."""
    rows, preds = _run()
    assert not any(r["evaluation_split"] == SA.EVAL_COMPLETE_TARGET for r in rows)
    assert not any(p["evaluation_split"] == SA.EVAL_COMPLETE_TARGET for p in preds)
    retired = {"source_only", "matched_source", "matched_target", "fixed_total_mixed"}
    assert not (retired & {r["label_access_route"] for r in rows})
    assert not (retired & {p["label_access_route"] for p in preds})


def test_runtime_config_globals_drive_the_emitted_route_set(monkeypatch):
    """The run config lives in ``runstate`` module globals; the emitted rows must follow them exactly
    (and must keep matching ``runstate.label_access_expected_rows()`` under a non-default config)."""
    monkeypatch.setattr(runstate, "ALLOCATION_PERCENTS", (0, 50, 100))
    monkeypatch.setattr(runstate, "ADDITIVE_TARGET_COUNTS", (5, 10))
    monkeypatch.setattr(runstate, "FULL_TARGET_REFERENCE", False)
    rows, _ = _run()
    emitted = {(r["label_access_route"], r["label_budget"], r["evaluation_split"]) for r in rows}
    assert emitted == set(runstate.label_access_expected_rows())
    assert {artifacts.cell_key(r) for r in rows} == _expected_keys()
    assert SA.ROUTE_TARGET_ONLY_FULL not in {r["label_access_route"] for r in rows}
    assert {r["label_budget"] for r in rows if r["label_access_route"] == SA.ROUTE_FIXED_BUDGET_ALLOCATION} \
        == {0, 50, 100}


def test_runtime_controlled_cap_shrinks_the_realized_budget(monkeypatch):
    monkeypatch.setattr(runstate, "CONTROLLED_TOTAL_BUDGET_CAP", 20)
    rows, _ = _run()
    by = _by_route(rows)
    for f in SA.ALLOCATION_PERCENTS:
        k = SA.allocation_target_count(f, 20)
        r = by[(SA.ROUTE_FIXED_BUDGET_ALLOCATION, f)]
        assert (r["n_source_labels"], r["n_target_labels"], r["n_total_labels"]) == (20 - k, k, 20)
    assert all(r["allocation_total_budget"] == 20 for r in rows)
    assert all(r["controlled_budget_cap"] == 20 for r in rows)
    assert all(r["benchmark_budget"] == BUDGET for r in rows)   # B_d stays recorded verbatim
    # the cap governs the BUDGET only -- the additive routes still hold the complete source pool.
    assert by[(SA.ROUTE_SOURCE_PLUS_TARGET, 25)]["n_source_labels"] == len(TRAIN)


# --------------------------------------------------------------------------- #
# identity: every row carries the full 9-field key; cell_key and budget_row_key agree
# --------------------------------------------------------------------------- #
def test_runtime_rows_carry_the_nine_field_identity():
    rows, _ = _run()
    assert {artifacts.cell_key(r) for r in rows} == _expected_keys()
    # cell_key (readers) and budget_row_key (resume/prune) are the SAME 9-tuple for every row -- if they
    # ever diverged, a pruned row would not match its planned key and resume would loop or duplicate.
    for r in rows:
        assert artifacts.cell_key(r) == runstate.budget_row_key(r)
    # every label-access row carries a non-empty route from the canonical set; no row leaks the default "".
    assert all(r["label_access_route"] in set(SA.LABEL_ACCESS_ROUTES) for r in rows)
    assert all(r["budget_type"] == "label_access" for r in rows)


def test_routes_sharing_a_budget_are_distinct_keys():
    """fixed_budget_allocation(f) is an integer PERCENT and source_plus_target(k) an absolute COUNT, so
    they collide numerically (25 and 50 are in both axes). They share label_budget and
    evaluation_split; only ``label_access_route`` disambiguates them. If it did not, resume would treat
    one as the other."""
    rows, _ = _run()
    shared = set(SA.ALLOCATION_PERCENTS) & set(SA.LABEL_ACCESS_COUNTS)
    assert shared, "the two budget axes must actually collide for this invariant to matter"
    for b in shared:
        alloc = next(r for r in rows if r["label_access_route"] == SA.ROUTE_FIXED_BUDGET_ALLOCATION
                     and r["label_budget"] == b and r["evaluation_split"] == SA.EVAL_TARGET_TEST)
        add = next(r for r in rows if r["label_access_route"] == SA.ROUTE_SOURCE_PLUS_TARGET
                   and r["label_budget"] == b and r["evaluation_split"] == SA.EVAL_TARGET_TEST)
        assert runstate.budget_row_key(alloc) != runstate.budget_row_key(add)
        assert artifacts.cell_key(alloc) != artifacts.cell_key(add)
        # and their supervision genuinely differs -- the two 25s are different experiments
        assert (alloc["n_source_labels"], alloc["n_total_labels"]) != (add["n_source_labels"], add["n_total_labels"])


# --------------------------------------------------------------------------- #
# prediction: per-sample rows carry the route + the right sample-id population
# --------------------------------------------------------------------------- #
def test_predictions_carry_route_and_correct_sample_population():
    rows, preds = _run(write_predictions=True)
    assert preds
    routes = {r["label_access_route"] for r in rows}
    assert all(p["label_access_route"] in routes for p in preds)
    # every route's predictions are over the frozen target_test and nothing else.
    tt_ids = {int(i) for i in TEST}
    assert {p["evaluation_split"] for p in preds} == {SA.EVAL_TARGET_TEST}
    assert {p["sample_id"] for p in preds} == tt_ids
    assert len(preds) == len(rows) * len(TEST)


def test_write_predictions_false_yields_no_preds():
    rows, preds = _run(write_predictions=False)
    assert rows and preds == []


def test_prediction_metadata_is_exact_per_route():
    """Every prediction row carries the EXACT supervision metadata of its route -- not just the route
    name -- so a per-sample reader can attribute a prediction without re-joining to summary.csv. The two
    same-budget routes (allocation@25% vs additive@25) must show DIFFERENT source/total counts."""
    _rows, preds = _run(write_predictions=True)
    S = len(TRAIN)
    want = {"budget_type": "label_access", "label_budget": 25, "evaluation_split": SA.EVAL_TARGET_TEST,
            "label_budget_unit": "samples"}
    k_alloc = SA.allocation_target_count(25, BUDGET)

    alloc = [p for p in preds if p["label_access_route"] == SA.ROUTE_FIXED_BUDGET_ALLOCATION
             and p["label_budget"] == 25]
    assert len(alloc) == len(TEST)  # one per frozen target_test sample
    for p in alloc:
        assert {k: p[k] for k in want} == want
        assert (p["n_source_labels"], p["n_target_labels"], p["n_total_labels"]) == (
            BUDGET - k_alloc, k_alloc, BUDGET)

    add = [p for p in preds if p["label_access_route"] == SA.ROUTE_SOURCE_PLUS_TARGET
           and p["label_budget"] == 25]
    assert len(add) == len(TEST)
    for p in add:
        assert {k: p[k] for k in want} == want
        assert (p["n_source_labels"], p["n_target_labels"], p["n_total_labels"]) == (S, 25, S + 25)


def test_reference_route_predictions_carry_their_own_supervision():
    """The two full references are ordinary scored routes now (there is no reused-diagnostic row), so
    their predictions must carry their own supervision counts, not another route's."""
    _rows, preds = _run(write_predictions=True)
    S, P = len(TRAIN), len(POOL)
    tof = [p for p in preds if p["label_access_route"] == SA.ROUTE_TARGET_ONLY_FULL]
    spt_full = [p for p in preds if p["label_access_route"] == SA.ROUTE_SOURCE_PLUS_TARGET_FULL]
    assert len(tof) == len(spt_full) == len(TEST)
    for p in tof:
        assert (p["n_source_labels"], p["n_target_labels"], p["n_total_labels"]) == (0, P, P)
        assert p["label_budget"] == 0
    for p in spt_full:
        assert (p["n_source_labels"], p["n_target_labels"], p["n_total_labels"]) == (S, P, S + P)
        assert p["label_budget"] == 0


# --------------------------------------------------------------------------- #
# completeness + resume: the planned key set matches, and one dropped route is caught precisely
# --------------------------------------------------------------------------- #
def test_completeness_ok_on_the_full_suite():
    rows, _ = _run()
    report = artifacts.completeness(_expected_keys(), rows)
    assert report["ok"]
    assert report["missing"] == [] and report["unexpected"] == [] and report["duplicate"] == []


def test_dropping_one_route_is_caught_and_does_not_implicate_its_budget_twin():
    rows, _ = _run()
    expected = _expected_keys()
    dropped_key = (*BASE_KEY, "label_access", 25, SA.EVAL_TARGET_TEST, SA.ROUTE_FIXED_BUDGET_ALLOCATION)
    kept = [r for r in rows if artifacts.cell_key(r) != dropped_key]
    assert len(kept) == len(rows) - 1
    report = artifacts.completeness(expected, kept)
    assert not report["ok"]
    assert report["missing"] == [list(dropped_key)]
    # the same-budget twin (source_plus_target(25)) is still present -- the route, not the budget, keys it.
    twin_key = (*BASE_KEY, "label_access", 25, SA.EVAL_TARGET_TEST, SA.ROUTE_SOURCE_PLUS_TARGET)
    assert list(twin_key) not in report["missing"]


def test_resume_gate_reruns_only_when_a_planned_route_is_absent():
    """main dispatches ONE label-access job per cell iff any planned cell_key is missing from ``done``.
    A run with every route present re-runs nothing; dropping a single route re-arms the cell."""
    cell_keys = [(*BASE_KEY, "label_access", b, es, route)
                 for (route, b, es) in runstate.label_access_expected_rows()]
    done_all = set(cell_keys)
    assert all(k in done_all for k in cell_keys)  # complete -> no rerun
    done_partial = set(cell_keys) - {cell_keys[1]}
    assert not all(k in done_partial for k in cell_keys)  # a hole -> rerun the cell


# --------------------------------------------------------------------------- #
# identity mapping is fail-closed: a foreign current index is a hard error, never silently dropped
# --------------------------------------------------------------------------- #
def test_to_positions_hard_fails_on_index_outside_its_partition():
    emb, y = _emb_labels()
    source_ranked, target_ranked = _orders()
    foreign = source_ranked.copy()
    foreign[0] = TEST[0]  # a target_test index can never be in the source-train partition
    with pytest.raises(ValueError, match="not in its partition"):
        runstate._probe_cell_label_access(
            EV.run_probes_label_access, emb, TRAIN, VAL, POOL, TEST,
            foreign, target_ranked, BUDGET, y, None, dict(BASE_META), 0, "logistic",
        )


def test_to_positions_hard_fails_on_foreign_target_index():
    emb, y = _emb_labels()
    source_ranked, target_ranked = _orders()
    foreign = target_ranked.copy()
    foreign[0] = TRAIN[0]  # a source-train index can never be in the target label pool
    with pytest.raises(ValueError, match="not in its partition"):
        runstate._probe_cell_label_access(
            EV.run_probes_label_access, emb, TRAIN, VAL, POOL, TEST,
            source_ranked, foreign, BUDGET, y, None, dict(BASE_META), 0, "logistic",
        )


def test_non_label_access_row_keys_with_empty_route():
    """A source-sweep row never sets label_access_route; both keyers must default it to "" so it keys
    stably against the planned (..., "") source keys -- never colliding with a real route."""
    src_row = {"seed": 0, "split_regime": SA.LABEL_ACCESS_REGIME, "holdout": "kenya", "method": "erm",
               "probe_family": "logistic", "budget_type": "source", "label_budget": 1.0,
               "evaluation_split": "test"}
    assert artifacts.cell_key(src_row)[-1] == ""
    assert runstate.budget_row_key(src_row)[-1] == ""
    assert artifacts.cell_key(src_row) == runstate.budget_row_key(src_row)
