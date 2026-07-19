"""Stage-3 runtime dispatch + identity threading for the geographic label-access suite.

The sweep itself (13 routes, calibration routing, cap, order validation, diagnostic no-refit) is covered
by ``test_label_access_sweep.py``; the frozen artifact (generation/validation/load) by
``test_label_access.py``; and the through-``main`` publish path by
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
# suite is feasible: source S=60 and pool P=55 both clear max(LABEL_ACCESS_COUNTS)=50.
TRAIN = np.arange(0, 60, dtype=np.int64)
VAL = np.arange(60, 70, dtype=np.int64)
POOL = np.arange(70, 125, dtype=np.int64)
TEST = np.arange(125, 137, dtype=np.int64)
N = 137

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
    """The three frozen orders as CURRENT indices in ascending-rank order (as ``load_label_access``
    returns them): two independent source permutations + one target-pool permutation."""
    rng = np.random.default_rng(0)
    return rng.permutation(TRAIN), rng.permutation(TRAIN), rng.permutation(POOL)


def _run(write_predictions=True, groups=None):
    emb, y = _emb_labels()
    matched, fixed, target = _orders()
    rows, preds = runstate._probe_cell_label_access(
        EV.run_probes_label_access, emb, TRAIN, VAL, POOL, TEST,
        matched, fixed, target, y, groups, dict(BASE_META), 0, "logistic",
        write_predictions=write_predictions,
    )
    return rows, preds


def _expected_keys():
    return {(*BASE_KEY, "label_access", b, es, route) for (route, b, es) in SA.label_access_expected_rows()}


# --------------------------------------------------------------------------- #
# runtime: the cell dispatches the whole suite, mapping frozen orders -> positions
# --------------------------------------------------------------------------- #
def test_runtime_emits_full_suite_on_frozen_target_test():
    rows, _ = _run()
    tt = [r for r in rows if r["evaluation_split"] == SA.EVAL_TARGET_TEST]
    diag = [r for r in rows if r["evaluation_split"] == SA.EVAL_COMPLETE_TARGET]
    assert len(tt) == 13 and len(diag) == 1
    # every route is scored on the SAME frozen target_test; the diagnostic on the whole target region.
    assert {r["n_test"] for r in tt} == {len(TEST)}
    assert diag[0]["n_test"] == len(POOL) + len(TEST)
    # the frozen orders were mapped to positions and consumed: label accounting matches the contract.
    by = {(r["label_access_route"], r["label_budget"]): r for r in tt}
    assert by[(SA.ROUTE_SOURCE_ONLY, 0)]["n_source_labels"] == len(TRAIN)
    for k in SA.LABEL_ACCESS_COUNTS:
        assert by[(SA.ROUTE_SOURCE_PLUS_TARGET, k)]["n_total_labels"] == len(TRAIN) + k
        assert by[(SA.ROUTE_FIXED_TOTAL_MIXED, k)]["n_total_labels"] == len(TRAIN)


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
    routes = {SA.ROUTE_SOURCE_ONLY, SA.ROUTE_SOURCE_PLUS_TARGET, SA.ROUTE_TARGET_ONLY_FULL,
              SA.ROUTE_SOURCE_PLUS_TARGET_FULL, SA.ROUTE_MATCHED_SOURCE, SA.ROUTE_MATCHED_TARGET,
              SA.ROUTE_FIXED_TOTAL_MIXED}
    assert all(r["label_access_route"] in routes for r in rows)
    assert all(r["budget_type"] == "label_access" for r in rows)


def test_routes_sharing_a_budget_are_distinct_keys():
    """source_plus_target(k) and fixed_total_mixed(k) share label_budget=k and evaluation_split; only the
    route disambiguates them. If it did not, resume would treat one as the other."""
    rows, _ = _run()
    for k in SA.LABEL_ACCESS_COUNTS:
        spt = next(r for r in rows if r["label_access_route"] == SA.ROUTE_SOURCE_PLUS_TARGET
                   and r["label_budget"] == k and r["evaluation_split"] == SA.EVAL_TARGET_TEST)
        ftm = next(r for r in rows if r["label_access_route"] == SA.ROUTE_FIXED_TOTAL_MIXED
                   and r["label_budget"] == k and r["evaluation_split"] == SA.EVAL_TARGET_TEST)
        assert runstate.budget_row_key(spt) != runstate.budget_row_key(ftm)


# --------------------------------------------------------------------------- #
# prediction: per-sample rows carry the route + the right sample-id population
# --------------------------------------------------------------------------- #
def test_predictions_carry_route_and_correct_sample_population():
    rows, preds = _run(write_predictions=True)
    assert preds
    routes = {r["label_access_route"] for r in rows}
    assert all(p["label_access_route"] in routes for p in preds)
    # route predictions are over the frozen target_test; the diagnostic over the complete target region.
    tt_ids = {int(i) for i in TEST}
    full_ids = {int(i) for i in np.concatenate([POOL, TEST])}
    tt_preds = [p for p in preds if p["evaluation_split"] == SA.EVAL_TARGET_TEST]
    diag_preds = [p for p in preds if p["evaluation_split"] == SA.EVAL_COMPLETE_TARGET]
    assert {p["sample_id"] for p in tt_preds} == tt_ids
    assert {p["sample_id"] for p in diag_preds} == full_ids
    # the diagnostic predictions come only from source_only (the reused fit), never a second route.
    assert {p["label_access_route"] for p in diag_preds} == {SA.ROUTE_SOURCE_ONLY}


def test_write_predictions_false_yields_no_preds():
    rows, preds = _run(write_predictions=False)
    assert rows and preds == []


def test_prediction_metadata_is_exact_per_route():
    """Every prediction row carries the EXACT supervision metadata of its route -- not just the route
    name -- so a per-sample reader can attribute a prediction without re-joining to summary.csv. The two
    same-budget routes (additive vs fixed-total at k=25) must show DIFFERENT source/total counts."""
    _rows, preds = _run(write_predictions=True)
    S = len(TRAIN)
    want = {"budget_type": "label_access", "label_budget": 25, "evaluation_split": SA.EVAL_TARGET_TEST,
            "label_budget_unit": "samples"}

    spt = [p for p in preds if p["label_access_route"] == SA.ROUTE_SOURCE_PLUS_TARGET and p["label_budget"] == 25]
    assert len(spt) == len(TEST)  # one per frozen target_test sample
    for p in spt:
        assert {k: p[k] for k in want} == want
        assert (p["n_source_labels"], p["n_target_labels"], p["n_total_labels"]) == (S, 25, S + 25)

    ftm = [p for p in preds if p["label_access_route"] == SA.ROUTE_FIXED_TOTAL_MIXED and p["label_budget"] == 25]
    assert len(ftm) == len(TEST)
    for p in ftm:
        assert {k: p[k] for k in want} == want
        assert (p["n_source_labels"], p["n_target_labels"], p["n_total_labels"]) == (S - 25, 25, S)


def test_diagnostic_predictions_carry_source_only_supervision():
    """The complete-target diagnostic scores the source_only FIT, so its predictions must carry the
    source-only supervision metadata (S source, 0 target, total S) -- never a distinct route's."""
    _rows, preds = _run(write_predictions=True)
    S = len(TRAIN)
    diag = [p for p in preds if p["evaluation_split"] == SA.EVAL_COMPLETE_TARGET]
    assert diag
    for p in diag:
        assert p["label_access_route"] == SA.ROUTE_SOURCE_ONLY
        assert p["budget_type"] == "label_access" and p["label_budget"] == 0
        assert (p["n_source_labels"], p["n_target_labels"], p["n_total_labels"]) == (S, 0, S)
        assert p["label_budget_unit"] == "samples"


# --------------------------------------------------------------------------- #
# diagnostic-reuse: the complete-target row is single, is source_only, and adds no route
# --------------------------------------------------------------------------- #
def test_diagnostic_is_single_source_only_complete_target():
    rows, _ = _run()
    diag = [r for r in rows if r["evaluation_split"] == SA.EVAL_COMPLETE_TARGET]
    assert len(diag) == 1
    assert diag[0]["label_access_route"] == SA.ROUTE_SOURCE_ONLY
    # it sits on a DISTINCT evaluation split from every scored route, so Stage-5 paired contrasts (which
    # operate on target_test) exclude it by construction rather than by an ad-hoc filter.
    assert diag[0]["evaluation_split"] == SA.EVAL_COMPLETE_TARGET != SA.EVAL_TARGET_TEST
    # the source_only route appears on target_test exactly once too (the diagnostic is NOT a duplicate fit).
    src_only_tt = [r for r in rows if r["label_access_route"] == SA.ROUTE_SOURCE_ONLY
                   and r["evaluation_split"] == SA.EVAL_TARGET_TEST]
    assert len(src_only_tt) == 1


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
    dropped_key = (*BASE_KEY, "label_access", 25, SA.EVAL_TARGET_TEST, SA.ROUTE_FIXED_TOTAL_MIXED)
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
                 for (route, b, es) in SA.label_access_expected_rows()]
    done_all = set(cell_keys)
    assert all(k in done_all for k in cell_keys)  # complete -> no rerun
    done_partial = set(cell_keys) - {cell_keys[7]}
    assert not all(k in done_partial for k in cell_keys)  # a hole -> rerun the cell


# --------------------------------------------------------------------------- #
# identity mapping is fail-closed: a foreign current index is a hard error, never silently dropped
# --------------------------------------------------------------------------- #
def test_to_positions_hard_fails_on_index_outside_its_partition():
    emb, y = _emb_labels()
    matched, fixed, target = _orders()
    foreign = matched.copy()
    foreign[0] = TEST[0]  # a target_test index can never be in the source-train partition
    with pytest.raises(ValueError, match="not in its partition"):
        runstate._probe_cell_label_access(
            EV.run_probes_label_access, emb, TRAIN, VAL, POOL, TEST,
            foreign, fixed, target, y, None, dict(BASE_META), 0, "logistic",
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
