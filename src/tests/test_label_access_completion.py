"""Stage-3 artifact-integrity: label-access completion validation, failure-summary rendering, and the
run-manifest contract.

Completeness (the 9-field cell key) certifies that every planned (route, budget, split) row is PRESENT;
it says nothing about whether the supervision COUNTS on those rows are internally consistent. These tests
pin the SEMANTIC layer that runs alongside it -- a tampered count or unit keeps the key intact, so it must
be a separate guard that still blocks ``run_complete.json``.

Under the fixed-budget contract the accounting the validator re-derives is arithmetic, not merely
self-consistent: every allocation row must agree on ONE realized budget ``B`` and satisfy
``n_target == round_half_up(f% * B)``, ``n_source == B - k``, ``n_total == B``; additive@k must hold the
COMPLETE source pool and add exactly ``k``; the retired routes must never appear; label access must emit
no ``complete_target`` row (the deployment estimand rides on the ordinary full-source E1 row); and
exactly ONE E1 row must exist per cell, since the contrasts subtract it instead of refitting a
source-only probe. They also lock the readable manifest contract and the route/split-aware failure
summary.
"""

from __future__ import annotations

import pytest

from evals import split_artifacts as SA
from utils import artifacts, runstate
from utils import ioutils as IOU

BASE = {
    "model": "raw", "benchmark": "cropharvest", "method": "erm", "probe_family": "logistic",
    "split_regime": SA.LABEL_ACCESS_REGIME, "holdout": "kenya", "seed": 0, "domain_basis": "geo",
    "budget_type": "label_access",
}

#: The realized budget B, the COMPLETE source pool S, the target label pool P and the frozen target_test
#: size t of the reference cell. B=50 puts 25% and 75% exactly on .5, so half-up rounding (k=13 / k=38)
#: is distinguishable from banker's rounding here.
B, S_POOL, P_POOL, N_TEST = 50, 60, 55, 12


def _k(f, budget=B):
    return SA.allocation_target_count(f, budget)


def _counts(route, budget, b=B, s=S_POOL, p=P_POOL):
    """The correct supervision (n_source, n_target, n_total) for one route -- the accounting the
    semantic validator enforces."""
    if route == SA.ROUTE_FIXED_BUDGET_ALLOCATION:
        k = _k(budget, b)
        return (b - k, k, b)
    return {
        SA.ROUTE_SOURCE_PLUS_TARGET: (s, budget, s + budget),
        SA.ROUTE_TARGET_ONLY_FULL: (0, p, p),
        SA.ROUTE_SOURCE_PLUS_TARGET_FULL: (s, p, s + p),
    }[route]


def _e1_row(**over):
    """The ordinary full-source geographic row (E1: budget_type=source, label_budget=1.0,
    evaluation_split=test). Label access does NOT refit a source-only probe -- the contrasts subtract
    this row, and the whole-region ``complete_target`` deployment score rides on it -- so exactly one
    must exist per cell."""
    return {**BASE, "budget_type": "source", "label_budget": 1.0, "evaluation_split": "test",
            "label_access_route": "", "n_test": N_TEST, "f1": 1.0, **over}


def _valid_rows(b=B, s=S_POOL, p=P_POOL, t=N_TEST):
    """The rows one fully-eligible cell emits -- every planned label-access row (derived from
    label_access_expected_rows() so the cell keys match the completeness plan exactly) plus the single
    E1 row the suite reuses. Every route is scored on the frozen target_test; there is deliberately no
    source_only row and no complete_target diagnostic."""
    rows = []
    for route, budget, es in SA.label_access_expected_rows():
        ns, nt, ntot = _counts(route, budget, b, s, p)
        rows.append({**BASE, "label_access_route": route, "label_budget": budget, "evaluation_split": es,
                     "n_source_labels": ns, "n_target_labels": nt, "n_total_labels": ntot, "n_test": t,
                     "allocation_total_budget": b, "controlled_budget_cap": 0, "benchmark_budget": b,
                     "n_source_pool": s, "n_target_pool": p,
                     "label_budget_unit": SA.LABEL_ACCESS_TABULAR_UNIT, "f1": 1.0})
    rows.append(_e1_row())
    # The whole-region DEPLOYMENT estimand: the same full-source probe scored on the complete target
    # region. It is an ordinary budget_type=source row, NOT a label-access route -- so the validator must
    # neither reject it as a stray complete_target row nor count it as a second E1.
    rows.append(_e1_row(evaluation_split=SA.EVAL_COMPLETE_TARGET, n_test=t + p))
    return rows


def _find(rows, route, budget=0, es=SA.EVAL_TARGET_TEST):
    return next(r for r in rows if r.get("label_access_route") == route
               and r.get("label_budget") == budget and r.get("evaluation_split") == es)


def _keys(rows):
    return {artifacts.cell_key(r) for r in rows}


#: A random_id full-source in-distribution row -- the in-distribution reference every real label-access
#: run also produces. Included so write_run_complete's delta validation can resolve.
_ANCHOR = {
    "model": "raw", "benchmark": "cropharvest", "method": "erm", "probe_family": "logistic",
    "split_regime": "random_id", "holdout": "random_id", "seed": 0, "domain_basis": "geo",
    "budget_type": "source", "evaluation_split": "test", "label_budget": 1.0, "label_access_route": "",
    "f1": 1.0,
}


def _publish(tmp_path, rows, keys, **kw):
    """Write the required artifacts + the Stage-5 contrast artifacts + call write_run_complete (raises
    IncompleteRunError on any fault). A real label-access run also carries the random_id anchor and the
    paired-contrast artifacts, so both are added here for the completion contract to hold."""
    from evals import contrasts

    rows = [*rows, _ANCHOR]
    keys = set(keys) | {artifacts.cell_key(_ANCHOR)}
    IOU.append_jsonl(tmp_path / "probe_results.jsonl", rows)
    for name in ("probe_results.csv", "summary.csv", artifacts.ENVIRONMENT_FILE):
        (tmp_path / name).write_text("{}\n")
    # Real delta table: the anchor + geographic label-access rows above make one mandatory.
    IOU.write_csv(tmp_path / "deltas.csv", IOU.compute_deltas(rows, ["f1"]))
    contrasts.compute_and_write(tmp_path, rows)
    return artifacts.write_run_complete(tmp_path, run_manifest_sha256="sig", expected_keys=keys, rows=rows, **kw)


# --------------------------------------------------------------------------- #
# semantic validation: valid suite passes; each tampered invariant is caught
# --------------------------------------------------------------------------- #
def test_semantic_validation_passes_on_a_valid_suite():
    assert artifacts._validate_label_access_semantics(_valid_rows()) == []


def test_semantic_validation_is_vacuous_without_label_access_rows():
    assert artifacts._validate_label_access_semantics([{"budget_type": "source", "label_budget": 1.0}]) == []


def test_tampered_balance_is_caught():
    rows = _valid_rows()
    _find(rows, SA.ROUTE_FIXED_BUDGET_ALLOCATION, 0)["n_total_labels"] = B + 1  # != n_source + n_target
    assert any("n_total" in p for p in artifacts._validate_label_access_semantics(rows))


def test_tampered_unit_is_caught():
    rows = _valid_rows()
    _find(rows, SA.ROUTE_TARGET_ONLY_FULL)["label_budget_unit"] = "target_patches"
    assert any("unit" in p for p in artifacts._validate_label_access_semantics(rows))


# --- the fixed-budget allocation arithmetic (the whole headline claim) -----------------------------
def test_valid_allocation_rows_use_half_up_rounding():
    """The reference suite is built at k = round_half_up(f% x 50): 0, 13, 25, 38, 50. Banker's rounding
    would put 12 at 25%, so a suite built the other way would NOT validate."""
    rows = _valid_rows()
    assert [_k(f) for f in SA.ALLOCATION_PERCENTS] == [0, 13, 25, 38, 50]
    assert round(0.25 * B) == 12 != _k(25)
    for f in SA.ALLOCATION_PERCENTS:
        r = _find(rows, SA.ROUTE_FIXED_BUDGET_ALLOCATION, f)
        assert r["n_target_labels"] == _k(f)
        assert r["n_source_labels"] + r["n_target_labels"] == r["n_total_labels"] == B


def test_tampered_allocation_target_count_is_caught():
    """A wrong k that still BALANCES and still totals B -- only re-deriving round_half_up(f% x B)
    catches it. 12 is exactly what banker's rounding would have produced at 25%."""
    rows = _valid_rows()
    r = _find(rows, SA.ROUTE_FIXED_BUDGET_ALLOCATION, 25)
    r["n_target_labels"], r["n_source_labels"] = 12, B - 12
    probs = artifacts._validate_label_access_semantics(rows)
    assert any("round_half_up(25% x 50) = 13" in p for p in probs)


def test_tampered_allocation_total_is_caught():
    """Balanced, but the fit no longer spends exactly the realized budget -- the fixed-budget claim is
    that TOTAL supervision never moves, only its composition."""
    rows = _valid_rows()
    r = _find(rows, SA.ROUTE_FIXED_BUDGET_ALLOCATION, 50)
    r["n_source_labels"], r["n_total_labels"] = r["n_source_labels"] + 5, B + 5
    probs = artifacts._validate_label_access_semantics(rows)
    assert any("n_source" in p and "B - k" in p for p in probs)
    assert any("realized budget" in p for p in probs)


def test_allocation_rows_disagreeing_on_the_realized_budget_are_caught():
    """Every allocation point must be drawn at ONE realized budget B; two points at different budgets
    would confound budget size with composition even if each is internally consistent."""
    rows = _valid_rows()
    r = _find(rows, SA.ROUTE_FIXED_BUDGET_ALLOCATION, 100)
    other = 40
    r["allocation_total_budget"] = other
    r["n_target_labels"], r["n_source_labels"], r["n_total_labels"] = other, 0, other
    probs = artifacts._validate_label_access_semantics(rows)
    assert any("disagree on the realized budget" in p for p in probs)


# --- the additive appendix routes ------------------------------------------------------------------
def test_tampered_additive_count_is_caught():
    rows = _valid_rows()
    r = _find(rows, SA.ROUTE_SOURCE_PLUS_TARGET, 25)
    r["n_target_labels"], r["n_total_labels"] = 24, S_POOL + 24  # balances, but no longer +25
    assert any("source_plus_target@25" in p for p in artifacts._validate_label_access_semantics(rows))


def test_additive_route_not_on_the_complete_source_pool_is_caught():
    """The additive question is 'hold the COMPLETE source pool and add k' -- a route that quietly trained
    on a budgeted subset would be an allocation point wearing an additive label."""
    rows = _valid_rows()
    r = _find(rows, SA.ROUTE_SOURCE_PLUS_TARGET, 10)
    r["n_source_labels"], r["n_total_labels"] = B, B + 10       # B, not the complete pool S
    assert any("COMPLETE source pool" in p for p in artifacts._validate_label_access_semantics(rows))


# --- the full references ---------------------------------------------------------------------------
def test_tampered_target_only_full_source_leak_is_caught():
    rows = _valid_rows()
    r = _find(rows, SA.ROUTE_TARGET_ONLY_FULL)
    r["n_source_labels"], r["n_total_labels"] = 3, P_POOL + 3   # target_only_full must train on 0 source
    assert any("target_only_full" in p for p in artifacts._validate_label_access_semantics(rows))


def test_source_plus_target_full_not_on_the_complete_pool_is_caught():
    rows = _valid_rows()
    r = _find(rows, SA.ROUTE_SOURCE_PLUS_TARGET_FULL)
    r["n_source_labels"], r["n_total_labels"] = 50, 50 + P_POOL
    assert any("source_plus_target_full" in p for p in artifacts._validate_label_access_semantics(rows))


# --- retired routes / retired evaluation split / the E1 anchor -------------------------------------
@pytest.mark.parametrize("retired", ["source_only", "matched_source", "matched_target", "fixed_total_mixed"])
def test_retired_route_name_is_caught(retired):
    """The four retired routes were replaced by the single nested allocation curve. A table still
    carrying one is a stale runtime, not a valid suite -- the name alone must fail."""
    rows = _valid_rows()
    rows.append({**BASE, "label_access_route": retired, "label_budget": 0,
                 "evaluation_split": SA.EVAL_TARGET_TEST, "n_source_labels": S_POOL,
                 "n_target_labels": 0, "n_total_labels": S_POOL, "n_test": N_TEST,
                 "allocation_total_budget": B, "controlled_budget_cap": 0, "benchmark_budget": B,
                 "n_source_pool": S_POOL, "n_target_pool": P_POOL,
                 "label_budget_unit": SA.LABEL_ACCESS_TABULAR_UNIT, "f1": 1.0})
    probs = artifacts._validate_label_access_semantics(rows)
    assert any("retired route" in p and retired in p for p in probs)


def test_stray_complete_target_label_access_row_is_caught():
    """LABEL_ACCESS_EVAL_SPLITS is (target_test,) only. The whole-region deployment estimand now rides on
    the ordinary full-source E1 row, so a label-access complete_target row is a stale emission."""
    assert SA.LABEL_ACCESS_EVAL_SPLITS == (SA.EVAL_TARGET_TEST,)
    rows = _valid_rows()
    stray = dict(_find(rows, SA.ROUTE_TARGET_ONLY_FULL))
    stray["evaluation_split"] = SA.EVAL_COMPLETE_TARGET
    rows.append(stray)
    probs = artifacts._validate_label_access_semantics(rows)
    assert any(SA.EVAL_COMPLETE_TARGET in p for p in probs)


def test_the_deployment_estimand_rides_on_the_full_source_row_not_a_label_access_route():
    """The whole-region ``complete_target`` score is emitted as an ordinary full-source row
    (budget_type=source), so it is exempt from the label-access complete_target ban and does not count
    as a second E1 -- only the ``evaluation_split=test`` leg does."""
    rows = _valid_rows()
    deployment = [r for r in rows
                  if r.get("evaluation_split") == SA.EVAL_COMPLETE_TARGET]
    assert len(deployment) == 1
    assert deployment[0]["budget_type"] == "source" and deployment[0]["label_access_route"] == ""
    assert artifacts._validate_label_access_semantics(rows) == []
    # ... and dropping it does not disturb the label-access accounting either
    assert artifacts._validate_label_access_semantics([r for r in rows if r not in deployment]) == []


def test_missing_e1_row_is_caught():
    rows = [r for r in _valid_rows() if r["budget_type"] != "source"]
    probs = artifacts._validate_label_access_semantics(rows)
    assert any("exactly ONE full-source geographic row" in p and "found 0" in p for p in probs)


def test_duplicated_e1_row_is_caught():
    """Two E1 rows make the subtrahend of every contrast arbitrary, so a duplicate is as fatal as an
    absence."""
    rows = [*_valid_rows(), _e1_row(f1=0.9)]
    probs = artifacts._validate_label_access_semantics(rows)
    assert any("exactly ONE full-source geographic row" in p and "found 2" in p for p in probs)


def test_e1_row_from_another_cell_does_not_satisfy_this_one():
    rows = [r for r in _valid_rows() if r["budget_type"] != "source"]
    rows.append(_e1_row(holdout="togo"))          # right shape, wrong cell
    assert any("found 0" in p for p in artifacts._validate_label_access_semantics(rows))


# --- strict integer counts -------------------------------------------------------------------------
def test_non_integral_float_count_is_rejected():
    rows = _valid_rows()
    _find(rows, SA.ROUTE_FIXED_BUDGET_ALLOCATION, 0)["n_source_labels"] = 50.5
    probs = artifacts._validate_label_access_semantics(rows)
    assert any("n_source_labels=50.5" in p and "not an integer" in p for p in probs)


def test_integral_float_count_is_accepted():
    """JSONL round-trips ints as floats, so 50.0 must NOT be rejected -- only NON-integral floats are."""
    rows = _valid_rows()
    r = _find(rows, SA.ROUTE_FIXED_BUDGET_ALLOCATION, 0)
    r["n_source_labels"], r["n_target_labels"], r["n_total_labels"] = 50.0, 0.0, 50.0
    assert artifacts._validate_label_access_semantics(rows) == []


def test_boolean_count_is_rejected():
    """bool is an int subclass, so a naive int()-coerce would accept True as 1 -- it must be rejected."""
    rows = _valid_rows()
    _find(rows, SA.ROUTE_TARGET_ONLY_FULL)["n_source_labels"] = True
    assert any("boolean" in p for p in artifacts._validate_label_access_semantics(rows))


def test_negative_count_is_rejected():
    rows = _valid_rows()
    _find(rows, SA.ROUTE_SOURCE_PLUS_TARGET, 25)["n_source_labels"] = -1
    assert any("is negative" in p for p in artifacts._validate_label_access_semantics(rows))


def test_missing_count_is_caught():
    rows = _valid_rows()
    del _find(rows, SA.ROUTE_SOURCE_PLUS_TARGET_FULL)["n_source_labels"]
    assert any("n_source_labels is missing" in p for p in artifacts._validate_label_access_semantics(rows))


def test_missing_allocation_budget_is_caught():
    rows = _valid_rows()
    del _find(rows, SA.ROUTE_FIXED_BUDGET_ALLOCATION, 75)["allocation_total_budget"]
    assert any("allocation_total_budget is missing" in p
               for p in artifacts._validate_label_access_semantics(rows))


# --------------------------------------------------------------------------- #
# write_run_complete: a valid suite publishes; a tampered count / unit does NOT
# --------------------------------------------------------------------------- #
def test_valid_suite_publishes_the_completion_marker(tmp_path):
    rows = _valid_rows()
    marker = _publish(tmp_path, rows, _keys(rows))
    assert marker and (tmp_path / artifacts.RUN_COMPLETE_FILE).exists()


def test_tampered_count_prevents_publication(tmp_path):
    rows = _valid_rows()
    # break the fixed-budget arithmetic WITHOUT touching any cell-key field, so completeness still
    # passes and ONLY the semantic validator can catch it.
    r = _find(rows, SA.ROUTE_FIXED_BUDGET_ALLOCATION, 25)
    r["n_target_labels"], r["n_source_labels"] = 12, B - 12
    with pytest.raises(artifacts.IncompleteRunError, match="inconsistent supervision accounting"):
        _publish(tmp_path, rows, _keys(_valid_rows()))
    assert artifacts.read_run_complete(tmp_path) is None


def test_tampered_unit_prevents_publication(tmp_path):
    rows = _valid_rows()
    _find(rows, SA.ROUTE_FIXED_BUDGET_ALLOCATION, 0)["label_budget_unit"] = "target_patches"
    with pytest.raises(artifacts.IncompleteRunError, match="inconsistent supervision accounting"):
        _publish(tmp_path, rows, _keys(_valid_rows()))
    assert artifacts.read_run_complete(tmp_path) is None


# --------------------------------------------------------------------------- #
# failure summary: the run-completion message names the route + evaluation split
# --------------------------------------------------------------------------- #
def test_run_completion_failure_summary_names_route_and_split(tmp_path):
    rows = _valid_rows()
    fails = [
        {"method": "erm", "holdout": "kenya", "label_budget": 25, "evaluation_split": SA.EVAL_TARGET_TEST,
         "label_access_route": SA.ROUTE_FIXED_BUDGET_ALLOCATION, "reason": "ValueError: boom"},
        {"method": "erm", "holdout": "kenya", "label_budget": 25, "evaluation_split": SA.EVAL_TARGET_TEST,
         "label_access_route": SA.ROUTE_SOURCE_PLUS_TARGET, "reason": "ValueError: boom"},
    ]
    with pytest.raises(artifacts.IncompleteRunError) as ei:
        _publish(tmp_path, rows, _keys(rows), cell_failures=fails)
    msg = str(ei.value)
    # the two routes collide numerically at 25 (a PERCENT vs a COUNT); the route name disambiguates them
    assert f"[{SA.ROUTE_FIXED_BUDGET_ALLOCATION}/{SA.EVAL_TARGET_TEST}]" in msg
    assert f"[{SA.ROUTE_SOURCE_PLUS_TARGET}/{SA.EVAL_TARGET_TEST}]" in msg


# --------------------------------------------------------------------------- #
# manifest: the readable label-access contract, enabled only when geographic_ood is requested
# --------------------------------------------------------------------------- #
def _manifest(regimes):
    return runstate.build_run_manifest(
        "raw", "cropharvest", "artifact", "digest", regimes, [0], {},
        active_probes=["logistic"], budget_regimes={"source": [1.0]}, max_dense_pixels=None,
    )


def _assert_canonical_contract(la):
    assert la["allocation_percents"] == list(SA.ALLOCATION_PERCENTS)
    assert la["additive_counts"] == list(SA.LABEL_ACCESS_COUNTS)
    assert la["full_target_reference"] is True and la["full_combined_reference"] is True
    assert la["controlled_budget_cap"] is None
    assert la["routes"] == list(SA.LABEL_ACCESS_ROUTES)
    assert la["evaluation_splits"] == [SA.EVAL_TARGET_TEST]
    assert la["unit"] == SA.LABEL_ACCESS_TABULAR_UNIT
    assert "counts" not in la          # the retired single-axis key is gone


def test_manifest_records_enabled_contract_when_geographic_ood_requested():
    la = _manifest(["geographic_ood", "random_id"])["label_access"]
    assert la["enabled"] is True
    _assert_canonical_contract(la)


def test_manifest_contract_disabled_without_geographic_ood():
    la = _manifest(["random_id", "official"])["label_access"]
    assert la["enabled"] is False
    # the canonical contract is still recorded (a self-describing manifest), just not active.
    _assert_canonical_contract(la)


def test_manifest_contract_records_the_configured_axes_not_just_the_defaults():
    """Two machines running different fractions / counts / caps must be distinguishable from the
    manifest alone, so the contract echoes the caller's config rather than the constants."""
    la = SA.label_access_contract(
        enabled=True, benchmark="cropharvest", percents=(0, 50, 100), counts=(5,),
        full_target_reference=False, full_combined_reference=False, controlled_budget_cap=32,
    )
    assert la["allocation_percents"] == [0, 50, 100]
    assert la["additive_counts"] == [5]
    assert la["full_target_reference"] is False and la["full_combined_reference"] is False
    assert la["controlled_budget_cap"] == 32


def test_manifest_contract_unit_is_patches_for_dense_pastis():
    assert SA.label_access_contract(enabled=True, benchmark="pastis")["unit"] == SA.LABEL_ACCESS_DENSE_UNIT
