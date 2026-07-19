"""Stage-3 artifact-integrity: label-access completion validation, failure-summary rendering, and the
run-manifest contract.

Completeness (the 9-field cell key) certifies that every planned (route, budget, split) row is PRESENT;
it says nothing about whether the supervision COUNTS on those rows are internally consistent. These tests
pin the SEMANTIC layer that runs alongside it -- a tampered count or unit keeps the key intact, so it must
be a separate guard that still blocks ``run_complete.json``. They also lock the readable manifest contract
and the route/split-aware failure summary.
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


def _counts(route, budget, b_src=60, p=55):
    """The correct supervision (n_source, n_target, n_total) for one route -- the accounting the
    semantic validator enforces."""
    m = min(b_src, p)
    return {
        SA.ROUTE_SOURCE_ONLY: (b_src, 0, b_src),
        SA.ROUTE_SOURCE_PLUS_TARGET: (b_src, budget, b_src + budget),
        SA.ROUTE_TARGET_ONLY_FULL: (0, p, p),
        SA.ROUTE_SOURCE_PLUS_TARGET_FULL: (b_src, p, b_src + p),
        SA.ROUTE_MATCHED_SOURCE: (m, 0, m),
        SA.ROUTE_MATCHED_TARGET: (0, m, m),
        SA.ROUTE_FIXED_TOTAL_MIXED: (b_src - budget, budget, b_src),
    }[route]


def _valid_rows(b_src=60, p=55, t=12):
    """The 14 rows one fully-eligible cell emits, each with correct supervision accounting -- derived from
    label_access_expected_rows() so the cell keys match the completeness plan exactly. n_test is the
    frozen target_test size ``t`` on every route, and ``t + p`` on the complete-target diagnostic, so the
    realized pool P = complete_target.n_test - target_test.n_test = p."""
    rows = []
    for route, budget, es in SA.label_access_expected_rows():
        ns, nt, ntot = _counts(route, budget, b_src, p)
        n_test = t + p if es == SA.EVAL_COMPLETE_TARGET else t
        rows.append({**BASE, "label_access_route": route, "label_budget": budget, "evaluation_split": es,
                     "n_source_labels": ns, "n_target_labels": nt, "n_total_labels": ntot, "n_test": n_test,
                     "label_budget_unit": SA.LABEL_ACCESS_TABULAR_UNIT, "f1": 1.0})
    return rows


def _find(rows, route, budget=0, es=SA.EVAL_TARGET_TEST):
    return next(r for r in rows if r["label_access_route"] == route
               and r["label_budget"] == budget and r["evaluation_split"] == es)


def _keys(rows):
    return {artifacts.cell_key(r) for r in rows}


#: A random_id full-source in-distribution row -- the Stage-5 source_ID_reference anchor that every real
#: label-access run also produces. Included so write_run_complete's contrast validation can resolve.
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
    for name in ("probe_results.csv", "summary.csv", "deltas.csv", artifacts.ENVIRONMENT_FILE):
        (tmp_path / name).write_text("{}\n")
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
    _find(rows, SA.ROUTE_SOURCE_ONLY)["n_total_labels"] = 61  # 61 != 60 + 0
    assert any("n_total" in p for p in artifacts._validate_label_access_semantics(rows))


def test_tampered_unit_is_caught():
    rows = _valid_rows()
    _find(rows, SA.ROUTE_MATCHED_TARGET)["label_budget_unit"] = "target_patches"
    assert any("unit" in p for p in artifacts._validate_label_access_semantics(rows))


def test_tampered_additive_count_is_caught():
    rows = _valid_rows()
    r = _find(rows, SA.ROUTE_SOURCE_PLUS_TARGET, 25)
    r["n_target_labels"], r["n_total_labels"] = 24, 84  # still balances, but no longer +25 over the base
    assert any("source_plus_target@25" in p for p in artifacts._validate_label_access_semantics(rows))


def test_tampered_fixed_total_invariance_is_caught():
    rows = _valid_rows()
    r = _find(rows, SA.ROUTE_FIXED_TOTAL_MIXED, 25)
    r["n_source_labels"], r["n_total_labels"] = 40, 65  # balances, but total no longer held at B_src=60
    assert any("fixed_total_mixed@25" in p for p in artifacts._validate_label_access_semantics(rows))


def test_tampered_matched_size_equality_is_caught():
    rows = _valid_rows()
    r = _find(rows, SA.ROUTE_MATCHED_TARGET)
    r["n_target_labels"], r["n_total_labels"] = 50, 50  # matched_target 50 != matched_source 55
    assert any("matched sizes differ" in p for p in artifacts._validate_label_access_semantics(rows))


def test_tampered_target_only_full_source_leak_is_caught():
    rows = _valid_rows()
    r = _find(rows, SA.ROUTE_TARGET_ONLY_FULL)
    r["n_source_labels"], r["n_total_labels"] = 3, 58  # target_only_full must train on 0 source
    assert any("target_only_full" in p for p in artifacts._validate_label_access_semantics(rows))


def test_tampered_diagnostic_disagreement_is_caught():
    rows = _valid_rows()
    r = _find(rows, SA.ROUTE_SOURCE_ONLY, 0, SA.EVAL_COMPLETE_TARGET)
    r["n_source_labels"], r["n_total_labels"] = 59, 59  # diagnostic no longer matches the source_only fit
    assert any("diagnostic" in p for p in artifacts._validate_label_access_semantics(rows))


# --- absolute checks anchored on the realized pool size P (not just internal consistency) ----------
def test_both_matched_rows_changed_to_the_same_wrong_size_is_caught():
    """Equal-but-wrong matched sizes pass the OLD source==target equality; only the min(B, P) anchor
    catches them. P=55, B=60 -> min=55, so 40==40 must still fail."""
    rows = _valid_rows()  # b_src=60, p=55 -> min(B,P)=55
    for route in (SA.ROUTE_MATCHED_SOURCE, SA.ROUTE_MATCHED_TARGET):
        r = _find(rows, route)
        tgt = "n_source_labels" if route == SA.ROUTE_MATCHED_SOURCE else "n_target_labels"
        r[tgt], r["n_total_labels"] = 40, 40  # both = 40 (equal to each other, balanced), but != min(B,P)=55
    probs = artifacts._validate_label_access_semantics(rows)
    assert not any("matched sizes differ" in p for p in probs)   # equality alone would NOT catch it
    assert any("min(B" in p for p in probs)                       # the min(B, P) anchor does


def test_both_full_target_rows_changed_to_the_same_wrong_pool_is_caught():
    """target_only_full and source_plus_target_full agreeing with each other but NOT with the realized
    pool P must be caught by the P anchor (the old tof==sptf equality would have passed)."""
    rows = _valid_rows()  # P = 55
    tof = _find(rows, SA.ROUTE_TARGET_ONLY_FULL)
    tof["n_target_labels"], tof["n_total_labels"] = 50, 50            # source 0 + 50
    sptf = _find(rows, SA.ROUTE_SOURCE_PLUS_TARGET_FULL)
    sptf["n_target_labels"], sptf["n_total_labels"] = 50, 60 + 50     # source 60 + 50
    probs = artifacts._validate_label_access_semantics(rows)
    assert sum("realized target pool P" in p for p in probs) == 2     # BOTH full-target rows flagged


def test_non_integral_float_count_is_rejected():
    rows = _valid_rows()
    _find(rows, SA.ROUTE_SOURCE_ONLY)["n_source_labels"] = 60.5
    probs = artifacts._validate_label_access_semantics(rows)
    assert any("n_source_labels=60.5" in p and "not an integer" in p for p in probs)


def test_boolean_count_is_rejected():
    """bool is an int subclass, so a naive int()-coerce would accept True as 1 -- it must be rejected."""
    rows = _valid_rows()
    _find(rows, SA.ROUTE_MATCHED_SOURCE)["n_source_labels"] = True
    assert any("boolean" in p for p in artifacts._validate_label_access_semantics(rows))


def test_negative_count_is_rejected():
    rows = _valid_rows()
    _find(rows, SA.ROUTE_SOURCE_PLUS_TARGET, 25)["n_source_labels"] = -1
    assert any("is negative" in p for p in artifacts._validate_label_access_semantics(rows))


def test_diagnostic_n_test_smaller_than_target_test_is_caught():
    """complete_target must cover target_test + the pool, so its n_test >= target_test.n_test. A smaller
    value yields a negative realized pool P and must be rejected (never a negative min(B, P))."""
    rows = _valid_rows(t=12)
    _find(rows, SA.ROUTE_SOURCE_ONLY, 0, SA.EVAL_COMPLETE_TARGET)["n_test"] = 5  # < target_test n_test=12
    assert any("realized target pool" in p and "negative" in p
               for p in artifacts._validate_label_access_semantics(rows))


def test_missing_n_test_on_source_only_rows_is_caught():
    rows = _valid_rows()
    del _find(rows, SA.ROUTE_SOURCE_ONLY)["n_test"]
    assert any("n_test is missing" in p for p in artifacts._validate_label_access_semantics(rows))


# --------------------------------------------------------------------------- #
# write_run_complete: a valid suite publishes; a tampered count / unit does NOT
# --------------------------------------------------------------------------- #
def test_valid_suite_publishes_the_completion_marker(tmp_path):
    rows = _valid_rows()
    marker = _publish(tmp_path, rows, _keys(rows))
    assert marker and (tmp_path / artifacts.RUN_COMPLETE_FILE).exists()


def test_tampered_count_prevents_publication(tmp_path):
    rows = _valid_rows()
    # break fixed-total invariance WITHOUT touching any cell-key field, so completeness still passes and
    # ONLY the semantic validator can catch it.
    _find(rows, SA.ROUTE_FIXED_TOTAL_MIXED, 25)["n_total_labels"] = 61
    with pytest.raises(artifacts.IncompleteRunError, match="inconsistent supervision accounting"):
        _publish(tmp_path, rows, _keys(_valid_rows()))
    assert artifacts.read_run_complete(tmp_path) is None


def test_tampered_unit_prevents_publication(tmp_path):
    rows = _valid_rows()
    _find(rows, SA.ROUTE_SOURCE_ONLY)["label_budget_unit"] = "target_patches"
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
         "label_access_route": SA.ROUTE_SOURCE_PLUS_TARGET, "reason": "ValueError: boom"},
        {"method": "erm", "holdout": "kenya", "label_budget": 25, "evaluation_split": SA.EVAL_TARGET_TEST,
         "label_access_route": SA.ROUTE_FIXED_TOTAL_MIXED, "reason": "ValueError: boom"},
    ]
    with pytest.raises(artifacts.IncompleteRunError) as ei:
        _publish(tmp_path, rows, _keys(rows), cell_failures=fails)
    msg = str(ei.value)
    assert f"[{SA.ROUTE_SOURCE_PLUS_TARGET}/{SA.EVAL_TARGET_TEST}]" in msg
    assert f"[{SA.ROUTE_FIXED_TOTAL_MIXED}/{SA.EVAL_TARGET_TEST}]" in msg


# --------------------------------------------------------------------------- #
# manifest: the readable label-access contract, enabled only when geographic_ood is requested
# --------------------------------------------------------------------------- #
def _manifest(regimes):
    return runstate.build_run_manifest(
        "raw", "cropharvest", "artifact", "digest", regimes, [0], {},
        active_probes=["logistic"], budget_regimes={"source": [1.0]}, max_dense_pixels=None,
    )


def test_manifest_records_enabled_contract_when_geographic_ood_requested():
    la = _manifest(["geographic_ood", "random_id"])["label_access"]
    assert la["enabled"] is True
    assert la["counts"] == list(SA.LABEL_ACCESS_COUNTS)
    assert la["routes"] == list(SA.LABEL_ACCESS_ROUTES)
    assert la["evaluation_splits"] == [SA.EVAL_TARGET_TEST, SA.EVAL_COMPLETE_TARGET]
    assert la["unit"] == "samples"


def test_manifest_contract_disabled_without_geographic_ood():
    la = _manifest(["random_id", "official"])["label_access"]
    assert la["enabled"] is False
    # the canonical contract is still recorded (a self-describing manifest), just not active.
    assert la["counts"] == list(SA.LABEL_ACCESS_COUNTS)
    assert la["routes"] == list(SA.LABEL_ACCESS_ROUTES)
    assert la["unit"] == "samples"
