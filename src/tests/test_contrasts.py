"""Stage 5: paired label-access post-processing (pure post-processing on completed probe rows).

Covers the FOUR canonical contrast families and their exact per-fraction / per-k pairing, the
``geographic_full_source`` (E1) anchor resolution, the surviving ``source_ID_reference`` machinery,
equal-region weighting, the TARGET-REGION bootstrap (seeds averaged within a target, never resampled),
the headline-target floor, the duplicate/missing-operand hard failures, and the completion-marker
validation (missing / stale / hash). Tabular (samples) AND PASTIS (patches)."""

from __future__ import annotations

import numpy as np
import pytest

from evals import contrasts
from evals import split_artifacts as SA
from utils import artifacts
from utils import ioutils as IOU

R = SA  # route/anchor/contrast constants live on split_artifacts

#: Fixture sizing. ``BUDGET`` is the benchmark-common fixed budget B_d the allocation curve pins;
#: ``POOL`` is the COMPLETE source pool the additive routes and E1 train on (deliberately > BUDGET, so
#: an allocation row can never be confused with a complete-source row); ``P`` / ``T`` are the target
#: label pool and the frozen target_test.
BUDGET, POOL, P, T = 60, 80, 55, 12

#: Distinct per-(route, budget) metric values so every contrast has a uniquely checkable difference.
ALLOC_VAL: dict[int, float] = {0: 0.50, 25: 0.52, 50: 0.54, 75: 0.56, 100: 0.58}
SPT_VAL: dict[int, float] = {5: 0.55, 10: 0.60, 25: 0.65, 50: 0.70}
TOF_VAL = 0.80        # target_only_full
SPTF_VAL = 0.85       # source_plus_target_full
E1_VAL = 0.50         # the ordinary full-source geographic row (E1), scored on target_test
E1_COMPLETE_VAL = 0.99  # the SAME fit scored on the complete region -- never a contrast operand
ANCHOR_VALUE = 0.90   # random_id full-source (source_ID_reference)

#: The expected difference of each contrast family under the fixture above.
EXPECT_TARGET_LABEL_ADVANTAGE = TOF_VAL - E1_VAL                       # 0.30
EXPECT_FULL_SUPERVISION_GAIN = SPTF_VAL - E1_VAL                       # 0.35
EXPECT_ADDITIVE = {k: SPT_VAL[k] - E1_VAL for k in SPT_VAL}            # 0.05 .. 0.20
EXPECT_ALLOCATION = {f: ALLOC_VAL[f] - ALLOC_VAL[0] for f in (25, 50, 75, 100)}  # 0.02 .. 0.08


def _route_value(route: str, budget: int) -> float:
    if route == R.ROUTE_FIXED_BUDGET_ALLOCATION:
        return ALLOC_VAL[int(budget)]
    if route == R.ROUTE_SOURCE_PLUS_TARGET:
        return SPT_VAL[int(budget)]
    return TOF_VAL if route == R.ROUTE_TARGET_ONLY_FULL else SPTF_VAL


def _counts(route: str, budget: int) -> tuple[int, int, int]:
    """(n_source, n_target, n_total) label units for one route, honouring the accounting contract:
    the allocation curve pins the TOTAL at ``BUDGET``; the additive routes hold the COMPLETE pool."""
    if route == R.ROUTE_FIXED_BUDGET_ALLOCATION:
        k = R.allocation_target_count(int(budget), BUDGET)
        return BUDGET - k, k, BUDGET
    if route == R.ROUTE_SOURCE_PLUS_TARGET:
        return POOL, int(budget), POOL + int(budget)
    if route == R.ROUTE_TARGET_ONLY_FULL:
        return 0, P, P
    return POOL, P, POOL + P


def _la(route, budget, seed, target, value, *, benchmark="cropharvest", unit="samples", metric="f1"):
    """One label-access probe row, scored on the frozen target_test."""
    ns, nt, ntot = _counts(route, int(budget))
    row = {
        "benchmark": benchmark, "model": "raw", "probe_family": "logistic", "seed": seed,
        "holdout": target, "split_regime": R.LABEL_ACCESS_REGIME, "budget_type": "label_access",
        "evaluation_split": R.EVAL_TARGET_TEST, "label_access_route": route, "label_budget": int(budget),
        "n_source_labels": ns, "n_target_labels": nt, "n_total_labels": ntot,
        "n_source_pool": POOL, "allocation_total_budget": BUDGET,
        "label_budget_unit": unit, metric: value,
    }
    if unit == "patches":
        row["n_eval_patches"] = T        # patch count the validator uses to derive P
        row["n_test"] = T * 10           # evaluated PIXELS (a different unit; not used for P)
    else:
        row["n_test"] = T                # samples == the label unit
    return row


def _e1(seed, target, value, *, es="test", benchmark="cropharvest", metric="f1"):
    """The ordinary full-source geographic row -- the ONE complete-source leg of the whole suite. It is
    fit once and scored twice: on the frozen ``target_test`` (the contrast operand, ``es="test"``) and on
    the ``complete_target`` region (the deployment estimand, never a contrast operand)."""
    return {
        "benchmark": benchmark, "model": "raw", "probe_family": "logistic", "seed": seed,
        "holdout": target, "split_regime": R.LABEL_ACCESS_REGIME, "budget_type": "source",
        "evaluation_split": es, "label_budget": 1.0, "label_access_route": "",
        "n_test": (T + P) if es == R.EVAL_COMPLETE_TARGET else T, metric: value,
    }


def _anchor(seed, value, *, benchmark="cropharvest", metric="f1"):
    """The random_id full-source in-distribution reference (``source_ID_reference``)."""
    return {
        "benchmark": benchmark, "model": "raw", "probe_family": "logistic", "seed": seed,
        "holdout": "random_id", "split_regime": "random_id", "budget_type": "source",
        "evaluation_split": "test", "label_budget": 1.0, "label_access_route": "", metric: value,
    }


def _cell(seed, target, *, benchmark="cropharvest", unit="samples", metric="f1",
          overrides=None, e1=E1_VAL):
    """The 11 label-access rows one fully-eligible cell emits (5 allocation fractions, 4 additive
    counts, the two full references) PLUS the cell's two E1 scopes. There is deliberately no
    ``source_only`` route and no ``complete_target`` label-access row."""
    overrides = overrides or {}
    rows = [
        _la(route, budget, seed, target,
            overrides.get((route, int(budget)), _route_value(route, int(budget))),
            benchmark=benchmark, unit=unit, metric=metric)
        for route, budget, _es in R.label_access_expected_rows()
    ]
    rows.append(_e1(seed, target, e1, benchmark=benchmark, metric=metric))
    rows.append(_e1(seed, target, E1_COMPLETE_VAL, es=R.EVAL_COMPLETE_TARGET,
                    benchmark=benchmark, metric=metric))
    return rows


def _run_rows(seeds=(0,), targets=("kenya",), *, benchmark="cropharvest", unit="samples", metric="f1"):
    rows = []
    for s in seeds:
        for t in targets:
            rows.extend(_cell(s, t, benchmark=benchmark, unit=unit, metric=metric))
        rows.append(_anchor(s, ANCHOR_VALUE, benchmark=benchmark, metric=metric))
    return rows


def _paired_by(paired, contrast, budget=0, target="kenya", seed=0):
    return next(r for r in paired if r["contrast"] == contrast and r["budget"] == budget
                and r["target"] == target and r["seed"] == seed)


def _summary_by(summary, contrast, budget=0):
    return next(s for s in summary if s["contrast"] == contrast and s["budget"] == budget)


# --------------------------------------------------------------------------- #
# the FOUR contrast families + exact per-fraction / per-k pairing (tabular)
# --------------------------------------------------------------------------- #
def test_all_four_contrast_families_and_exact_pairing_tabular():
    paired, _summary = contrasts.compute_contrasts(_run_rows())
    assert _paired_by(paired, "target_label_advantage")["difference"] == pytest.approx(
        EXPECT_TARGET_LABEL_ADVANTAGE)                                          # target_only_full - E1
    assert _paired_by(paired, "full_supervision_gain")["difference"] == pytest.approx(
        EXPECT_FULL_SUPERVISION_GAIN)                                           # source_plus_target_full - E1
    # additive(k) = source_plus_target(k) - E1 ; each k paired EXACTLY.
    for k in SA.LABEL_ACCESS_COUNTS:
        assert _paired_by(paired, "additive_target_label_gain", budget=k)["difference"] == pytest.approx(
            EXPECT_ADDITIVE[k])
    # allocation(f) = fixed_budget_allocation(f) - fixed_budget_allocation(0) ; each f paired EXACTLY.
    for f in (25, 50, 75, 100):
        assert _paired_by(paired, "allocation_effect", budget=f)["difference"] == pytest.approx(
            EXPECT_ALLOCATION[f])
    assert {c for c, _m, _s in SA.LABEL_ACCESS_CONTRASTS} == {r["contrast"] for r in paired}


def test_retired_contrasts_and_routes_are_absent():
    """The retired size-matched / fixed-total-mixed vocabulary must not reappear anywhere in Stage 5."""
    paired, summary = contrasts.compute_contrasts(_run_rows())
    names = {r["contrast"] for r in paired} | {s["contrast"] for s in summary}
    assert not names & {"size_matched_source_target_difference", "label_source_allocation_effect",
                        "target_reference_deficit"}
    routes = {r["minuend_route"] for r in paired} | {r["subtrahend_route"] for r in paired}
    assert not routes & {"source_only", "matched_source", "matched_target", "fixed_total_mixed"}
    for retired in ("ROUTE_SOURCE_ONLY", "ROUTE_MATCHED_SOURCE", "ROUTE_MATCHED_TARGET",
                    "ROUTE_FIXED_TOTAL_MIXED"):
        assert not hasattr(SA, retired), retired


def test_allocation_effect_pairs_against_its_own_zero_fraction_not_e1():
    """The allocation EFFECT holds TOTAL supervision fixed at B_d and moves only its composition, so
    every non-zero fraction is subtracted from the SAME cell's f=0 endpoint -- never from E1, which
    trains on the complete source pool and would confound budget SIZE with composition."""
    paired, _ = contrasts.compute_contrasts(_run_rows())
    for f in (25, 50, 75, 100):
        row = _paired_by(paired, "allocation_effect", budget=f)
        assert row["subtrahend_route"] == SA.ALLOCATION_BASELINE
        assert row["subtrahend_value"] == pytest.approx(ALLOC_VAL[0])   # the curve's own f=0 endpoint
        assert row["minuend_value"] == pytest.approx(ALLOC_VAL[f])
    # f=0 is an ENDPOINT of the curve, not a contrast of its own.
    assert {r["budget"] for r in paired if r["contrast"] == "allocation_effect"} == {25, 50, 75, 100}


def test_allocation_effect_is_independent_of_e1_while_the_others_track_it():
    """Moving E1 shifts the three E1-anchored families by exactly that amount and leaves the allocation
    curve untouched -- the structural proof that allocation is not anchored on E1."""
    base, _ = contrasts.compute_contrasts(_run_rows())
    shifted_rows = [
        {**r, "f1": r["f1"] - 0.10} if (r.get("budget_type") == "source"
                                        and r.get("split_regime") == R.LABEL_ACCESS_REGIME
                                        and r.get("evaluation_split") == "test") else r
        for r in _run_rows()
    ]
    shifted, _ = contrasts.compute_contrasts(shifted_rows)
    for f in (25, 50, 75, 100):
        assert (_paired_by(shifted, "allocation_effect", budget=f)["difference"]
                == pytest.approx(_paired_by(base, "allocation_effect", budget=f)["difference"]))
    for name, budget in (("target_label_advantage", 0), ("full_supervision_gain", 0),
                         ("additive_target_label_gain", 25)):
        assert (_paired_by(shifted, name, budget=budget)["difference"]
                == pytest.approx(_paired_by(base, name, budget=budget)["difference"] + 0.10))


def test_complete_target_scope_of_e1_is_never_an_operand():
    """E1 is scored twice from ONE fit: on target_test (the operand) and on the complete region (the
    deployment estimand, 0.99 here). Only the target_test scope may enter a contrast."""
    paired, _ = contrasts.compute_contrasts(_run_rows())
    row = _paired_by(paired, "target_label_advantage")
    assert row["subtrahend_value"] == pytest.approx(E1_VAL)      # E1@target_test, not 0.99
    assert row["minuend_value"] == pytest.approx(TOF_VAL)
    assert all(r["evaluation_split"] == R.EVAL_TARGET_TEST for r in paired)


def test_label_access_emits_no_complete_target_row():
    """The label-access suite itself has no complete_target row at all under the new contract."""
    assert R.EVAL_COMPLETE_TARGET not in {es for _r, _b, es in R.label_access_expected_rows()}
    assert R.LABEL_ACCESS_EVAL_SPLITS == (R.EVAL_TARGET_TEST,)
    assert all(r["evaluation_split"] == R.EVAL_TARGET_TEST
               for r in _run_rows() if r["budget_type"] == "label_access")


def test_provenance_columns_present_and_ordered():
    paired, summary = contrasts.compute_contrasts(_run_rows())
    r = _paired_by(paired, "additive_target_label_gain", budget=25)
    for col in ("minuend_route", "subtrahend_route", "minuend_value", "subtrahend_value",
                "subtraction_order", "metric", "metric_direction", "budget", "seed", "target",
                "benchmark", "model", "probe_family", "difference"):
        assert col in r
    assert r["minuend_route"] == R.ROUTE_SOURCE_PLUS_TARGET
    assert r["subtrahend_route"] == R.ANCHOR_GEOGRAPHIC_FULL_SOURCE
    assert r["subtraction_order"] == f"{R.ROUTE_SOURCE_PLUS_TARGET} - {R.ANCHOR_GEOGRAPHIC_FULL_SOURCE}"
    assert r["metric_direction"] == "higher_is_better"
    alloc = _paired_by(paired, "allocation_effect", budget=50)
    assert alloc["minuend_route"] == R.ROUTE_FIXED_BUDGET_ALLOCATION
    assert alloc["subtraction_order"] == f"{R.ROUTE_FIXED_BUDGET_ALLOCATION} - {R.ALLOCATION_BASELINE}"
    assert all("region_weighting" in s and s["region_weighting"] == "equal" for s in summary)


def test_summary_schema_is_exactly_the_new_contract():
    _paired, summary = contrasts.compute_contrasts(_run_rows(seeds=(0, 1), targets=("a", "b", "c")))
    assert list(summary[0]) == [
        "contrast", "benchmark", "model", "probe_family", "metric", "metric_direction", "budget",
        "minuend_route", "subtrahend_route", "subtraction_order", "region_weighting",
        "mean_difference", "std_across_seeds", "std_across_targets", "n_seeds", "n_targets",
        "headline", "ci_convention", "ci_lo", "ci_hi",
    ]
    assert all(list(s) == list(summary[0]) for s in summary)
    # the retired aggregation keys are gone
    assert not {"std_difference", "n_targets_per_seed"} & set(summary[0])
    assert {s["ci_convention"] for s in summary} == {"target_region_bootstrap_2.5_97.5"}


def test_metric_direction_error_metrics_are_lower_is_better():
    assert contrasts.metric_direction("f1") == "higher_is_better"
    assert contrasts.metric_direction("miou") == "higher_is_better"
    for m in ("ece", "nll", "brier", "shared_nll", "union_brier", "top_label_ece_all"):
        assert contrasts.metric_direction(m) == "lower_is_better"


# --------------------------------------------------------------------------- #
# PASTIS (patches) -- identical contrast logic, patch unit
# --------------------------------------------------------------------------- #
def test_all_four_families_pastis_patches():
    rows = _run_rows(benchmark="pastis", unit="patches", metric="miou")
    paired, summary = contrasts.compute_contrasts(rows)
    assert _paired_by(paired, "target_label_advantage")["difference"] == pytest.approx(
        EXPECT_TARGET_LABEL_ADVANTAGE)
    assert _paired_by(paired, "full_supervision_gain")["difference"] == pytest.approx(
        EXPECT_FULL_SUPERVISION_GAIN)
    for k in SA.LABEL_ACCESS_COUNTS:
        assert _paired_by(paired, "additive_target_label_gain", budget=k)["difference"] == pytest.approx(
            EXPECT_ADDITIVE[k])
    for f in (25, 50, 75, 100):
        assert _paired_by(paired, "allocation_effect", budget=f)["difference"] == pytest.approx(
            EXPECT_ALLOCATION[f])
    assert {r["metric"] for r in paired} == {"miou"}
    assert {s["benchmark"] for s in summary} == {"pastis"}
    assert SA.label_access_unit("pastis") == "patches"


# --------------------------------------------------------------------------- #
# anchor resolution: E1 (the subtrahend) and the surviving source_ID_reference machinery
# --------------------------------------------------------------------------- #
def test_missing_e1_row_hard_fails():
    """Label access no longer refits a source-only probe, so the ordinary full-source geographic row is
    REQUIRED -- its absence is a hard failure, never a silently dropped contrast."""
    rows = [
        r for r in _run_rows()
        if not (r.get("budget_type") == "source" and r.get("split_regime") == R.LABEL_ACCESS_REGIME
                and r.get("evaluation_split") == "test")
    ]
    with pytest.raises(contrasts.ContrastError, match="full-source geographic"):
        contrasts.compute_contrasts(rows)


def test_duplicate_e1_row_hard_fails():
    """Two E1 rows would make the choice of complete-source leg arbitrary."""
    rows = _run_rows() + [_e1(0, "kenya", 0.51)]
    with pytest.raises(contrasts.ContrastError, match="duplicate full-source geographic"):
        contrasts.compute_contrasts(rows)


def test_e1_is_matched_on_the_full_cell_not_just_the_benchmark():
    """E1 is resolved per (benchmark, model, probe_family, seed, holdout): an E1 row for a DIFFERENT
    target cannot stand in for the missing one."""
    rows = [
        r for r in _run_rows(targets=("kenya", "togo"))
        if not (r.get("budget_type") == "source" and r.get("holdout") == "togo"
                and r.get("split_regime") == R.LABEL_ACCESS_REGIME
                and r.get("evaluation_split") == "test")
    ]
    with pytest.raises(contrasts.ContrastError, match="full-source geographic"):
        contrasts.compute_contrasts(rows)


def test_source_id_reference_anchor_machinery_still_resolves():
    """No contrast family consumes ``source_ID_reference`` any more, but the anchor resolution it
    depends on is retained (and still exercised on every compute) -- it must pick the random_id
    FULL-source row and nothing else."""
    rows = _run_rows()
    rows.append({**_anchor(0, 0.10), "label_budget": 0.1})   # a PARTIAL-source random_id row
    resolved = contrasts._resolve_anchors(rows)
    assert set(resolved) == {("cropharvest", "raw", "logistic", 0)}
    assert resolved[("cropharvest", "raw", "logistic", 0)]["f1"] == pytest.approx(ANCHOR_VALUE)
    contrasts.compute_contrasts(rows)   # the partial row does not disturb the suite


def test_duplicate_source_id_reference_anchor_hard_fails():
    rows = _run_rows() + [_anchor(0, 0.91)]   # a second random_id full-source row for the same cell
    with pytest.raises(contrasts.ContrastError, match="not unique"):
        contrasts.compute_contrasts(rows)


# --------------------------------------------------------------------------- #
# missing / duplicate operand hard failures (never silently dropped)
# --------------------------------------------------------------------------- #
def test_missing_operand_hard_fails():
    rows = [r for r in _run_rows() if r.get("label_access_route") != R.ROUTE_TARGET_ONLY_FULL]
    with pytest.raises(contrasts.ContrastError, match="missing operand"):
        contrasts.compute_contrasts(rows)


def test_missing_allocation_fraction_hard_fails():
    """A hole in the allocation curve is a hard failure, not a shorter curve."""
    rows = [
        r for r in _run_rows()
        if not (r.get("label_access_route") == R.ROUTE_FIXED_BUDGET_ALLOCATION
                and r.get("label_budget") == 75)
    ]
    with pytest.raises(contrasts.ContrastError, match="missing operand"):
        contrasts.compute_contrasts(rows)


def test_missing_allocation_baseline_hard_fails():
    """Losing the f=0 endpoint removes the subtrahend of the whole allocation family."""
    rows = [
        r for r in _run_rows()
        if not (r.get("label_access_route") == R.ROUTE_FIXED_BUDGET_ALLOCATION
                and r.get("label_budget") == 0)
    ]
    with pytest.raises(contrasts.ContrastError, match="missing operand"):
        contrasts.compute_contrasts(rows)


def test_duplicate_operand_hard_fails():
    rows = _run_rows()
    rows.append(_la(R.ROUTE_TARGET_ONLY_FULL, 0, 0, "kenya", 0.81))   # a second target_only_full
    with pytest.raises(contrasts.ContrastError, match="duplicate label-access operand"):
        contrasts.compute_contrasts(rows)


def test_allocation_percent_and_additive_count_do_not_collide():
    """allocation@25 (a PERCENT) and source_plus_target@25 (a COUNT) share a numeric budget; the route
    keeps them distinct, so neither is a duplicate operand nor pairs against the other."""
    paired, _ = contrasts.compute_contrasts(_run_rows())
    alloc = _paired_by(paired, "allocation_effect", budget=25)
    additive = _paired_by(paired, "additive_target_label_gain", budget=25)
    assert alloc["minuend_value"] == pytest.approx(ALLOC_VAL[25])
    assert additive["minuend_value"] == pytest.approx(SPT_VAL[25])
    assert alloc["difference"] != pytest.approx(additive["difference"])


# --------------------------------------------------------------------------- #
# aggregation: equal region weighting, then the TARGET-REGION bootstrap
# --------------------------------------------------------------------------- #
def _advantage_cell(seed, target, advantage, **kw):
    return _cell(seed, target, overrides={(R.ROUTE_TARGET_ONLY_FULL, 0): E1_VAL + advantage}, **kw)


def test_equal_region_weighting_never_sample_weighted():
    # three targets in one seed with very different target_test SIZES; the aggregate is the SIMPLE mean
    # of their contrasts (0.30, 0.20, 0.40 -> 0.30), never the size-weighted value.
    rows = _advantage_cell(0, "kenya", 0.30)
    for adv, target, scale in ((0.20, "togo", 25), (0.40, "mali", 1)):
        cell = _advantage_cell(0, target, adv)
        for r in cell:
            r["n_test"] = r["n_test"] * scale
        rows += cell
    rows.append(_anchor(0, ANCHOR_VALUE))
    _paired, summary = contrasts.compute_contrasts(rows)
    s = _summary_by(summary, "target_label_advantage")
    assert s["mean_difference"] == pytest.approx(0.30)   # equal weight, NOT size-weighted (~0.22)
    assert s["region_weighting"] == "equal" and s["n_targets"] == 3


def test_seed_and_target_variation_are_reported_separately():
    # 3 seeds x 3 targets. Seed means vary; target means vary; both spreads are reported on their own
    # key and NEITHER is folded into the other.
    adv = {(s, t): 0.30 + 0.10 * si + 0.05 * ti
           for si, s in enumerate((0, 1, 2)) for ti, t in enumerate(("a", "b", "c"))}
    rows = []
    for s in (0, 1, 2):
        for t in ("a", "b", "c"):
            rows.extend(_advantage_cell(s, t, adv[(s, t)]))
        rows.append(_anchor(s, ANCHOR_VALUE))
    _paired, summary = contrasts.compute_contrasts(rows)
    s = _summary_by(summary, "target_label_advantage")
    assert s["n_seeds"] == 3 and s["n_targets"] == 3
    per_seed = [float(np.mean([adv[(sd, t)] for t in ("a", "b", "c")])) for sd in (0, 1, 2)]
    per_target = [float(np.mean([adv[(sd, t)] for sd in (0, 1, 2)])) for t in ("a", "b", "c")]
    assert s["mean_difference"] == pytest.approx(float(np.mean(per_target)))
    assert s["std_across_seeds"] == pytest.approx(float(np.std(per_seed)))
    assert s["std_across_targets"] == pytest.approx(float(np.std(per_target)))
    assert s["headline"] is True
    assert s["ci_lo"] <= s["mean_difference"] <= s["ci_hi"]


@pytest.mark.parametrize("targets", [("kenya",), ("kenya", "togo")])
def test_fewer_than_three_targets_gets_no_headline_interval(targets):
    """A region-level bootstrap over one or two regions has no usable uncertainty: the CI is NaN and the
    aggregate is explicitly NOT headline -- never a narrow interval no one should read."""
    assert SA.MIN_HEADLINE_TARGETS == 3
    _paired, summary = contrasts.compute_contrasts(_run_rows(seeds=(0, 1, 2), targets=targets))
    for s in summary:
        assert s["n_targets"] == len(targets)
        assert s["headline"] is False
        assert np.isnan(s["ci_lo"]) and np.isnan(s["ci_hi"])
        assert not np.isnan(s["mean_difference"])   # the point estimate is still reported


def test_three_targets_reaches_the_headline_floor():
    _paired, summary = contrasts.compute_contrasts(_run_rows(seeds=(0,), targets=("a", "b", "c")))
    for s in summary:
        assert s["n_targets"] == 3 and s["headline"] is True
        assert not (np.isnan(s["ci_lo"]) or np.isnan(s["ci_hi"]))


# --------------------------------------------------------------------------- #
# the region bootstrap: seeds averaged WITHIN a target, regions resampled
# --------------------------------------------------------------------------- #
def _seed_only_ci(per_seed_means):
    rng = np.random.default_rng(0)
    arr = np.asarray(per_seed_means, dtype=float)
    boot = np.array([arr[rng.integers(0, arr.size, arr.size)].mean() for _ in range(2000)])
    return float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def test_region_bootstrap_widens_when_targets_disagree_even_if_seed_means_coincide():
    """3 seeds x 3 targets whose per-SEED equal-region means are all exactly 0.30 while the per-TARGET
    means are 0.15 / 0.30 / 0.45. A seed-first bootstrap collapses to a zero-width interval here; the
    region bootstrap -- the unit we actually generalize over -- does not."""
    adv = {(0, "a"): 0.20, (0, "b"): 0.30, (0, "c"): 0.40,
           (1, "a"): 0.10, (1, "b"): 0.35, (1, "c"): 0.45,
           (2, "a"): 0.15, (2, "b"): 0.25, (2, "c"): 0.50}
    rows = []
    for seed in (0, 1, 2):
        for target in ("a", "b", "c"):
            rows.extend(_advantage_cell(seed, target, adv[(seed, target)]))
        rows.append(_anchor(seed, ANCHOR_VALUE))
    _paired, summary = contrasts.compute_contrasts(rows)
    s = _summary_by(summary, "target_label_advantage")

    assert s["mean_difference"] == pytest.approx(0.30)
    assert s["std_across_seeds"] == pytest.approx(0.0)                  # the seed means coincide
    assert s["std_across_targets"] == pytest.approx(float(np.std([0.15, 0.30, 0.45])))
    # a seed-ONLY bootstrap over the coincident per-seed means is DEGENERATE ...
    lo0, hi0 = _seed_only_ci([0.30, 0.30, 0.30])
    assert lo0 == pytest.approx(0.30) and hi0 == pytest.approx(0.30)
    # ... but the region bootstrap carries the target disagreement into the interval.
    assert s["ci_hi"] > s["ci_lo"]
    assert s["ci_lo"] < 0.30 < s["ci_hi"]
    assert s["ci_convention"] == "target_region_bootstrap_2.5_97.5"


def test_seeds_are_averaged_within_a_target_and_never_resampled():
    """Replicating identical seeds does not narrow (or otherwise move) the region interval: three seeds
    of the same region are ONE observation of the population, not three."""
    adv = {"a": 0.15, "b": 0.30, "c": 0.45}

    def _rows(seeds):
        out = []
        for seed in seeds:
            for t, a in adv.items():
                out.extend(_advantage_cell(seed, t, a))
            out.append(_anchor(seed, ANCHOR_VALUE))
        return out

    one = _summary_by(contrasts.compute_contrasts(_rows((0,)))[1], "target_label_advantage")
    three = _summary_by(contrasts.compute_contrasts(_rows((0, 1, 2)))[1], "target_label_advantage")
    assert three["n_seeds"] == 3 and one["n_seeds"] == 1
    assert three["ci_lo"] == pytest.approx(one["ci_lo"])
    assert three["ci_hi"] == pytest.approx(one["ci_hi"])
    assert three["mean_difference"] == pytest.approx(one["mean_difference"])
    assert three["std_across_seeds"] == pytest.approx(0.0)


def test_region_bootstrap_is_deterministic():
    rows = _run_rows(seeds=(0, 1, 2), targets=("a", "b", "c"))
    a = contrasts.compute_contrasts(rows)[1]
    b = contrasts.compute_contrasts(rows)[1]
    assert contrasts._csv_bytes(a) == contrasts._csv_bytes(b)   # fixed-seed bootstrap -> identical bytes


def test_aggregation_is_deterministic():
    rows = _run_rows(seeds=(0, 1, 2))
    a = contrasts.compute_contrasts(rows)[1]
    b = contrasts.compute_contrasts(rows)[1]
    assert contrasts._csv_bytes(a) == contrasts._csv_bytes(b)


# --------------------------------------------------------------------------- #
# write + completion validation (missing / stale / hash)
# --------------------------------------------------------------------------- #
def test_write_and_validate_round_trip(tmp_path):
    rows = _run_rows(seeds=(0, 1, 2), targets=("kenya", "togo", "mali"))
    contrasts.compute_and_write(tmp_path, rows)
    assert (tmp_path / contrasts.CONTRAST_FILE).exists()
    problems, hashes = contrasts.validate_written_contrasts(tmp_path, rows)
    assert problems == []
    assert set(hashes) == {contrasts.CONTRAST_FILE, contrasts.CONTRAST_SUMMARY_FILE}
    assert all("sha256" in h and h["bytes"] > 0 for h in hashes.values())


def test_validate_detects_missing_artifact(tmp_path):
    rows = _run_rows()
    contrasts.compute_and_write(tmp_path, rows)
    (tmp_path / contrasts.CONTRAST_FILE).unlink()
    problems, _ = contrasts.validate_written_contrasts(tmp_path, rows)
    assert any("missing" in p for p in problems)


def test_validate_detects_stale_or_tampered_artifact(tmp_path):
    rows = _run_rows()
    contrasts.compute_and_write(tmp_path, rows)
    path = tmp_path / contrasts.CONTRAST_SUMMARY_FILE
    path.write_bytes(path.read_bytes() + b"tampered,row\n")   # inconsistent with a fresh recompute
    problems, _ = contrasts.validate_written_contrasts(tmp_path, rows)
    assert any("stale" in p or "inconsistent" in p for p in problems)


def test_validate_reports_an_uncomputable_contrast_set(tmp_path):
    """A run whose contrasts cannot be recomputed (here: E1 was removed after the fact) is reported as a
    problem, not raised through the completion path."""
    rows = _run_rows()
    contrasts.compute_and_write(tmp_path, rows)
    broken = [
        r for r in rows
        if not (r.get("budget_type") == "source" and r.get("split_regime") == R.LABEL_ACCESS_REGIME
                and r.get("evaluation_split") == "test")
    ]
    problems, hashes = contrasts.validate_written_contrasts(tmp_path, broken)
    assert hashes == {} and any("could not be recomputed" in p for p in problems)


def test_non_label_access_run_is_a_no_op(tmp_path):
    rows = [{"split_regime": "random_id", "budget_type": "source", "evaluation_split": "test", "f1": 0.5}]
    contrasts.compute_and_write(tmp_path, rows)
    assert not (tmp_path / contrasts.CONTRAST_FILE).exists()
    assert contrasts.validate_written_contrasts(tmp_path, rows) == ([], {})


# --------------------------------------------------------------------------- #
# completion-marker integration: run_complete.json blocked without valid contrasts
# --------------------------------------------------------------------------- #
_ENV = {"schema": 1, "captured_at": "2026-07-11T00:00:00+00:00", "python": "3.11.15",
        "numerical_core": {"numpy": "1.26.4", "scipy": "1.17.1", "scikit-learn": "1.9.0", "torch": "2.7.1"},
        "encoder_packages": {}, "cuda": {}, "git": {"commit": "abc", "dirty": False}}


def _publish_required(results_dir, rows):
    IOU.append_jsonl(results_dir / "probe_results.jsonl", rows)
    for name in ("probe_results.csv", "summary.csv"):
        (results_dir / name).write_text("{}\n")
    # deltas.csv is computed for real: these rows carry a random_id reference AND a geographic_ood
    # label-access result, so completion validation requires a genuine non-empty delta table.
    IOU.write_csv(results_dir / "deltas.csv", IOU.compute_deltas(rows, ["f1"]))
    IOU.write_json(results_dir / artifacts.ENVIRONMENT_FILE, _ENV)   # schema-valid for validate_run_complete


def test_run_complete_records_contrast_hashes(tmp_path):
    rows = _run_rows(seeds=(0,), targets=("kenya",))
    _publish_required(tmp_path, rows)
    contrasts.compute_and_write(tmp_path, rows)
    keys = {artifacts.cell_key(r) for r in rows}
    marker = artifacts.write_run_complete(tmp_path, run_manifest_sha256="sig", expected_keys=keys, rows=rows)
    assert contrasts.CONTRAST_FILE in marker["artifacts"]
    assert contrasts.CONTRAST_SUMMARY_FILE in marker["artifacts"]
    assert artifacts.read_run_complete(tmp_path) is not None


def test_run_complete_blocked_when_contrasts_missing(tmp_path):
    rows = _run_rows(seeds=(0,), targets=("kenya",))
    _publish_required(tmp_path, rows)
    # deliberately do NOT write the contrast artifacts
    keys = {artifacts.cell_key(r) for r in rows}
    with pytest.raises(artifacts.IncompleteRunError, match="contrast artifact"):
        artifacts.write_run_complete(tmp_path, run_manifest_sha256="sig", expected_keys=keys, rows=rows)
    assert artifacts.read_run_complete(tmp_path) is None


def test_run_complete_blocked_when_contrasts_stale(tmp_path):
    rows = _run_rows(seeds=(0,), targets=("kenya",))
    _publish_required(tmp_path, rows)
    contrasts.compute_and_write(tmp_path, rows)
    p = tmp_path / contrasts.CONTRAST_FILE
    p.write_bytes(p.read_bytes().replace(b"0.5", b"0.9"))   # a value inconsistent with probe_results
    keys = {artifacts.cell_key(r) for r in rows}
    with pytest.raises(artifacts.IncompleteRunError, match="contrast artifact"):
        artifacts.write_run_complete(tmp_path, run_manifest_sha256="sig", expected_keys=keys, rows=rows)


# --------------------------------------------------------------------------- #
# validate_run_complete (post-publication) also covers the contrast artifacts
# --------------------------------------------------------------------------- #
def _completed_run(tmp_path, rows):
    _publish_required(tmp_path, rows)
    contrasts.compute_and_write(tmp_path, rows)
    keys = {artifacts.cell_key(r) for r in rows}
    return artifacts.write_run_complete(tmp_path, run_manifest_sha256="sig", expected_keys=keys, rows=rows)


def test_validate_run_complete_accepts_a_valid_label_access_run(tmp_path):
    _completed_run(tmp_path, _run_rows(seeds=(0, 1), targets=("kenya", "togo", "mali")))
    ok, problems = artifacts.validate_run_complete(tmp_path)
    assert ok, problems


@pytest.mark.parametrize("artifact_name", ["label_access_contrasts.csv", "label_access_contrasts_summary.csv"])
def test_validate_run_complete_rejects_deleted_contrast_csv(tmp_path, artifact_name):
    _completed_run(tmp_path, _run_rows(seeds=(0,), targets=("kenya",)))
    assert artifacts.validate_run_complete(tmp_path)[0]        # valid before deletion
    (tmp_path / artifact_name).unlink()
    ok, problems = artifacts.validate_run_complete(tmp_path)
    assert not ok and any(artifact_name in p and "missing" in p for p in problems)


@pytest.mark.parametrize("artifact_name", ["label_access_contrasts.csv", "label_access_contrasts_summary.csv"])
def test_validate_run_complete_rejects_tampered_contrast_csv(tmp_path, artifact_name):
    _completed_run(tmp_path, _run_rows(seeds=(0,), targets=("kenya",)))
    path = tmp_path / artifact_name
    path.write_bytes(path.read_bytes() + b"tampered,extra,row\n")   # any edit -> no longer matches a recompute
    ok, problems = artifacts.validate_run_complete(tmp_path)
    assert not ok
    assert any(artifact_name in p and ("stale" in p or "inconsistent" in p or "changed" in p) for p in problems)


def test_validate_run_complete_rejects_contrast_inconsistent_with_probe_results(tmp_path):
    """Editing probe_results.jsonl after completion (so a fresh recompute no longer matches the CSV) is
    caught -- validation recomputes the contrasts, not just checks REQUIRED_ARTIFACTS hashes."""
    _completed_run(tmp_path, _run_rows(seeds=(0,), targets=("kenya",)))
    rows_path = tmp_path / "probe_results.jsonl"
    text = rows_path.read_text().replace('"f1": 0.8', '"f1": 0.42')   # move target_only_full
    rows_path.write_text(text)
    ok, problems = artifacts.validate_run_complete(tmp_path)
    assert not ok and any("contrast" in p.lower() or "inconsistent" in p for p in problems)


# --------------------------------------------------------------------------- #
# metric operand integrity + explicit direction policy
# --------------------------------------------------------------------------- #
def test_missing_metric_value_hard_fails():
    rows = _run_rows()
    for r in rows:  # drop f1 from the target_only_full operand
        if r.get("label_access_route") == R.ROUTE_TARGET_ONLY_FULL:
            del r["f1"]
    with pytest.raises(contrasts.ContrastError, match="missing metric"):
        contrasts.compute_contrasts(rows)


def test_missing_metric_value_on_the_e1_operand_hard_fails():
    rows = _run_rows()
    for r in rows:
        if (r.get("budget_type") == "source" and r.get("split_regime") == R.LABEL_ACCESS_REGIME
                and r.get("evaluation_split") == "test"):
            del r["f1"]
    with pytest.raises(contrasts.ContrastError, match="missing metric"):
        contrasts.compute_contrasts(rows)


def test_malformed_metric_value_hard_fails():
    rows = _run_rows()
    for r in rows:
        if (r.get("label_access_route") == R.ROUTE_FIXED_BUDGET_ALLOCATION
                and r.get("label_budget") == 0):
            r["f1"] = "not-a-number"
    with pytest.raises(contrasts.ContrastError, match="malformed"):
        contrasts.compute_contrasts(rows)


def test_none_metric_value_hard_fails():
    rows = _run_rows()
    for r in rows:
        if r.get("label_access_route") == R.ROUTE_SOURCE_PLUS_TARGET_FULL:
            r["f1"] = None
    with pytest.raises(contrasts.ContrastError, match="not numeric"):
        contrasts.compute_contrasts(rows)


def test_boolean_metric_value_hard_fails():
    rows = _run_rows()
    for r in rows:
        if r.get("label_access_route") == R.ROUTE_TARGET_ONLY_FULL:
            r["f1"] = True
    with pytest.raises(contrasts.ContrastError, match="boolean"):
        contrasts.compute_contrasts(rows)


def test_genuine_nan_metric_is_kept_not_errored():
    rows = _run_rows()
    for r in rows:
        if r.get("label_access_route") == R.ROUTE_TARGET_ONLY_FULL:
            r["f1"] = float("nan")
    paired, _ = contrasts.compute_contrasts(rows)                    # no error -- genuine NaN is valid
    row = _paired_by(paired, "target_label_advantage")
    assert np.isnan(row["minuend_value"]) and np.isnan(row["difference"])


def test_string_nan_and_inf_are_valid_numeric():
    rows = _run_rows()
    for r in rows:
        if r.get("label_access_route") == R.ROUTE_SOURCE_PLUS_TARGET_FULL:
            r["f1"] = "inf"
    with np.errstate(invalid="ignore"):   # inf propagates into the bootstrap; that is expected
        paired, _ = contrasts.compute_contrasts(rows)
    assert np.isinf(_paired_by(paired, "full_supervision_gain")["minuend_value"])


def test_metric_direction_policy_is_explicit_and_exhaustive():
    # the whole METRICS_* universe is explicitly classified -- no metric falls through to a default.
    for m in contrasts._metric_universe():
        assert contrasts.metric_direction(m) in ("higher_is_better", "lower_is_better", "structural"), m
    assert contrasts.metric_direction("f1") == "higher_is_better"
    assert contrasts.metric_direction("miou") == "higher_is_better"
    assert contrasts.metric_direction("ece") == "lower_is_better"
    assert contrasts.metric_direction("n_classes_unseen") == "lower_is_better"      # fewer unseen = better
    assert contrasts.metric_direction("n_tiles_scored") == "structural"             # count, not performance
    assert contrasts.metric_direction("n_classes_seen") == "structural"
    # an UNKNOWN metric is NOT silently classified higher_is_better.
    assert contrasts.metric_direction("some_new_metric") == "unknown"
