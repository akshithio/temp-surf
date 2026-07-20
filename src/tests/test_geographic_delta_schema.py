"""Geographic ID-to-OOD deltas under the schema-v2 label-access row shape.

The label-access suite replaced the legacy target-budget sweep for ``geographic_ood``, so the
deployment OOD score now arrives as ``budget_type="label_access"`` / ``label_access_route="source_only"``
/ ``evaluation_split="complete_target"`` instead of ``budget_type="target"`` / ``label_budget=0``.
Selecting only the legacy shape produced an EMPTY ``deltas.csv`` that still certified as complete.

These tests pin: the new shape resolves, it agrees numerically with the legacy shape, the two
source-only rows are not confused (complete_target is the deployment score, target_test is the paired
contrast operand), and an empty/incomplete delta table can no longer certify a run that should have one.
"""

from __future__ import annotations

import pytest

from evals import confounds
from evals import split_artifacts as SA
from utils import artifacts
from utils import ioutils as IOU

METRIC = "f1"
KEYS = {"model": "raw", "benchmark": "cropharvest", "method": "erm", "probe_family": "logistic"}


def _row(**kw):
    return {**KEYS, "seed": 0, "target_role": "headline", METRIC: 0.5, **kw}


def _id_row(value=0.80, seed=0):
    return _row(split_regime="random_id", holdout="random_id", budget_type="source",
                label_budget=1.0, evaluation_split="test", label_access_route="", seed=seed,
                **{METRIC: value})


def _la_row(route, es, value, *, holdout="kenya", seed=0, budget=0):
    return _row(split_regime="geographic_ood", holdout=holdout, budget_type="label_access",
                label_budget=budget, evaluation_split=es, label_access_route=route, seed=seed,
                **{METRIC: value})


def _legacy_ood_row(value, *, holdout="kenya", seed=0, budget=0, es="full"):
    return _row(split_regime="geographic_ood", holdout=holdout, budget_type="target",
                label_budget=budget, evaluation_split=es, label_access_route="", seed=seed,
                **{METRIC: value})


# --------------------------------------------------------------------------- #
# the new schema resolves and agrees with the legacy schema
# --------------------------------------------------------------------------- #
def test_new_schema_geographic_pair_produces_a_nonempty_delta():
    rows = [
        _id_row(0.80),
        _la_row(SA.ROUTE_SOURCE_ONLY, SA.EVAL_COMPLETE_TARGET, 0.50, holdout="kenya"),
        _la_row(SA.ROUTE_SOURCE_ONLY, SA.EVAL_COMPLETE_TARGET, 0.60, holdout="brazil"),
    ]
    deltas = IOU.compute_deltas(rows, [METRIC])
    assert len(deltas) == 1
    d = deltas[0]
    assert d["ood_regime"] == "geographic_ood"
    assert d["id"] == pytest.approx(0.80)
    assert d["ood"] == pytest.approx(0.55)          # equal-weight mean of the two held-out regions
    assert d["delta"] == pytest.approx(0.25)
    assert d["n_ood"] == 2


def test_new_and_legacy_schemas_yield_the_same_delta():
    id_row = _id_row(0.80)
    new = IOU.compute_deltas(
        [id_row,
         _la_row(SA.ROUTE_SOURCE_ONLY, SA.EVAL_COMPLETE_TARGET, 0.50, holdout="kenya"),
         _la_row(SA.ROUTE_SOURCE_ONLY, SA.EVAL_COMPLETE_TARGET, 0.60, holdout="brazil")],
        [METRIC],
    )
    legacy = IOU.compute_deltas(
        [id_row,
         _legacy_ood_row(0.50, holdout="kenya"),
         _legacy_ood_row(0.60, holdout="brazil")],
        [METRIC],
    )
    for field in ("id", "ood", "delta", "n_ood"):
        assert new[0][field] == pytest.approx(legacy[0][field]), field


def test_complete_target_is_the_deployment_score_not_target_test():
    """Both source_only rows exist in a real run; only the complete_target one is the OOD score."""
    rows = [
        _id_row(0.80),
        _la_row(SA.ROUTE_SOURCE_ONLY, SA.EVAL_COMPLETE_TARGET, 0.50),   # deployment score
        _la_row(SA.ROUTE_SOURCE_ONLY, SA.EVAL_TARGET_TEST, 0.30),       # paired contrast operand
    ]
    d = IOU.compute_deltas(rows, [METRIC])[0]
    assert d["ood"] == pytest.approx(0.50)
    assert d["n_ood"] == 1


def test_target_test_source_only_is_the_operand_paired_with_the_reference():
    """ood_matched pairs with the target reference on IDENTICAL target_test examples."""
    rows = [
        _id_row(0.80),
        _la_row(SA.ROUTE_SOURCE_ONLY, SA.EVAL_COMPLETE_TARGET, 0.50),
        _la_row(SA.ROUTE_SOURCE_ONLY, SA.EVAL_TARGET_TEST, 0.30),
        _la_row(SA.ROUTE_TARGET_ONLY_FULL, SA.EVAL_TARGET_TEST, 0.70),
    ]
    d = IOU.compute_deltas(rows, [METRIC])[0]
    assert d["ood"] == pytest.approx(0.50)              # deployment score: complete_target
    assert d["target_id"] == pytest.approx(0.70)        # reference: target_only_full on target_test
    assert d["ood_matched"] == pytest.approx(0.30)      # operand: source_only on target_test
    assert d["adjusted_delta"] == pytest.approx(0.40)   # 0.70 - 0.30, both on target_test


def test_route_qualification_separates_rows_sharing_budget_and_eval_split():
    """source_only and target_only_full share seed/budget/eval split -- only the route distinguishes them."""
    rows = [
        _id_row(0.80),
        _la_row(SA.ROUTE_SOURCE_ONLY, SA.EVAL_TARGET_TEST, 0.30),
        _la_row(SA.ROUTE_TARGET_ONLY_FULL, SA.EVAL_TARGET_TEST, 0.70),
        _la_row(SA.ROUTE_MATCHED_TARGET, SA.EVAL_TARGET_TEST, 0.65),
        _la_row(SA.ROUTE_SOURCE_ONLY, SA.EVAL_COMPLETE_TARGET, 0.50),
    ]
    d = IOU.compute_deltas(rows, [METRIC])[0]
    assert d["ood"] == pytest.approx(0.50)
    assert d["ood_matched"] == pytest.approx(0.30)
    assert d["target_id"] == pytest.approx(0.70)


def test_worst_region_uses_the_new_schema_anchor():
    rows = [
        _id_row(0.80),
        _la_row(SA.ROUTE_SOURCE_ONLY, SA.EVAL_COMPLETE_TARGET, 0.20, holdout="kenya"),
        _la_row(SA.ROUTE_SOURCE_ONLY, SA.EVAL_COMPLETE_TARGET, 0.60, holdout="brazil"),
    ]
    d = IOU.compute_deltas(rows, [METRIC])[0]
    assert d["ood_worst_region"] == pytest.approx(0.20)


def test_supplementary_stress_targets_stay_out_of_the_new_schema_aggregation():
    stress = _la_row(SA.ROUTE_SOURCE_ONLY, SA.EVAL_COMPLETE_TARGET, 0.05, holdout="tanzania")
    stress["target_role"] = "supplementary_stress"
    rows = [
        _id_row(0.80),
        _la_row(SA.ROUTE_SOURCE_ONLY, SA.EVAL_COMPLETE_TARGET, 0.50, holdout="kenya"),
        stress,
    ]
    d = IOU.compute_deltas(rows, [METRIC])[0]
    assert d["ood"] == pytest.approx(0.50)
    assert d["n_ood"] == 1


def test_regime_discovery_sees_label_access_rows():
    """Discovery gated on budget_type=='target' alone found nothing once geographic_ood moved schema."""
    rows = [_id_row(0.80), _la_row(SA.ROUTE_SOURCE_ONLY, SA.EVAL_COMPLETE_TARGET, 0.50)]
    assert IOU.compute_deltas(rows, [METRIC])[0]["ood_regime"] == "geographic_ood"


def test_anchor_order_prefers_the_new_schema_over_a_stale_legacy_row():
    """A tree carrying both shapes resolves on the schema-v2 row, never the historical one."""
    rows = [
        _id_row(0.80),
        _la_row(SA.ROUTE_SOURCE_ONLY, SA.EVAL_COMPLETE_TARGET, 0.50),
        _legacy_ood_row(0.10),          # stale pre-label-access row for the same cell
    ]
    assert IOU.compute_deltas(rows, [METRIC])[0]["ood"] == pytest.approx(0.50)


# --------------------------------------------------------------------------- #
# an empty / incomplete delta table can no longer certify
# --------------------------------------------------------------------------- #
def _geographic_rows():
    return [_id_row(0.80), _la_row(SA.ROUTE_SOURCE_ONLY, SA.EVAL_COMPLETE_TARGET, 0.50)]


def test_expects_geographic_deltas_requires_both_legs():
    assert artifacts.expects_geographic_deltas(_geographic_rows())
    assert not artifacts.expects_geographic_deltas([_id_row(0.80)])
    assert not artifacts.expects_geographic_deltas(
        [_la_row(SA.ROUTE_SOURCE_ONLY, SA.EVAL_COMPLETE_TARGET, 0.50)]
    )


@pytest.mark.parametrize("content", ["", "\n", "   \n"])
def test_empty_deltas_csv_cannot_certify_a_geographic_run(tmp_path, content):
    (tmp_path / "deltas.csv").write_text(content)
    problems = artifacts._validate_deltas(tmp_path, _geographic_rows())
    assert problems and "empty" in problems[0]


def test_header_only_deltas_csv_cannot_certify(tmp_path):
    (tmp_path / "deltas.csv").write_text("metric,id,ood,delta,ood_regime\n")
    problems = artifacts._validate_deltas(tmp_path, _geographic_rows())
    assert problems and "no delta rows" in problems[0]


def test_deltas_csv_missing_required_columns_cannot_certify(tmp_path):
    (tmp_path / "deltas.csv").write_text("metric,id\nf1,0.8\n")
    problems = artifacts._validate_deltas(tmp_path, _geographic_rows())
    assert problems and "missing required column" in problems[0]


def test_deltas_csv_without_a_geographic_row_cannot_certify(tmp_path):
    (tmp_path / "deltas.csv").write_text(
        "metric,id,ood,delta,ood_regime\nf1,0.8,0.5,0.3,official\n"
    )
    problems = artifacts._validate_deltas(tmp_path, _geographic_rows())
    assert problems and "no geographic_ood row" in problems[0]


def test_absent_deltas_csv_cannot_certify_a_geographic_run(tmp_path):
    problems = artifacts._validate_deltas(tmp_path, _geographic_rows())
    assert problems and "absent" in problems[0]


def test_a_real_delta_table_certifies(tmp_path):
    rows = _geographic_rows()
    IOU.write_csv(tmp_path / "deltas.csv", IOU.compute_deltas(rows, [METRIC]))
    assert artifacts._validate_deltas(tmp_path, rows) == []


def test_a_run_with_no_geographic_rows_is_not_forced_to_have_deltas(tmp_path):
    """Negative control: the guard must not fire on an embedding-only or random_id-only run."""
    (tmp_path / "deltas.csv").write_text("")
    assert artifacts._validate_deltas(tmp_path, [_id_row(0.80)]) == []


def test_anchor_specs_cover_both_schemas():
    """The legacy shape stays reachable for historical trees; the new shape is tried first."""
    new, legacy = confounds._ood_anchors(0.0)
    assert (new.budget_type, new.route, new.eval_splits) == (
        "label_access", SA.ROUTE_SOURCE_ONLY, (SA.EVAL_COMPLETE_TARGET,)
    )
    assert legacy.budget_type == "target" and legacy.route == ""


# --------------------------------------------------------------------------- #
# Worst-region aggregation order: average each target over seeds, THEN pick the worst
# --------------------------------------------------------------------------- #
def test_worst_region_averages_each_target_over_seeds_before_ranking():
    """Two-seed ranking reversal: which target is worst FLIPS between seeds.

    kenya  = (0.1, 0.9) -> mean 0.5      brazil = (0.9, 0.1) -> mean 0.5

    Averaging each target over seeds and then picking the worst gives 0.5 -- a value a region
    actually attains. Picking the worst target within each seed first and averaging those gives
    min(0.1,0.9)=0.1 and min(0.9,0.1)=0.1 -> 0.1, which no region attains in expectation and which
    simply tracks seed noise. The required answer is 0.5.
    """
    rows = [_id_row(0.80, seed=0), _id_row(0.80, seed=1)]
    for seed, (kenya, brazil) in enumerate([(0.1, 0.9), (0.9, 0.1)]):
        rows.append(_la_row(SA.ROUTE_SOURCE_ONLY, SA.EVAL_COMPLETE_TARGET, kenya, holdout="kenya", seed=seed))
        rows.append(_la_row(SA.ROUTE_SOURCE_ONLY, SA.EVAL_COMPLETE_TARGET, brazil, holdout="brazil", seed=seed))
    d = IOU.compute_deltas(rows, [METRIC])[0]
    assert d["ood_worst_region"] == pytest.approx(0.5)
    assert d["ood_worst_region"] != pytest.approx(0.1)


def test_worst_region_picks_the_genuinely_worst_target_not_the_noisiest():
    """A target that is consistently mediocre must lose to one that is consistently bad."""
    rows = [_id_row(0.80, seed=0), _id_row(0.80, seed=1)]
    for seed, (bad, ok) in enumerate([(0.20, 0.60), (0.24, 0.64)]):
        rows.append(_la_row(SA.ROUTE_SOURCE_ONLY, SA.EVAL_COMPLETE_TARGET, bad, holdout="sudan", seed=seed))
        rows.append(_la_row(SA.ROUTE_SOURCE_ONLY, SA.EVAL_COMPLETE_TARGET, ok, holdout="kenya", seed=seed))
    d = IOU.compute_deltas(rows, [METRIC])[0]
    assert d["ood_worst_region"] == pytest.approx(0.22)          # sudan's seed-mean
    assert d["ood_worst_region_std"] == pytest.approx(0.02)      # spread of THAT region


def test_worst_region_respects_metric_direction_after_seed_averaging():
    """For a lower-is-better metric the worst region is the HIGHEST seed-mean."""
    rows = [
        _row(split_regime="random_id", holdout="random_id", budget_type="source", label_budget=1.0,
             evaluation_split="test", label_access_route="", seed=s, brier=0.10)
        for s in (0, 1)
    ]
    for seed, (kenya, brazil) in enumerate([(0.20, 0.60), (0.60, 0.20)]):
        for holdout, v in (("kenya", kenya), ("brazil", brazil)):
            rows.append(_row(split_regime="geographic_ood", holdout=holdout, budget_type="label_access",
                             label_budget=0, evaluation_split=SA.EVAL_COMPLETE_TARGET,
                             label_access_route=SA.ROUTE_SOURCE_ONLY, seed=seed, brier=v))
    d = IOU.compute_deltas(rows, ["brier"])[0]
    # both regions average to 0.40; the worst (max, since lower is better) is 0.40 -- not the 0.60
    # a within-seed worst-then-average would have produced
    assert d["ood_worst_region"] == pytest.approx(0.40)


def test_worst_region_still_excludes_supplementary_stress_targets():
    stress = _la_row(SA.ROUTE_SOURCE_ONLY, SA.EVAL_COMPLETE_TARGET, 0.01, holdout="tanzania")
    stress["target_role"] = "supplementary_stress"
    rows = [
        _id_row(0.80),
        _la_row(SA.ROUTE_SOURCE_ONLY, SA.EVAL_COMPLETE_TARGET, 0.50, holdout="kenya"),
        stress,
    ]
    assert IOU.compute_deltas(rows, [METRIC])[0]["ood_worst_region"] == pytest.approx(0.50)
