"""Stage 5: paired label-access post-processing (pure post-processing on completed probe rows).

Covers every canonical contrast + its exact k pairing, source_ID_reference anchor resolution, exclusion
of the source_only complete-target diagnostic, equal-region within-seed aggregation, three-seed
uncertainty (mean/std/n_seeds/bootstrap CI), the duplicate/missing-operand hard failures, and the
completion-marker validation (missing / stale / hash). Tabular (samples) AND PASTIS (patches)."""

from __future__ import annotations

import numpy as np
import pytest

from evals import contrasts
from evals import split_artifacts as SA
from utils import artifacts
from utils import ioutils as IOU

R = SA  # route/anchor/contrast constants live on split_artifacts
B, P, T, M = 60, 55, 12, 55   # source base, target pool, target_test, matched size = min(B, P)

# Distinct per-(route, budget) metric values so each contrast has a uniquely checkable difference.
VAL: dict[tuple, float] = {
    (R.ROUTE_SOURCE_ONLY, 0): 0.50,
    (R.ROUTE_SOURCE_PLUS_TARGET, 5): 0.55, (R.ROUTE_SOURCE_PLUS_TARGET, 10): 0.60,
    (R.ROUTE_SOURCE_PLUS_TARGET, 25): 0.65, (R.ROUTE_SOURCE_PLUS_TARGET, 50): 0.70,
    (R.ROUTE_TARGET_ONLY_FULL, 0): 0.80, (R.ROUTE_SOURCE_PLUS_TARGET_FULL, 0): 0.85,
    (R.ROUTE_MATCHED_SOURCE, 0): 0.40, (R.ROUTE_MATCHED_TARGET, 0): 0.75,
    (R.ROUTE_FIXED_TOTAL_MIXED, 5): 0.52, (R.ROUTE_FIXED_TOTAL_MIXED, 10): 0.54,
    (R.ROUTE_FIXED_TOTAL_MIXED, 25): 0.56, (R.ROUTE_FIXED_TOTAL_MIXED, 50): 0.58,
}
DIAG_VALUE = 0.99   # source_only complete_target diagnostic -- MUST be excluded from every contrast
ANCHOR_VALUE = 0.90


def _counts(route, budget):
    return {
        R.ROUTE_SOURCE_ONLY: (B, 0, B),
        R.ROUTE_SOURCE_PLUS_TARGET: (B, budget, B + budget),
        R.ROUTE_TARGET_ONLY_FULL: (0, P, P),
        R.ROUTE_SOURCE_PLUS_TARGET_FULL: (B, P, B + P),
        R.ROUTE_MATCHED_SOURCE: (M, 0, M),
        R.ROUTE_MATCHED_TARGET: (0, M, M),
        R.ROUTE_FIXED_TOTAL_MIXED: (B - budget, budget, B),
    }[route]


def _la(route, budget, seed, target, value, *, es=None, benchmark="cropharvest", unit="samples", metric="f1"):
    es = es or R.EVAL_TARGET_TEST
    ns, nt, ntot = _counts(route, budget)
    n_units = (T + P) if es == R.EVAL_COMPLETE_TARGET else T   # evaluated units in the label unit
    row = {
        "benchmark": benchmark, "model": "raw", "probe_family": "logistic", "seed": seed,
        "holdout": target, "split_regime": R.LABEL_ACCESS_REGIME, "budget_type": "label_access",
        "evaluation_split": es, "label_access_route": route, "label_budget": budget,
        "n_source_labels": ns, "n_target_labels": nt, "n_total_labels": ntot,
        "label_budget_unit": unit, metric: value,
    }
    if unit == "patches":
        row["n_eval_patches"] = n_units          # patch count the validator uses to derive P
        row["n_test"] = n_units * 10             # evaluated PIXELS (a different unit; not used for P)
    else:
        row["n_test"] = n_units                  # samples == the label unit
    return row


def _anchor(seed, value, *, benchmark="cropharvest", metric="f1"):
    return {
        "benchmark": benchmark, "model": "raw", "probe_family": "logistic", "seed": seed,
        "holdout": "random_id", "split_regime": "random_id", "budget_type": "source",
        "evaluation_split": "test", "label_budget": 1.0, "label_access_route": "", metric: value,
    }


def _cell(seed, target, *, benchmark="cropharvest", unit="samples", metric="f1", overrides=None):
    """The 14 valid label-access rows of one cell + the source_only complete-target diagnostic."""
    vals = {**VAL, **(overrides or {})}
    rows = []
    for route, budget, es in R.label_access_expected_rows():
        v = DIAG_VALUE if es == R.EVAL_COMPLETE_TARGET else vals[(route, budget)]
        rows.append(_la(route, budget, seed, target, v, es=es, benchmark=benchmark, unit=unit, metric=metric))
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


# --------------------------------------------------------------------------- #
# every canonical contrast + exact k pairing + diagnostic exclusion (tabular)
# --------------------------------------------------------------------------- #
def test_all_six_contrasts_and_k_pairing_tabular():
    paired, _summary = contrasts.compute_contrasts(_run_rows())
    assert _paired_by(paired, "target_label_advantage")["difference"] == pytest.approx(0.30)   # tof - so
    assert _paired_by(paired, "target_reference_deficit")["difference"] == pytest.approx(0.10)  # anchor - tof
    assert _paired_by(paired, "full_supervision_gain")["difference"] == pytest.approx(0.35)     # sptf - so
    assert _paired_by(paired, "size_matched_source_target_difference")["difference"] == pytest.approx(0.35)  # mt - ms
    # additive(k) = source_plus_target(k) - source_only ; each k paired EXACTLY.
    for k, exp in zip(SA.LABEL_ACCESS_COUNTS, (0.05, 0.10, 0.15, 0.20), strict=True):
        assert _paired_by(paired, "additive_target_label_gain", budget=k)["difference"] == pytest.approx(exp)
    # label_source_allocation(k) = fixed_total_mixed(k) - source_only.
    for k, exp in zip(SA.LABEL_ACCESS_COUNTS, (0.02, 0.04, 0.06, 0.08), strict=True):
        assert _paired_by(paired, "label_source_allocation_effect", budget=k)["difference"] == pytest.approx(exp)


def test_diagnostic_complete_target_is_excluded():
    """target_label_advantage uses source_only on target_test (0.50), never the 0.99 complete-target
    diagnostic; if the diagnostic leaked, the value would be wrong and source_only would duplicate."""
    paired, _ = contrasts.compute_contrasts(_run_rows())
    row = _paired_by(paired, "target_label_advantage")
    assert row["subtrahend_value"] == pytest.approx(0.50)   # source_only@target_test, not 0.99
    assert row["minuend_value"] == pytest.approx(0.80)
    # no paired row references the complete_target split
    assert all(r["evaluation_split"] == R.EVAL_TARGET_TEST for r in paired)


def test_provenance_columns_present_and_ordered():
    paired, summary = contrasts.compute_contrasts(_run_rows())
    r = _paired_by(paired, "additive_target_label_gain", budget=25)
    for col in ("minuend_route", "subtrahend_route", "minuend_value", "subtrahend_value",
                "subtraction_order", "metric", "metric_direction", "budget", "seed", "target",
                "benchmark", "model", "probe_family", "difference"):
        assert col in r
    assert r["minuend_route"] == R.ROUTE_SOURCE_PLUS_TARGET and r["subtrahend_route"] == R.ROUTE_SOURCE_ONLY
    assert r["subtraction_order"] == f"{R.ROUTE_SOURCE_PLUS_TARGET} - {R.ROUTE_SOURCE_ONLY}"
    assert r["metric_direction"] == "higher_is_better"
    # target_reference_deficit's minuend is the anchor route name
    assert _paired_by(paired, "target_reference_deficit")["minuend_route"] == R.ANCHOR_SOURCE_ID_REFERENCE
    assert all("region_weighting" in s and s["region_weighting"] == "equal" for s in summary)


def test_metric_direction_error_metrics_are_lower_is_better():
    assert contrasts.metric_direction("f1") == "higher_is_better"
    assert contrasts.metric_direction("miou") == "higher_is_better"
    for m in ("ece", "nll", "brier", "shared_nll", "union_brier", "top_label_ece_all"):
        assert contrasts.metric_direction(m) == "lower_is_better"


# --------------------------------------------------------------------------- #
# PASTIS (patches) -- identical contrast logic, patch unit
# --------------------------------------------------------------------------- #
def test_all_contrasts_pastis_patches():
    rows = _run_rows(benchmark="pastis", unit="patches", metric="miou")
    paired, summary = contrasts.compute_contrasts(rows)
    assert _paired_by(paired, "target_label_advantage")["difference"] == pytest.approx(0.30)
    assert _paired_by(paired, "additive_target_label_gain", budget=50)["difference"] == pytest.approx(0.20)
    assert {r["metric"] for r in paired} == {"miou"}
    assert {s["benchmark"] for s in summary} == {"pastis"}


# --------------------------------------------------------------------------- #
# anchor resolution: exactly one random_id full-source or a clear failure
# --------------------------------------------------------------------------- #
def test_anchor_resolves_from_random_id_full_source():
    rows = _run_rows()
    assert _paired_by(contrasts.compute_contrasts(rows)[0], "target_reference_deficit")["minuend_value"] == pytest.approx(0.90)


def test_missing_anchor_hard_fails():
    rows = [r for r in _run_rows() if r["split_regime"] != "random_id"]   # drop the anchor
    with pytest.raises(contrasts.ContrastError, match="source_ID_reference"):
        contrasts.compute_contrasts(rows)


def test_duplicate_anchor_hard_fails():
    rows = _run_rows() + [_anchor(0, 0.91)]   # a second random_id full-source row for the same cell
    with pytest.raises(contrasts.ContrastError, match="not unique"):
        contrasts.compute_contrasts(rows)


def test_only_full_source_random_id_is_the_anchor():
    """A partial-source random_id row (budget < 1.0) must NOT be picked as the anchor."""
    rows = _run_rows()
    rows.append({**_anchor(0, 0.10), "label_budget": 0.1})   # a non-full random_id source row
    assert _paired_by(contrasts.compute_contrasts(rows)[0], "target_reference_deficit")["minuend_value"] == pytest.approx(0.90)


# --------------------------------------------------------------------------- #
# missing / duplicate operand hard failures (never silently dropped)
# --------------------------------------------------------------------------- #
def test_missing_operand_hard_fails():
    rows = [r for r in _run_rows() if r["label_access_route"] != R.ROUTE_TARGET_ONLY_FULL]
    with pytest.raises(contrasts.ContrastError, match="missing operand"):
        contrasts.compute_contrasts(rows)


def test_duplicate_operand_hard_fails():
    rows = _run_rows()
    rows.append(_la(R.ROUTE_SOURCE_ONLY, 0, 0, "kenya", 0.51))   # a second source_only@target_test
    with pytest.raises(contrasts.ContrastError, match="duplicate label-access operand"):
        contrasts.compute_contrasts(rows)


# --------------------------------------------------------------------------- #
# aggregation: equal region weighting, then three-seed uncertainty
# --------------------------------------------------------------------------- #
def test_equal_region_weighting_never_sample_weighted():
    # two targets in one seed with different target_test SIZES; the per-seed value is the SIMPLE mean of
    # their contrasts (0.30 and 0.20 -> 0.25), never the size-weighted 0.225.
    kenya = _cell(0, "kenya")
    togo = _cell(0, "togo", overrides={(R.ROUTE_TARGET_ONLY_FULL, 0): 0.70})   # togo advantage = 0.20
    for r in togo:                                    # make togo's region far larger, to expose weighting
        r["n_test"] = r["n_test"] * 25
    rows = kenya + togo + [_anchor(0, ANCHOR_VALUE)]
    _paired, summary = contrasts.compute_contrasts(rows)
    s = next(x for x in summary if x["contrast"] == "target_label_advantage")
    assert s["mean_difference"] == pytest.approx(0.25)   # equal weight, NOT 0.225
    assert s["region_weighting"] == "equal" and s["n_targets_per_seed"] == "2"


def test_three_seed_uncertainty():
    # target_label_advantage per seed = 0.30, 0.40, 0.50 (via source_only 0.50 and target_only_full varied)
    rows = []
    for s, tof in zip((0, 1, 2), (0.80, 0.90, 1.00), strict=True):
        rows.extend(_cell(s, "kenya", overrides={(R.ROUTE_TARGET_ONLY_FULL, 0): tof}))
        rows.append(_anchor(s, ANCHOR_VALUE))
    _paired, summary = contrasts.compute_contrasts(rows)
    s = next(x for x in summary if x["contrast"] == "target_label_advantage")
    assert s["n_seeds"] == 3
    assert s["mean_difference"] == pytest.approx(0.40)
    assert s["std_difference"] == pytest.approx(float(np.std([0.30, 0.40, 0.50])))
    assert s["ci_convention"] == "hierarchical_seed_target_bootstrap_2.5_97.5"
    assert s["ci_lo"] <= s["mean_difference"] <= s["ci_hi"]


def test_aggregation_is_deterministic():
    rows = _run_rows(seeds=(0, 1, 2))
    a = contrasts.compute_contrasts(rows)[1]
    b = contrasts.compute_contrasts(rows)[1]
    assert contrasts._csv_bytes(a) == contrasts._csv_bytes(b)   # fixed-seed bootstrap -> identical bytes


# --------------------------------------------------------------------------- #
# write + completion validation (missing / stale / hash)
# --------------------------------------------------------------------------- #
def test_write_and_validate_round_trip(tmp_path):
    rows = _run_rows(seeds=(0, 1, 2), targets=("kenya", "togo"))
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
# G1: validate_run_complete (post-publication) also covers the contrast artifacts
# --------------------------------------------------------------------------- #
def _completed_run(tmp_path, rows):
    _publish_required(tmp_path, rows)
    contrasts.compute_and_write(tmp_path, rows)
    keys = {artifacts.cell_key(r) for r in rows}
    return artifacts.write_run_complete(tmp_path, run_manifest_sha256="sig", expected_keys=keys, rows=rows)


def test_validate_run_complete_accepts_a_valid_label_access_run(tmp_path):
    _completed_run(tmp_path, _run_rows(seeds=(0, 1), targets=("kenya", "togo")))
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
# G2: hierarchical seed->target bootstrap captures target uncertainty
# --------------------------------------------------------------------------- #
def _seed_only_ci(per_seed_means):
    rng = np.random.default_rng(0)
    arr = np.asarray(per_seed_means, dtype=float)
    boot = np.array([arr[rng.integers(0, arr.size, arr.size)].mean() for _ in range(2000)])
    return float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def test_hierarchical_bootstrap_captures_target_uncertainty_when_seed_means_coincide():
    # 3 seeds x 2 targets; per-target advantage differs but EVERY seed's equal-region mean is exactly 0.30.
    tof = {(0, "kenya"): 0.70, (0, "togo"): 0.90, (1, "kenya"): 0.60, (1, "togo"): 1.00,
           (2, "kenya"): 0.75, (2, "togo"): 0.85}
    rows = []
    for seed in (0, 1, 2):
        for target in ("kenya", "togo"):
            rows.extend(_cell(seed, target, overrides={(R.ROUTE_TARGET_ONLY_FULL, 0): tof[(seed, target)]}))
        rows.append(_anchor(seed, ANCHOR_VALUE))
    _paired, summary = contrasts.compute_contrasts(rows)
    s = next(x for x in summary if x["contrast"] == "target_label_advantage")

    # point estimate preserved: equal-region mean within seed (0.30 each), then across seeds.
    assert s["mean_difference"] == pytest.approx(0.30)
    assert s["std_difference"] == pytest.approx(0.0)                 # the seed means coincide
    # a seed-ONLY bootstrap over the coincident per-seed means [0.30, 0.30, 0.30] is DEGENERATE ...
    lo0, hi0 = _seed_only_ci([0.30, 0.30, 0.30])
    assert lo0 == pytest.approx(0.30) and hi0 == pytest.approx(0.30)
    # ... but the required seed->target bootstrap captures the within-seed target spread.
    assert s["ci_hi"] > s["ci_lo"]
    assert s["ci_lo"] < 0.30 < s["ci_hi"]
    assert s["ci_convention"] == "hierarchical_seed_target_bootstrap_2.5_97.5"


def test_hierarchical_bootstrap_is_deterministic():
    rows = _run_rows(seeds=(0, 1, 2), targets=("kenya", "togo"))
    a = contrasts.compute_contrasts(rows)[1]
    b = contrasts.compute_contrasts(rows)[1]
    assert contrasts._csv_bytes(a) == contrasts._csv_bytes(b)


# --------------------------------------------------------------------------- #
# G3: metric operand integrity + explicit direction policy
# --------------------------------------------------------------------------- #
def test_missing_metric_value_hard_fails():
    rows = _run_rows()
    for r in rows:  # drop f1 from the target_only_full operand
        if r.get("label_access_route") == R.ROUTE_TARGET_ONLY_FULL and r.get("evaluation_split") == R.EVAL_TARGET_TEST:
            del r["f1"]
    with pytest.raises(contrasts.ContrastError, match="missing metric"):
        contrasts.compute_contrasts(rows)


def test_malformed_metric_value_hard_fails():
    rows = _run_rows()
    for r in rows:
        if r.get("label_access_route") == R.ROUTE_SOURCE_ONLY and r.get("evaluation_split") == R.EVAL_TARGET_TEST:
            r["f1"] = "not-a-number"
    with pytest.raises(contrasts.ContrastError, match="malformed"):
        contrasts.compute_contrasts(rows)


def test_none_metric_value_hard_fails():
    rows = _run_rows()
    for r in rows:
        if r.get("label_access_route") == R.ROUTE_MATCHED_TARGET:
            r["f1"] = None
    with pytest.raises(contrasts.ContrastError, match="not numeric"):
        contrasts.compute_contrasts(rows)


def test_genuine_nan_metric_is_kept_not_errored():
    rows = _run_rows()
    for r in rows:
        if r.get("label_access_route") == R.ROUTE_TARGET_ONLY_FULL and r.get("evaluation_split") == R.EVAL_TARGET_TEST:
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
