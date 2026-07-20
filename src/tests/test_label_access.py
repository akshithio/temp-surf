"""Frozen geographic-OOD ``label_access.csv`` artifact: label-blind generation of TWO independent
orders (ONE nested source order + the target order), structural validation, the eligibility /
budget / allocation-audit PREDICATES, and runtime loading into two separately-ranked index arrays
consumed by the fixed-budget allocation curve and the additive routes.

The single source order is the scientific point of the current schema: every fixed-budget
allocation point slices ``source_rank[:B-k]``, so the five points are strictly NESTED rather than
five independent draws.
"""

from __future__ import annotations

import numpy as np
import pytest

from evals import split_artifacts as SA
from evals.regimes import base

SOURCE = [f"{100 + i}" for i in range(8)]        # source label pool ids
POOL = [f"{200 + i}" for i in range(6)]          # target label pool ids
TEST = [f"{300 + i}" for i in range(3)]          # target_test ids (never ranked)


def _rows(seed=0, source=SOURCE, pool=POOL, test=TEST):
    return SA.build_label_access_rows(seed=seed, source_ids=source, target_pool_ids=pool, target_test_ids=test)


def _map(rows, pop, col):
    return {r["stable_id"]: int(r[col]) for r in rows if r["population"] == pop and r[col] != ""}


def _kw():
    return {"source_ids": SOURCE, "target_pool_ids": POOL, "target_test_ids": TEST}


# --- schema: four columns, ONE source rank column -------------------------------------------------

def test_schema_is_four_columns_with_one_source_rank():
    assert SA.LABEL_ACCESS_HEADER == ["stable_id", "population", "source_rank", "target_rank"]
    assert SA.SOURCE_RANK_COLS == ("source_rank",)
    assert set(_rows(0)[0]) == set(SA.LABEL_ACCESS_HEADER)


# --- generation: deterministic, label-blind, contiguous, population-correct ------------------------

def test_generation_is_deterministic():
    assert SA._label_access_bytes(_rows(0)) == SA._label_access_bytes(_rows(0))


def test_generation_is_label_blind_and_input_order_independent():
    # build takes NO labels; the orders depend only on the id SET + seed, not input list order.
    a = _rows(0)
    b = _rows(0, source=list(reversed(SOURCE)), pool=[POOL[i] for i in (3, 0, 5, 1, 4, 2)])
    for pop, col in ((SA.POP_SOURCE, "source_rank"), (SA.POP_TARGET_POOL, "target_rank")):
        assert _map(a, pop, col) == _map(b, pop, col)


def test_generation_differs_by_seed():
    assert _map(_rows(0), SA.POP_TARGET_POOL, "target_rank") != _map(_rows(1), SA.POP_TARGET_POOL, "target_rank")
    assert _map(_rows(0), SA.POP_SOURCE, "source_rank") != _map(_rows(1), SA.POP_SOURCE, "source_rank")


def test_exactly_two_blind_draws_in_source_then_target_order():
    """The collapse from three draws to two is part of the frozen contract: the run seed feeds ONE
    Generator drawn twice, source first then target. A third (or reordered) draw would change every
    frozen artifact, so the draw sequence itself is locked here."""
    rng = np.random.default_rng(0)
    want_src = SA._blind_order(list(SOURCE), rng)
    want_tgt = SA._blind_order(list(POOL), rng)
    rows = _rows(0)
    assert _map(rows, SA.POP_SOURCE, "source_rank") == want_src
    assert _map(rows, SA.POP_TARGET_POOL, "target_rank") == want_tgt


def test_single_source_order_is_complete_and_contiguous():
    rows = _rows(0)
    src = _map(rows, SA.POP_SOURCE, "source_rank")
    # covers the whole source set exactly once, contiguously 0..S-1 ...
    assert set(src) == set(SOURCE)
    assert sorted(src.values()) == list(range(len(SOURCE)))
    # ... and is deterministic: same seed reproduces the one source order exactly.
    assert src == _map(_rows(0), SA.POP_SOURCE, "source_rank")


def test_allocation_source_prefixes_are_nested_as_the_target_share_grows():
    """The reason there is only ONE source order: at fraction f the cell trains on ``source_rank[:B-k]``,
    so as k grows the source prefix SHRINKS INTO the previous one. Two independent draws would make the
    five allocation points five unrelated source samples and destroy the curve's interpretation."""
    src_ids, _tgt_ids = SA.ranked_ids(_rows(0))
    budget = len(SOURCE)
    prefixes = []
    for pct in SA.ALLOCATION_PERCENTS:
        k = SA.allocation_target_count(pct, budget)
        prefixes.append(src_ids[: budget - k])
    # strictly nested (each prefix contained in the previous) and non-increasing in size
    for bigger, smaller in zip(prefixes[:-1], prefixes[1:], strict=True):
        assert len(smaller) <= len(bigger)
        assert set(smaller) <= set(bigger)
        assert smaller == bigger[: len(smaller)]
    assert prefixes[0] == src_ids[:budget] and prefixes[-1] == []


def test_target_rank_contiguous_and_target_test_unranked():
    rows = _rows(0)
    tgt = _map(rows, SA.POP_TARGET_POOL, "target_rank")
    assert set(tgt) == set(POOL) and sorted(tgt.values()) == list(range(len(POOL)))
    test_rows = [r for r in rows if r["population"] == SA.POP_TARGET_TEST]
    assert {r["stable_id"] for r in test_rows} == set(TEST)
    assert all(r["source_rank"] == "" and r["target_rank"] == "" for r in test_rows)


def test_cross_rank_blankness():
    """A source row carries no target_rank and a target-pool row no source_rank -- the populations are
    ranked in their own column only, so a prefix can never accidentally pull from the wrong pool."""
    rows = _rows(0)
    for r in rows:
        if r["population"] == SA.POP_SOURCE:
            assert r["source_rank"] != "" and r["target_rank"] == ""
        elif r["population"] == SA.POP_TARGET_POOL:
            assert r["target_rank"] != "" and r["source_rank"] == ""


def test_ranked_ids_returns_both_pools_in_frozen_rank_order():
    rows = _rows(0)
    src_ids, tgt_ids = SA.ranked_ids(rows)
    assert src_ids == _ordered_ids(rows, SA.POP_SOURCE, "source_rank")
    assert tgt_ids == _ordered_ids(rows, SA.POP_TARGET_POOL, "target_rank")
    assert set(src_ids) == set(SOURCE) and set(tgt_ids) == set(POOL)


def test_validate_passes_on_wellformed():
    assert SA.validate_label_access_rows(_rows(0), **_kw()) == "passed"


# --- malformed-order refusal (hard errors) --------------------------------------------------------

def _tamper(pop, col, value):
    rows = _rows(0)
    for r in rows:
        if r["population"] == pop:
            r[col] = value
            break
    return rows


def test_reject_noncontiguous_source_rank():
    with pytest.raises(SA.SplitArtifactError, match="contiguous"):
        SA.validate_label_access_rows(_tamper(SA.POP_SOURCE, "source_rank", "99"), **_kw())


def test_reject_noncontiguous_target_rank():
    with pytest.raises(SA.SplitArtifactError, match="contiguous"):
        SA.validate_label_access_rows(_tamper(SA.POP_TARGET_POOL, "target_rank", "99"), **_kw())


def test_reject_source_carrying_target_rank():
    with pytest.raises(SA.SplitArtifactError):
        SA.validate_label_access_rows(_tamper(SA.POP_SOURCE, "target_rank", "0"), **_kw())


def test_reject_target_carrying_source_rank():
    with pytest.raises(SA.SplitArtifactError):
        SA.validate_label_access_rows(_tamper(SA.POP_TARGET_POOL, "source_rank", "0"), **_kw())


def test_reject_target_test_with_rank():
    with pytest.raises(SA.SplitArtifactError, match="no rank"):
        SA.validate_label_access_rows(_tamper(SA.POP_TARGET_TEST, "target_rank", "0"), **_kw())


def test_reject_target_test_with_source_rank():
    with pytest.raises(SA.SplitArtifactError, match="no rank"):
        SA.validate_label_access_rows(_tamper(SA.POP_TARGET_TEST, "source_rank", "0"), **_kw())


def test_reject_population_mismatch():
    with pytest.raises(SA.SplitArtifactError, match="does not match"):
        SA.validate_label_access_rows(_rows(0), source_ids=SOURCE + ["999"], target_pool_ids=POOL, target_test_ids=TEST)


def test_reject_duplicate_id():
    rows = _rows(0)
    rows.append(dict(rows[0]))
    with pytest.raises(SA.SplitArtifactError, match="duplicate"):
        SA.validate_label_access_rows(rows, **_kw())


def test_reject_bad_header(tmp_path):
    p = tmp_path / "label_access.csv"
    p.write_text("wrong,header\n1,2\n")
    with pytest.raises(SA.SplitArtifactError, match="header"):
        SA.read_label_access_csv(p)


def test_reject_the_old_five_column_header(tmp_path):
    """The retired 5-column schema (two independent source draws) must not load as if it were current."""
    p = tmp_path / "label_access.csv"
    p.write_text("stable_id,population,matched_source_rank,fixed_source_removal_rank,target_rank\n"
                 "100,source,0,1,\n")
    with pytest.raises(SA.SplitArtifactError, match="header"):
        SA.read_label_access_csv(p)


# --- eligibility is a PREDICATE, not an assertion --------------------------------------------------

def test_eligibility_passes_at_boundary():
    # source == max(counts) is VALID: an additive route may consume all 50 source-equivalent units and
    # the target pool exactly supports the largest additive count.
    assert SA.label_access_eligibility(n_source=50, n_target_pool=50) == []
    assert SA.label_access_eligibility(n_source=51, n_target_pool=50) == []


def test_eligibility_reports_small_target_pool_without_raising():
    reasons = SA.label_access_eligibility(n_source=1000, n_target_pool=49)
    assert reasons and any("target_label_pool" in r for r in reasons)
    assert all("source_train" not in r for r in reasons)


def test_eligibility_reports_small_source_pool_without_raising():
    reasons = SA.label_access_eligibility(n_source=49, n_target_pool=1000)
    assert reasons and any("source_train" in r for r in reasons)
    assert all("target_label_pool" not in r for r in reasons)


def test_eligibility_reports_both_and_empty_pools():
    both = SA.label_access_eligibility(n_source=10, n_target_pool=10)
    assert len(both) == 2
    empty = SA.label_access_eligibility(n_source=0, n_target_pool=100)
    assert len(empty) == 1 and "empty pool" in empty[0]


def test_eligibility_never_raises_on_a_degenerate_region():
    """An undersized region is DEMOTED to supplementary stress, never an abort: a single small region
    must not be able to kill generation for the whole benchmark."""
    for n_s, n_t in ((0, 0), (1, 1), (-1, 5), (5, -1)):
        assert isinstance(SA.label_access_eligibility(n_source=n_s, n_target_pool=n_t), list)


def test_min_headline_targets_is_locked():
    assert SA.MIN_HEADLINE_TARGETS == 3


# --- the benchmark-common budget B_d ---------------------------------------------------------------

def test_benchmark_budget_is_the_min_over_cells():
    cells = [{"n_source": 100, "n_target_pool": 60}, {"n_source": 80, "n_target_pool": 70}]
    assert SA.benchmark_budget(cells) == 60


def test_an_ineligible_cell_excluded_by_the_caller_does_not_lower_the_budget():
    """The whole reason eligibility is a predicate: one tiny region would otherwise drag B_d down to
    its own size and shrink the allocation curve for every other region in the benchmark."""
    eligible_cells = [{"n_source": 100, "n_target_pool": 60}, {"n_source": 80, "n_target_pool": 70}]
    tiny = {"n_source": 12, "n_target_pool": 9}
    all_cells = [*eligible_cells, tiny]
    kept = [c for c in all_cells if not SA.label_access_eligibility(
        n_source=c["n_source"], n_target_pool=c["n_target_pool"])]
    assert kept == eligible_cells                       # the tiny cell is ineligible and filtered out
    assert SA.benchmark_budget(kept) == SA.benchmark_budget(eligible_cells) == 60
    assert SA.benchmark_budget(all_cells) == 9          # ... which is exactly what filtering prevents


def test_benchmark_budget_is_capped_by_max_label_budget():
    cells = [{"n_source": 100, "n_target_pool": 60}]
    assert SA.benchmark_budget(cells, max_label_budget=25) == 25
    assert SA.benchmark_budget(cells, max_label_budget=1000) == 60   # a ceiling only, never a floor


def test_benchmark_budget_refuses_an_empty_or_degenerate_budget():
    with pytest.raises(SA.SplitArtifactError, match="no eligible cells"):
        SA.benchmark_budget([])
    with pytest.raises(SA.SplitArtifactError, match="no allocation is constructible"):
        SA.benchmark_budget([{"n_source": 100, "n_target_pool": 60}], max_label_budget=0)


# --- post-construction allocation audit over the REALIZED prefixes ---------------------------------

def _classes(seq):
    return [frozenset({c}) for c in seq]


def test_audit_allocation_accepts_a_well_mixed_allocation():
    src = _classes("ababababab")
    tgt = _classes("babababa" + "ba")
    assert SA.audit_allocation(source_classes=src, target_classes=tgt, budget=8) == []


def test_audit_allocation_flags_a_single_class_prefix():
    """Row counts cannot see this: both pools are big enough, but the realized prefix at f=0 (all
    source) and at f=100 (all target) trains on ONE class and is unfittable."""
    src = _classes("aaaaaaaa")
    tgt = _classes("bbbbbbbb")
    problems = SA.audit_allocation(source_classes=src, target_classes=tgt, budget=4, percents=(0, 50, 100))
    assert len(problems) == 2
    assert any("f=0%" in p and "not fittable" in p for p in problems)
    assert any("f=100%" in p and "not fittable" in p for p in problems)
    # the mixed midpoint is fine, so the audit is per-fraction, not all-or-nothing
    assert all("f=50%" not in p for p in problems)


def test_audit_allocation_flags_an_undrawable_fraction():
    problems = SA.audit_allocation(
        source_classes=_classes("ab"), target_classes=_classes("abababab"), budget=8, percents=(0,))
    assert len(problems) == 1 and "needs 8 source units" in problems[0]
    problems = SA.audit_allocation(
        source_classes=_classes("abababab"), target_classes=_classes("ab"), budget=8, percents=(100,))
    assert len(problems) == 1 and "needs 8 target units" in problems[0]


def test_audit_allocation_carries_the_where_label():
    problems = SA.audit_allocation(
        source_classes=_classes("aa"), target_classes=_classes("aa"), budget=2, percents=(0,),
        where="cropharvest/Mali/seed0")
    assert problems and problems[0].startswith("cropharvest/Mali/seed0:")


# --- duplicate / overlap ids are hard errors (never silently de-duplicated) -----------------------

def test_build_rejects_duplicate_within_source():
    with pytest.raises(SA.SplitArtifactError, match="duplicate stable_id"):
        SA.build_label_access_rows(seed=0, source_ids=SOURCE + [SOURCE[0]], target_pool_ids=POOL, target_test_ids=TEST)


def test_build_rejects_duplicate_within_target_pool():
    with pytest.raises(SA.SplitArtifactError, match="duplicate stable_id"):
        SA.build_label_access_rows(seed=0, source_ids=SOURCE, target_pool_ids=POOL + [POOL[0]], target_test_ids=TEST)


def test_build_rejects_overlap_across_populations():
    with pytest.raises(SA.SplitArtifactError, match="appears in both"):
        SA.build_label_access_rows(seed=0, source_ids=SOURCE, target_pool_ids=POOL + [SOURCE[0]], target_test_ids=TEST)


def test_validate_rejects_overlap_across_populations():
    with pytest.raises(SA.SplitArtifactError, match="appears in both"):
        SA.validate_label_access_rows(_rows(0), source_ids=SOURCE, target_pool_ids=POOL, target_test_ids=TEST + [POOL[0]])


# --- the scientific contrast contract is locked and distinct --------------------------------------

def test_contrast_contract_is_locked_and_distinct():
    contract = {name: (minuend, subtrahend) for name, minuend, subtrahend in SA.LABEL_ACCESS_CONTRASTS}
    assert contract["target_label_advantage"] == (SA.ROUTE_TARGET_ONLY_FULL, SA.ANCHOR_GEOGRAPHIC_FULL_SOURCE)
    assert contract["allocation_effect"] == (SA.ROUTE_FIXED_BUDGET_ALLOCATION, SA.ALLOCATION_BASELINE)
    assert contract["additive_target_label_gain"] == (SA.ROUTE_SOURCE_PLUS_TARGET, SA.ANCHOR_GEOGRAPHIC_FULL_SOURCE)
    assert contract["full_supervision_gain"] == (SA.ROUTE_SOURCE_PLUS_TARGET_FULL, SA.ANCHOR_GEOGRAPHIC_FULL_SOURCE)
    assert len({n for n, _, _ in SA.LABEL_ACCESS_CONTRASTS}) == 4  # four DISTINCT questions, not merged
    assert len(SA.LABEL_ACCESS_CONTRASTS) == 4
    assert SA.LABEL_ACCESS_COUNTS == (5, 10, 25, 50)               # one canonical additive count set
    assert SA.ALLOCATION_PERCENTS == (0, 25, 50, 75, 100)          # one canonical allocation axis
    assert SA.ANCHOR_GEOGRAPHIC_FULL_SOURCE == "geographic_full_source"
    assert SA.ALLOCATION_BASELINE == "fixed_budget_allocation@0"


def test_the_retired_routes_are_gone():
    """source_only / matched_source / matched_target / fixed_total_mixed collapsed into the single
    fixed-budget allocation curve; leaving a stale alias around would let a caller silently fit the
    old, non-nested experiment."""
    for gone in ("ROUTE_SOURCE_ONLY", "ROUTE_MATCHED_SOURCE", "ROUTE_MATCHED_TARGET",
                 "ROUTE_FIXED_TOTAL_MIXED", "assert_label_access_feasible"):
        assert not hasattr(SA, gone), f"{gone} should have been removed from split_artifacts"


def test_routes_and_evaluation_split_identity():
    assert SA.ROUTE_FIXED_BUDGET_ALLOCATION == "fixed_budget_allocation"
    assert SA.LABEL_ACCESS_ROUTES == (
        SA.ROUTE_FIXED_BUDGET_ALLOCATION, SA.ROUTE_SOURCE_PLUS_TARGET,
        SA.ROUTE_TARGET_ONLY_FULL, SA.ROUTE_SOURCE_PLUS_TARGET_FULL,
    )
    # every label-access route is scored on the frozen target_test and nothing else.
    assert SA.LABEL_ACCESS_EVAL_SPLITS == (SA.EVAL_TARGET_TEST,)
    assert SA.EVAL_COMPLETE_TARGET not in SA.LABEL_ACCESS_EVAL_SPLITS


def test_allocation_target_count_is_explicit_half_up():
    """Never Python's banker's rounding: k must be reproducible across platforms and the source share
    exactly ``B - k``."""
    assert SA.allocation_target_count(50, 5) == 3     # 2.5 -> 3 (round() would give 2)
    assert SA.allocation_target_count(25, 10) == 3    # 2.5 -> 3
    assert SA.allocation_target_count(0, 40) == 0
    assert SA.allocation_target_count(100, 40) == 40
    for pct in SA.ALLOCATION_PERCENTS:
        k = SA.allocation_target_count(pct, 37)
        assert 0 <= k <= 37


def test_label_access_contract_keys():
    c = SA.label_access_contract(enabled=True, benchmark="cropharvest", controlled_budget_cap=64)
    assert set(c) == {
        "enabled", "allocation_percents", "additive_counts", "full_target_reference",
        "full_combined_reference", "controlled_budget_cap", "routes", "evaluation_splits", "unit",
    }
    assert "counts" not in c                                   # the old key is gone
    assert c["allocation_percents"] == list(SA.ALLOCATION_PERCENTS)
    assert c["additive_counts"] == list(SA.LABEL_ACCESS_COUNTS)
    assert c["controlled_budget_cap"] == 64
    assert c["routes"] == list(SA.LABEL_ACCESS_ROUTES)
    assert c["evaluation_splits"] == [SA.EVAL_TARGET_TEST]
    assert c["unit"] == SA.LABEL_ACCESS_TABULAR_UNIT
    assert SA.label_access_contract(enabled=True, benchmark="pastis")["unit"] == SA.LABEL_ACCESS_DENSE_UNIT


# --- write / read / load round-trip: two arrays consumed by the correct routes --------------------

def _headline_split(sample_ids):
    idx = {s: i for i, s in enumerate(sample_ids)}
    a = lambda ids: np.asarray([idx[s] for s in ids], dtype=np.int64)  # noqa: E731
    return base.SourceTargetSplit(
        label="R1", source_train=a(SOURCE), source_val=a(["108", "109"]), source_test=a(["110", "111"]),
        target_label_pool=a(POOL), target_test=a(TEST), has_target=True, supports_target_labels=True,
        group_kind="geography", target_role=base.TARGET_ROLE_HEADLINE,
    ), idx


def _ordered_ids(rows, pop, col):
    return [s for s, _ in sorted(_map(rows, pop, col).items(), key=lambda kv: kv[1])]


SAMPLE_IDS = SOURCE + ["108", "109", "110", "111"] + POOL + TEST


def test_write_load_resolves_two_orders_to_current_indices(tmp_path):
    split, id_map = _headline_split(SAMPLE_IDS)
    rows = _rows(2)
    _path, sha = SA.write_label_access(tmp_path, "cropharvest", 2, "R1", rows)
    loaded = SA.load_label_access(tmp_path, "cropharvest", 2, split, id_map, sha, benchmark_budget=6)

    # each array is a permutation of the right partition, mapped to CURRENT indices ...
    assert sorted(loaded.source_ranked_idx.tolist()) == sorted(id_map[s] for s in SOURCE)
    assert sorted(loaded.target_ranked_idx.tolist()) == sorted(id_map[s] for s in POOL)
    # ... consuming the CORRECT column (source <- source_rank, target <- target_rank).
    assert loaded.source_ranked_idx.tolist() == [id_map[s] for s in _ordered_ids(rows, SA.POP_SOURCE, "source_rank")]
    assert loaded.target_ranked_idx.tolist() == [id_map[s] for s in _ordered_ids(rows, SA.POP_TARGET_POOL, "target_rank")]
    # B_d round-trips through the loader: the allocation curve is undefined without it.
    assert loaded.benchmark_budget == 6
    assert loaded.holdout == "R1"


def test_load_requires_a_benchmark_budget(tmp_path):
    """Without B_d there is no allocation curve, so the loader fails closed rather than inventing one."""
    split, id_map = _headline_split(SAMPLE_IDS)
    _p, sha = SA.write_label_access(tmp_path, "cropharvest", 2, "R1", _rows(2))
    with pytest.raises(SA.SplitArtifactError, match="benchmark_budget"):
        SA.load_label_access(tmp_path, "cropharvest", 2, split, id_map, sha, benchmark_budget=None)


def test_load_missing_file_hard_errors(tmp_path):
    split, id_map = _headline_split(SAMPLE_IDS)
    with pytest.raises(SA.SplitArtifactError, match="missing label_access"):
        SA.load_label_access(tmp_path, "cropharvest", 2, split, id_map, "0" * 64, benchmark_budget=6)


def test_load_rejects_stale_order(tmp_path):
    split, id_map = _headline_split(SAMPLE_IDS)
    stale = SA.build_label_access_rows(seed=2, source_ids=SOURCE[:-1] + ["777"], target_pool_ids=POOL, target_test_ids=TEST)
    _p, stale_sha = SA.write_label_access(tmp_path, "cropharvest", 2, "R1", stale)
    with pytest.raises(SA.SplitArtifactError, match="does not match"):
        SA.load_label_access(tmp_path, "cropharvest", 2, split, id_map, stale_sha, benchmark_budget=6)


# --------------------------------------------------------------------------- #
# Checksum binding: a DIFFERENT valid permutation must be refused
# --------------------------------------------------------------------------- #
def test_write_returns_a_checksum_bound_to_the_bytes():
    import tempfile
    from pathlib import Path as _P

    with tempfile.TemporaryDirectory() as td:
        a_path, a_sha = SA.write_label_access(_P(td), "cropharvest", 0, "R1", _rows(0))
        assert len(a_sha) == 64
        assert a_sha == SA.sha256_bytes(a_path.read_bytes())


def test_a_different_valid_permutation_is_refused_by_the_checksum(tmp_path):
    """The whole point of the checksum: seed-1's draw is structurally VALID over the same id sets --
    same populations, same contiguous ranks in the same two columns -- so only the hash distinguishes
    it from the frozen draw. Accepting it would silently change every allocation and additive
    experiment while every structural check still passed."""
    split, id_map = _headline_split(SAMPLE_IDS)
    _p, frozen_sha = SA.write_label_access(tmp_path, "cropharvest", 2, "R1", _rows(0))
    SA.load_label_access(tmp_path, "cropharvest", 2, split, id_map, frozen_sha, benchmark_budget=6)  # frozen: fine

    # overwrite with a different seed's draw -- still structurally valid
    SA.write_label_access(tmp_path, "cropharvest", 2, "R1", _rows(1))
    SA.validate_label_access_rows(
        _rows(1), source_ids=list(SOURCE), target_pool_ids=list(POOL), target_test_ids=list(TEST),
        where="structural-check",
    )   # proves structure alone cannot catch it
    with pytest.raises(SA.SplitArtifactError, match="checksum mismatch"):
        SA.load_label_access(tmp_path, "cropharvest", 2, split, id_map, frozen_sha, benchmark_budget=6)


def test_a_missing_recorded_checksum_is_refused(tmp_path):
    """An unverifiable draw fails closed rather than loading on structure alone."""
    split, id_map = _headline_split(SAMPLE_IDS)
    SA.write_label_access(tmp_path, "cropharvest", 2, "R1", _rows(0))
    with pytest.raises(SA.SplitArtifactError, match="no label_access sha256"):
        SA.load_label_access(tmp_path, "cropharvest", 2, split, id_map, None, benchmark_budget=6)


def test_a_single_swapped_rank_is_refused(tmp_path):
    """Byte-level tamper: swap two source ranks. Structurally contiguous, cryptographically different."""
    split, id_map = _headline_split(SAMPLE_IDS)
    rows = _rows(0)
    _p, frozen_sha = SA.write_label_access(tmp_path, "cropharvest", 2, "R1", rows)
    src = [r for r in rows if r["population"] == SA.POP_SOURCE]
    src[0]["source_rank"], src[1]["source_rank"] = src[1]["source_rank"], src[0]["source_rank"]
    SA.write_label_access(tmp_path, "cropharvest", 2, "R1", rows)
    # the tampered file is still structurally valid -- only the hash catches it
    SA.validate_label_access_rows(rows, **_kw(), where="structural-check")
    with pytest.raises(SA.SplitArtifactError, match="checksum mismatch"):
        SA.load_label_access(tmp_path, "cropharvest", 2, split, id_map, frozen_sha, benchmark_budget=6)
