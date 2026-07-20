"""Frozen geographic-OOD ``label_access.csv`` artifact: label-blind generation of THREE independent
orders (matched-source, fixed-total-source-removal, target), structural validation, feasibility
gating, and runtime loading into three separately-ranked index arrays consumed by the correct
interventions. (Stage 1 -- the route/sweep wiring is tested separately once it consumes this order.)"""

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


# --- generation: deterministic, label-blind, contiguous, population-correct ------------------------

def test_generation_is_deterministic():
    assert SA._label_access_bytes(_rows(0)) == SA._label_access_bytes(_rows(0))


def test_generation_is_label_blind_and_input_order_independent():
    # build takes NO labels; the orders depend only on the id SET + seed, not input list order.
    a = _rows(0)
    b = _rows(0, source=list(reversed(SOURCE)), pool=[POOL[i] for i in (3, 0, 5, 1, 4, 2)])
    for pop, col in ((SA.POP_SOURCE, "matched_source_rank"), (SA.POP_SOURCE, "fixed_source_removal_rank"),
                     (SA.POP_TARGET_POOL, "target_rank")):
        assert _map(a, pop, col) == _map(b, pop, col)


def test_generation_differs_by_seed():
    assert _map(_rows(0), SA.POP_TARGET_POOL, "target_rank") != _map(_rows(1), SA.POP_TARGET_POOL, "target_rank")


def test_two_source_orders_are_separate_complete_and_contiguous():
    rows = _rows(0)
    matched = _map(rows, SA.POP_SOURCE, "matched_source_rank")
    fixed = _map(rows, SA.POP_SOURCE, "fixed_source_removal_rank")
    # both cover the whole source set exactly once, contiguously 0..S-1 ...
    assert set(matched) == set(SOURCE) and sorted(matched.values()) == list(range(len(SOURCE)))
    assert set(fixed) == set(SOURCE) and sorted(fixed.values()) == list(range(len(SOURCE)))
    # ... but they are DISTINCT orders (the two interventions must not share a draw).
    assert matched != fixed
    # deterministic: same seed reproduces both source orders exactly.
    r2 = _rows(0)
    assert matched == _map(r2, SA.POP_SOURCE, "matched_source_rank")
    assert fixed == _map(r2, SA.POP_SOURCE, "fixed_source_removal_rank")


def test_target_rank_contiguous_and_target_test_unranked():
    rows = _rows(0)
    tgt = _map(rows, SA.POP_TARGET_POOL, "target_rank")
    assert set(tgt) == set(POOL) and sorted(tgt.values()) == list(range(len(POOL)))
    test_rows = [r for r in rows if r["population"] == SA.POP_TARGET_TEST]
    assert {r["stable_id"] for r in test_rows} == set(TEST)
    assert all(r["matched_source_rank"] == "" and r["fixed_source_removal_rank"] == "" and r["target_rank"] == ""
               for r in test_rows)


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


def test_reject_noncontiguous_matched_source():
    with pytest.raises(SA.SplitArtifactError, match="contiguous"):
        SA.validate_label_access_rows(_tamper(SA.POP_SOURCE, "matched_source_rank", "99"), **_kw())


def test_reject_noncontiguous_fixed_removal():
    with pytest.raises(SA.SplitArtifactError, match="contiguous"):
        SA.validate_label_access_rows(_tamper(SA.POP_SOURCE, "fixed_source_removal_rank", "99"), **_kw())


def test_reject_source_carrying_target_rank():
    with pytest.raises(SA.SplitArtifactError):
        SA.validate_label_access_rows(_tamper(SA.POP_SOURCE, "target_rank", "0"), **_kw())


def test_reject_target_carrying_source_rank():
    with pytest.raises(SA.SplitArtifactError):
        SA.validate_label_access_rows(_tamper(SA.POP_TARGET_POOL, "matched_source_rank", "0"), **_kw())


def test_reject_target_test_with_rank():
    with pytest.raises(SA.SplitArtifactError, match="no rank"):
        SA.validate_label_access_rows(_tamper(SA.POP_TARGET_TEST, "target_rank", "0"), **_kw())


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


# --- feasibility: infeasible configured counts are a hard preprocessing failure -------------------

def test_feasibility_passes_at_boundary():
    # source == max(counts) is VALID: removing all k=50 source leaves an empty source contribution and
    # the fit trains on the 50 target units alone (S_{B-k}+T_k with B == k).
    SA.assert_label_access_feasible(n_source=50, n_target_pool=50)
    SA.assert_label_access_feasible(n_source=51, n_target_pool=50)


def test_feasibility_fails_small_target_pool():
    with pytest.raises(SA.SplitArtifactError, match="target label pool"):
        SA.assert_label_access_feasible(n_source=1000, n_target_pool=49)


def test_feasibility_fails_small_source_pool():
    with pytest.raises(SA.SplitArtifactError, match="source pool"):
        SA.assert_label_access_feasible(n_source=49, n_target_pool=1000)  # cannot remove 50 source units


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
    assert contract["target_label_advantage"] == (SA.ROUTE_TARGET_ONLY_FULL, SA.ROUTE_SOURCE_ONLY)
    assert contract["target_reference_deficit"] == (SA.ANCHOR_SOURCE_ID_REFERENCE, SA.ROUTE_TARGET_ONLY_FULL)
    assert contract["additive_target_label_gain"] == (SA.ROUTE_SOURCE_PLUS_TARGET, SA.ROUTE_SOURCE_ONLY)
    assert contract["full_supervision_gain"] == (SA.ROUTE_SOURCE_PLUS_TARGET_FULL, SA.ROUTE_SOURCE_ONLY)
    assert contract["size_matched_source_target_difference"] == (SA.ROUTE_MATCHED_TARGET, SA.ROUTE_MATCHED_SOURCE)
    assert contract["label_source_allocation_effect"] == (SA.ROUTE_FIXED_TOTAL_MIXED, SA.ROUTE_SOURCE_ONLY)
    assert len({n for n, _, _ in SA.LABEL_ACCESS_CONTRASTS}) == 6  # six DISTINCT questions, not merged
    assert SA.LABEL_ACCESS_COUNTS == (5, 10, 25, 50)               # one canonical count set


# --- write / read / load round-trip: three arrays consumed by the correct interventions -----------

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


def test_write_load_resolves_three_orders_to_current_indices(tmp_path):
    sample_ids = SOURCE + ["108", "109", "110", "111"] + POOL + TEST
    split, id_map = _headline_split(sample_ids)
    rows = _rows(2)
    _path, sha = SA.write_label_access(tmp_path, "cropharvest", 2, "R1", rows)
    loaded = SA.load_label_access(tmp_path, "cropharvest", 2, split, id_map, sha)

    # each array is a permutation of the right partition, mapped to CURRENT indices ...
    assert sorted(loaded.matched_source_ranked_idx.tolist()) == sorted(id_map[s] for s in SOURCE)
    assert sorted(loaded.fixed_source_removal_ranked_idx.tolist()) == sorted(id_map[s] for s in SOURCE)
    assert sorted(loaded.target_ranked_idx.tolist()) == sorted(id_map[s] for s in POOL)
    # ... consuming the CORRECT column (matched<-matched_source_rank, fixed<-fixed_source_removal_rank).
    assert loaded.matched_source_ranked_idx.tolist() == [id_map[s] for s in _ordered_ids(rows, SA.POP_SOURCE, "matched_source_rank")]
    assert loaded.fixed_source_removal_ranked_idx.tolist() == [id_map[s] for s in _ordered_ids(rows, SA.POP_SOURCE, "fixed_source_removal_rank")]
    assert loaded.target_ranked_idx.tolist() == [id_map[s] for s in _ordered_ids(rows, SA.POP_TARGET_POOL, "target_rank")]
    # the two source arrays are genuinely different orders of the same source set.
    assert loaded.matched_source_ranked_idx.tolist() != loaded.fixed_source_removal_ranked_idx.tolist()


def test_load_missing_file_hard_errors(tmp_path):
    sample_ids = SOURCE + ["108", "109", "110", "111"] + POOL + TEST
    split, id_map = _headline_split(sample_ids)
    with pytest.raises(SA.SplitArtifactError, match="missing label_access"):
        SA.load_label_access(tmp_path, "cropharvest", 2, split, id_map, "0" * 64)


def test_load_rejects_stale_order(tmp_path):
    sample_ids = SOURCE + ["108", "109", "110", "111"] + POOL + TEST
    split, id_map = _headline_split(sample_ids)
    stale = SA.build_label_access_rows(seed=2, source_ids=SOURCE[:-1] + ["777"], target_pool_ids=POOL, target_test_ids=TEST)
    _p, stale_sha = SA.write_label_access(tmp_path, "cropharvest", 2, "R1", stale)
    with pytest.raises(SA.SplitArtifactError, match="does not match"):
        SA.load_label_access(tmp_path, "cropharvest", 2, split, id_map, stale_sha)


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
    same populations, same contiguous ranks -- so only the hash distinguishes it from the frozen draw.
    Accepting it would silently change every matched-label and fixed-total experiment."""
    split, id_map = _headline_split(SOURCE + ["108", "109", "110", "111"] + POOL + TEST)
    _p, frozen_sha = SA.write_label_access(tmp_path, "cropharvest", 2, "R1", _rows(0))
    SA.load_label_access(tmp_path, "cropharvest", 2, split, id_map, frozen_sha)   # frozen draw: fine

    # overwrite with a different seed's draw -- still structurally valid
    SA.write_label_access(tmp_path, "cropharvest", 2, "R1", _rows(1))
    SA.validate_label_access_rows(
        _rows(1), source_ids=list(SOURCE), target_pool_ids=list(POOL), target_test_ids=list(TEST),
        where="structural-check",
    )   # proves structure alone cannot catch it
    with pytest.raises(SA.SplitArtifactError, match="checksum mismatch"):
        SA.load_label_access(tmp_path, "cropharvest", 2, split, id_map, frozen_sha)


def test_a_missing_recorded_checksum_is_refused(tmp_path):
    """An unverifiable draw fails closed rather than loading on structure alone."""
    split, id_map = _headline_split(SOURCE + ["108", "109", "110", "111"] + POOL + TEST)
    SA.write_label_access(tmp_path, "cropharvest", 2, "R1", _rows(0))
    with pytest.raises(SA.SplitArtifactError, match="no label_access sha256"):
        SA.load_label_access(tmp_path, "cropharvest", 2, split, id_map, None)


def test_a_single_swapped_rank_is_refused(tmp_path):
    """Byte-level tamper: swap two ranks. Structurally contiguous, cryptographically different."""
    split, id_map = _headline_split(SOURCE + ["108", "109", "110", "111"] + POOL + TEST)
    rows = _rows(0)
    _p, frozen_sha = SA.write_label_access(tmp_path, "cropharvest", 2, "R1", rows)
    src = [r for r in rows if r["population"] == SA.POP_SOURCE]
    src[0]["target_rank"], src[1]["target_rank"] = src[1]["target_rank"], src[0]["target_rank"]
    src[0]["matched_source_rank"], src[1]["matched_source_rank"] = (
        src[1]["matched_source_rank"], src[0]["matched_source_rank"])
    SA.write_label_access(tmp_path, "cropharvest", 2, "R1", rows)
    with pytest.raises(SA.SplitArtifactError, match="checksum mismatch"):
        SA.load_label_access(tmp_path, "cropharvest", 2, split, id_map, frozen_sha)
