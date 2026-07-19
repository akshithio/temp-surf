"""Structural-integrity, serialization, and runtime-round-trip tests for the frozen split artifacts.

The format is one ``assignments.csv`` per leaf (stable_id, partition, status, domain, reason) -- plus,
for geographic_ood headline targets, a sibling ``label_access.csv`` -- and one central
``data/logs/splits.json`` (provenance + per-leaf summary + SHA-256). These tests exercise the
scientific invariants directly (they do not run the regimes -- that is ``test_split_parity`` and the
per-regime files): partition disjointness, complete accounting, the route-capability contract, the
central-log checksum gate, PASTIS patch-level dense leaves + structural fold/tile checks, and the
generated-to-runtime round-trip.
"""

from __future__ import annotations

import csv

import numpy as np
import pytest

from evals import split_artifacts as SA
from evals.regimes.base import DenseSourceTargetSplit, SourceTargetSplit

_PROV = {"generation_timestamp": "t", "code_revision": "x", "run_seeds": [0], "cluster_seed": 0}


def _root(tmp_path):
    return tmp_path / "splits"


def _write_log(root, entries):
    SA.write_splits_log(SA.default_log_path(root), provenance=_PROV, entries=list(entries))


def _publish_tabular(root, benchmark, regime, seed, *, split, domains, labels, sample_ids, audit_events=(), purge_km=0.0):
    rows, entry = SA.build_tabular_leaf(
        benchmark, regime, seed, split=split, domains=domains, labels=labels,
        sample_ids=sample_ids, audit_events=list(audit_events), purge_km=purge_km,
    )
    _p, entry["sha256"] = SA.write_assignments(root, benchmark, regime, seed, split.label, rows)
    return entry


def _publish_dense(root, benchmark, regime, seed, *, dense_split, cache, audit_events=(), purge_km=0.0):
    rows, entry = SA.build_dense_leaf(
        benchmark, regime, seed, dense_split=dense_split, audit_events=list(audit_events), purge_km=purge_km, **cache,
    )
    _p, entry["sha256"] = SA.write_assignments(root, benchmark, regime, seed, dense_split.label, rows)
    return entry


def _write_raw_csv(root, benchmark, regime, seed, holdout, rows):
    """Write a raw assignments.csv (rows are 5-tuples) and return (entry-stub, sha256)."""
    path = SA.assignments_path(root, benchmark, regime, seed, holdout)
    path.parent.mkdir(parents=True, exist_ok=True)
    import io
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(SA.CSV_HEADER)
    for r in rows:
        w.writerow(r)
    data = buf.getvalue().encode()
    path.write_bytes(data)
    return SA.sha256_bytes(data)


def _entry(benchmark, regime, seed, holdout, sha, *, has_target=False, supports_target_labels=False,
           target_role="headline", group_kind="geography", target_unit="sample"):
    return {
        "benchmark": benchmark, "regime": regime, "seed": seed, "holdout": holdout,
        "target_unit": target_unit, "group_kind": group_kind, "has_target": has_target,
        "supports_target_labels": supports_target_labels, "target_role": target_role,
        "assignments_csv": SA.leaf_rel_path(benchmark, regime, seed, holdout), "sha256": sha,
    }


# --------------------------------------------------------------------------- #
# validate_rows: the generation-time invariant core
# --------------------------------------------------------------------------- #
def test_duplicate_stable_id_rejected():
    rows = [
        SA._row("a", "source_train", SA.STATUS_ASSIGNED, "d", ""),
        SA._row("a", "source_test", SA.STATUS_ASSIGNED, "d", ""),
    ]
    with pytest.raises(SA.SplitArtifactError, match="duplicate stable_id"):
        SA.validate_rows(rows, has_target=False, supports_target_labels=False)


def test_assigned_row_needs_a_non_blank_domain():
    rows = [SA._row("a", "source_train", SA.STATUS_ASSIGNED, "", "")]
    with pytest.raises(SA.SplitArtifactError, match="blank domain"):
        SA.validate_rows(rows, has_target=False, supports_target_labels=False)


@pytest.mark.parametrize("parts,has_t,supp,match", [
    ([("a", "source_train"), ("b", "target_label_pool")], True, False, "empty target_label_pool"),
    ([("a", "source_train"), ("b", "target_test")], True, True, "non-empty target_label_pool"),
    ([("a", "source_train"), ("b", "target_label_pool")], True, True, "non-empty target_test"),
    ([("a", "target_test")], False, True, "has_target=True"),
    ([("a", "target_test")], False, False, "both target partitions empty"),
])
def test_route_capability_contract_enforced(parts, has_t, supp, match):
    rows = [SA._row(sid, part, SA.STATUS_ASSIGNED, "d", "") for sid, part in parts]
    with pytest.raises(SA.SplitArtifactError, match=match):
        SA.validate_rows(rows, has_target=has_t, supports_target_labels=supp)


# --------------------------------------------------------------------------- #
# Row building: statuses + reasons are honest (assigned / purged / excluded)
# --------------------------------------------------------------------------- #
def test_build_tags_assigned_purged_and_excluded_rows():
    sample_ids = np.array(["p0", "p1", "u0", "x0", "t0", "t1"], dtype=object)
    domains = np.array(["A", "A", "unknown", "A", "K", "K"], dtype=object)
    labels = np.array([0, 1, 0, 0, 0, 1])
    audit = [{"kind": "purge", "purged_indices": [0]}]  # p0 purged
    split = SourceTargetSplit(
        label="K", source_train=np.array([], dtype=np.int64), source_val=np.array([1]),
        source_test=np.array([], dtype=np.int64),
        target_label_pool=np.array([4]), target_test=np.array([5]),
        has_target=True, supports_target_labels=True,
    )
    rows, summary = SA.build_tabular_leaf(
        "toy", "geographic_ood", 0, split=split, domains=domains, labels=labels,
        sample_ids=sample_ids, audit_events=audit, purge_km=50.0,
    )
    by_id = {r["stable_id"]: (r["status"], r["reason"]) for r in rows}
    assert by_id["p0"] == (SA.STATUS_PURGED, "purged_near_ood")
    assert by_id["u0"] == (SA.STATUS_EXCLUDED, "unknown_domain")
    assert by_id["x0"] == (SA.STATUS_EXCLUDED, "unassigned")
    assert by_id["p1"][0] == SA.STATUS_ASSIGNED
    # the log summary carries counts (not stable ids) + purge distance
    assert summary["status_counts"] == {"assigned": 3, "purged": 1, "excluded": 2}
    assert summary["purge_km"] == 50.0 and summary["purge_count"] == 1
    assert summary["validation"] == "passed"


def test_leaf_summary_has_per_partition_stratification():
    sample_ids = np.array([f"s{i}" for i in range(6)], dtype=object)
    domains = np.array(list("SSSSKK"), dtype=object)  # source vs kenya target
    labels = np.array([0, 1, 0, 1, 0, 1])
    _rows, summary = SA.build_tabular_leaf(
        "toy", "geographic_ood", 0, split=_geo_split(), domains=domains, labels=labels,
        sample_ids=sample_ids, audit_events=[],
    )
    # the GLOBAL class/domain/partition counts are gone; only per-partition stratification remains
    assert "class_counts" not in summary and "domain_counts" not in summary and "partition_counts" not in summary
    ps = summary["partition_stats"]
    assert set(ps) == set(SA.PARTITIONS)
    assert all({"n", "class_counts", "domain_counts"} == set(ps[p]) for p in SA.PARTITIONS)
    # source_train = rows 0,1 (domain S, classes 0/1); target_test = row 5 (domain K, class 1)
    assert ps["source_train"] == {"n": 2, "class_counts": {"0": 1, "1": 1}, "domain_counts": {"S": 2}}
    assert ps["target_test"] == {"n": 1, "class_counts": {"1": 1}, "domain_counts": {"K": 1}}


def test_dense_partition_stats_are_patch_level_class_presence():
    _rows, summary = SA.build_dense_leaf(
        "pastis", "official", 0, dense_split=_dense_official_split(), audit_events=[], **_dense_cache(),
    )
    ps = summary["partition_stats"]
    # patch 10 has classes {0,1}, patch 11 has {1}: source_train class PRESENCE is 0:1, 1:2 (not pixels)
    assert ps["source_train"] == {"n": 2, "class_counts": {"0": 1, "1": 2}, "domain_counts": {"1": 1, "2": 1}}
    # patch 30 (target_test) has classes {0, 2}
    assert ps["target_test"]["class_counts"] == {"0": 1, "2": 1}


# --------------------------------------------------------------------------- #
# Tabular round-trip (generation -> frozen CSV + central log -> runtime load)
# --------------------------------------------------------------------------- #
def _geo_split(label="kenya"):
    return SourceTargetSplit(
        label=label, source_train=np.array([0, 1]), source_val=np.array([2]), source_test=np.array([3]),
        target_label_pool=np.array([4]), target_test=np.array([5]),
        has_target=True, supports_target_labels=True, group_kind="geography",
    )


def test_tabular_round_trip_and_no_stable_ids_in_log(tmp_path):
    root = _root(tmp_path)
    sample_ids = np.array([f"s{i}" for i in range(6)], dtype=object)
    domains = np.array(list("SSSSKK"), dtype=object)
    labels = np.array([0, 1, 0, 1, 0, 1])
    entry = _publish_tabular(root, "toy", "geographic_ood", 1, split=_geo_split(),
                             domains=domains, labels=labels, sample_ids=sample_ids)
    _write_log(root, [entry])

    # the log entry carries counts + capability + checksum, never stable ids
    blob = str(entry)
    assert "s0" not in blob and "sha256" in entry and len(entry["sha256"]) == 64
    assert entry["has_target"] is True and entry["supports_target_labels"] is True

    loaded = SA.load_tabular_splits(root, "toy", sample_ids, ["geographic_ood"], [1])
    assert len(loaded) == 1
    ls = loaded[0]
    assert ls.split.source_train.tolist() == [0, 1]
    assert ls.split.target_label_pool.tolist() == [4] and ls.split.target_test.tolist() == [5]
    # per-sample domain array is reconstructed from the CSV (worst-group scoring)
    assert ls.domains[4] == "K" and ls.domains[0] == "S"


def test_load_checksum_mismatch_is_refused(tmp_path):
    root = _root(tmp_path)
    sample_ids = np.array([f"s{i}" for i in range(6)], dtype=object)
    entry = _publish_tabular(root, "toy", "geographic_ood", 0, split=_geo_split(),
                             domains=np.array(list("SSSSKK"), dtype=object),
                             labels=np.array([0, 1, 0, 1, 0, 1]), sample_ids=sample_ids)
    _write_log(root, [entry])
    # tamper the frozen CSV after the log recorded its checksum
    csv_path = SA.assignments_path(root, "toy", "geographic_ood", 0, "kenya")
    csv_path.write_bytes(csv_path.read_bytes() + b"s6,source_train,assigned,S,\n")
    with pytest.raises(SA.SplitArtifactError, match="checksum mismatch"):
        SA.load_tabular_splits(root, "toy", sample_ids, ["geographic_ood"], [0])


def test_load_rejects_incomplete_accounting_against_current_population(tmp_path):
    root = _root(tmp_path)
    sha = _write_raw_csv(root, "toy", "random_id", 0, "random_id",
                         [("a", "source_train", "assigned", "d", "")])
    _write_log(root, [_entry("toy", "random_id", 0, "random_id", sha)])
    # current benchmark has {a, b}; the CSV only accounts for a
    with pytest.raises(SA.SplitArtifactError, match="does not account for the current population"):
        SA.load_tabular_splits(root, "toy", np.array(["a", "b"], dtype=object), ["random_id"], [0])


def test_load_rejects_unexpected_id_against_current_population(tmp_path):
    root = _root(tmp_path)
    sha = _write_raw_csv(root, "toy", "random_id", 0, "random_id",
                         [("a", "source_train", "assigned", "d", ""), ("zzz", "source_test", "assigned", "d", "")])
    _write_log(root, [_entry("toy", "random_id", 0, "random_id", sha)])
    with pytest.raises(SA.SplitArtifactError, match="does not account for the current population"):
        SA.load_tabular_splits(root, "toy", np.array(["a", "zzz"], dtype=object)[:1], ["random_id"], [0])


def test_load_rejects_route_invariant_violation_on_disk(tmp_path):
    root = _root(tmp_path)
    # supports_target_labels=True but no target_label_pool rows -> refuse (route invariant at load)
    sha = _write_raw_csv(root, "toy", "geographic_ood", 0, "kenya",
                         [("a", "source_train", "assigned", "k", ""), ("b", "target_test", "assigned", "k", "")])
    _write_log(root, [_entry("toy", "geographic_ood", 0, "kenya", sha, has_target=True, supports_target_labels=True)])
    with pytest.raises(SA.SplitArtifactError, match="non-empty target_label_pool"):
        SA.load_tabular_splits(root, "toy", np.array(["a", "b"], dtype=object), ["geographic_ood"], [0])


def test_load_refuses_a_requested_regime_with_zero_leaves(tmp_path):
    root = _root(tmp_path)
    sha = _write_raw_csv(root, "toy", "random_id", 0, "random_id", [("a", "source_train", "assigned", "d", "")])
    _write_log(root, [_entry("toy", "random_id", 0, "random_id", sha)])
    with pytest.raises(SA.SplitArtifactError, match="zero leaves"):
        SA.load_tabular_splits(root, "toy", np.array(["a"], dtype=object), ["official"], [0])


def test_missing_log_is_refused(tmp_path):
    with pytest.raises(SA.SplitArtifactError, match="no split log"):
        SA.load_tabular_splits(_root(tmp_path), "toy", np.array(["a"], dtype=object), ["random_id"], [0])


# --------------------------------------------------------------------------- #
# Dense (PASTIS) leaves are patch-level; official/geographic structural checks
# --------------------------------------------------------------------------- #
def _dense_official_split():
    return DenseSourceTargetSplit(
        label="fold_5",
        source_train_patches=frozenset({10, 11}), source_val_patches=frozenset({20}),
        source_test_patches=frozenset({12}),
        target_label_pool_patches=frozenset(), target_test_patches=frozenset({30}),
        has_target=True, supports_target_labels=False, group_kind="geography",
    )


def _dense_cache():
    return dict(
        all_patch_ids=[10, 11, 12, 20, 30, 99],  # 99 has no coords -> no_coords exclusion
        domain_of={10: "1", 11: "2", 12: "3", 20: "4", 30: "5", 99: "3"},  # fold as domain
        class_sets={10: {0, 1}, 11: {1}, 12: {0}, 20: {0}, 30: {0, 2}, 99: {0}},
        patch_latlon={10: (1.0, 2.0), 11: (1.1, 2.0), 12: (1.2, 2.0), 20: (3.0, 4.0),
                      30: (5.0, 6.0), 99: (np.nan, np.nan)},
    )


def test_dense_leaf_is_patch_level_with_no_coords_exclusion():
    rows, summary = SA.build_dense_leaf(
        "pastis", "official", 0, dense_split=_dense_official_split(), audit_events=[], **_dense_cache(),
    )
    by_id = {r["stable_id"]: (r["partition"], r["status"], r["reason"]) for r in rows}
    assert by_id["10"] == ("source_train", "assigned", "")
    assert by_id["30"] == ("target_test", "assigned", "")
    assert by_id["99"] == ("", "excluded", "no_coords")
    assert summary["target_unit"] == "patch" and summary["supports_target_labels"] is False


def _dense_patch_fold():
    return {10: 1, 11: 2, 12: 3, 20: 4, 30: 5, 99: 3}


def test_dense_official_load_rejects_changed_fold(tmp_path):
    root = _root(tmp_path)
    entry = _publish_dense(root, "pastis", "official", 0, dense_split=_dense_official_split(), cache=_dense_cache())
    _write_log(root, [entry])
    patch_fold = _dense_patch_fold()
    patch_tile = {p: "T31TFM" for p in patch_fold}
    by_seed = SA.load_dense_splits(root, "pastis", patch_fold, patch_tile, ["official"], [0])
    assert by_seed[0][0].split.target_test_patches == frozenset({30})
    patch_fold[30] = 2  # target_test patch's published fold shifts 5 -> 2
    with pytest.raises(SA.SplitArtifactError, match="published fold"):
        SA.load_dense_splits(root, "pastis", patch_fold, patch_tile, ["official"], [0])


def test_dense_geographic_load_rejects_changed_tile(tmp_path):
    root = _root(tmp_path)
    split = DenseSourceTargetSplit(
        label="T31TFM",
        source_train_patches=frozenset({10, 11}), source_val_patches=frozenset({12}),
        source_test_patches=frozenset({20}),
        target_label_pool_patches=frozenset({30}), target_test_patches=frozenset({40}),
        has_target=True, supports_target_labels=True, group_kind="geography",
    )
    cache = dict(
        all_patch_ids=[10, 11, 12, 20, 30, 40],
        domain_of={10: "T30UXV", 11: "T30UXV", 12: "T31TFJ", 20: "T32ULU", 30: "T31TFM", 40: "T31TFM"},
        class_sets={p: {0, 1} for p in [10, 11, 12, 20, 30, 40]},
        patch_latlon={p: (1.0, 2.0) for p in [10, 11, 12, 20, 30, 40]},
    )
    entry = _publish_dense(root, "pastis", "geographic_ood", 0, dense_split=split, cache=cache)
    _write_log(root, [entry])
    patch_fold = {p: 1 for p in cache["all_patch_ids"]}
    patch_tile = dict(cache["domain_of"])
    SA.load_dense_splits(root, "pastis", patch_fold, patch_tile, ["geographic_ood"], [0])  # OK
    patch_tile[40] = "T30UXV"  # a target patch's tile changed
    with pytest.raises(SA.SplitArtifactError, match="Sentinel tile"):
        SA.load_dense_splits(root, "pastis", patch_fold, patch_tile, ["geographic_ood"], [0])


def test_dense_spatial_load_does_not_recompute_clusters(tmp_path, monkeypatch):
    """Spatial cells are frozen in the CSV; loading must consume them without re-clustering."""
    root = _root(tmp_path)
    split = DenseSourceTargetSplit(
        label="cluster_00",
        source_train_patches=frozenset({10, 11}), source_val_patches=frozenset({12}),
        source_test_patches=frozenset({20}),
        target_label_pool_patches=frozenset({30}), target_test_patches=frozenset({40}),
        has_target=True, supports_target_labels=True, group_kind="spatial_cluster",
    )
    cache = dict(
        all_patch_ids=[10, 11, 12, 20, 30, 40],
        domain_of={p: "cluster_00" for p in [10, 11, 12, 20, 30, 40]},
        class_sets={p: {0, 1} for p in [10, 11, 12, 20, 30, 40]},
        patch_latlon={p: (1.0, 2.0) for p in [10, 11, 12, 20, 30, 40]},
    )
    entry = _publish_dense(root, "pastis", "spatial_cluster_ood", 0, dense_split=split, cache=cache)
    _write_log(root, [entry])
    import evals.regimes.spatial_cluster_ood as sc
    monkeypatch.setattr(sc, "_cell_labels", lambda *a, **k: (_ for _ in ()).throw(
        RuntimeError("must not recompute clusters at load")))
    patch_fold = {p: 1 for p in cache["all_patch_ids"]}
    by_seed = SA.load_dense_splits(root, "pastis", patch_fold, dict(cache["domain_of"]), ["spatial_cluster_ood"], [0])
    assert by_seed[0][0].split.target_test_patches == frozenset({40})
