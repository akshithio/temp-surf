"""Structural-integrity, serialization, and overwrite-safety tests for canonical split artifacts.

No hashing anywhere: identity is the path, integrity is structural, and completeness is the presence
of ``manifest.json``. These tests exercise the guarantees directly (they do not run the regimes --
that is ``test_split_parity``).
"""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from evals import split_artifacts as SA


def _spec(partitions, *, benchmark="toy", regime="random_id", label="random_id", exclusions=(), stats=None, domains=None):
    partitions = dict(partitions)
    if domains is None:
        domains = {sid: "d" for ids in partitions.values() for sid in ids}
    return SA.LeafSpec(
        benchmark=benchmark, regime=regime, seed=0, holdout_label=label,
        target_unit="sample", domain_basis="geography", group_kind="geography", has_target=False,
        params={"assembly_seed": 0}, partitions=partitions,
        partition_stats=stats or {}, domains=domains, exclusions=list(exclusions), audit=[],
    )


# --------------------------------------------------------------------------- #
# Structural validation
# --------------------------------------------------------------------------- #
def test_duplicate_id_within_partition_rejected():
    spec = _spec({"train": ["a", "a"], "val": [], "test": ["b"]})
    with pytest.raises(SA.SplitArtifactError, match="duplicate id 'a' within partition"):
        SA.validate_leaf(spec, ["a", "b"])


def test_five_partition_pairwise_disjointness_enforced():
    # source_test overlaps train -> must be caught (not just train/val/test)
    spec = _spec({"train": ["a"], "val": ["b"], "test": ["c"], "source_val": [], "source_test": ["a"]})
    with pytest.raises(SA.SplitArtifactError, match=r"partitions 'train'/'source_test' overlap"):
        SA.validate_leaf(spec, ["a", "b", "c"])


def test_complete_accounting_missing_unit_rejected():
    spec = _spec({"train": ["a"], "test": ["b"]}, exclusions=[])
    with pytest.raises(SA.SplitArtifactError, match="incomplete accounting"):
        SA.validate_leaf(spec, ["a", "b", "c"])  # c is unaccounted


def test_complete_accounting_unexpected_unit_rejected():
    spec = _spec({"train": ["a", "x"], "test": ["b"]})
    with pytest.raises(SA.SplitArtifactError, match="incomplete accounting"):
        SA.validate_leaf(spec, ["a", "b"])  # x is not eligible


def test_assignments_and_exclusions_must_be_disjoint():
    spec = _spec(
        {"train": ["a"], "test": ["b"]},
        exclusions=[{"stable_id": "a", "reason": SA.UNASSIGNED_REASON, "status": "unknown"}],
    )
    with pytest.raises(SA.SplitArtifactError, match="appear in BOTH assignments and exclusions"):
        SA.validate_leaf(spec, ["a", "b"])


def test_complete_accounting_passes_when_exclusions_fill_the_gap():
    spec = _spec(
        {"train": ["a"], "test": ["b"]},
        exclusions=[{"stable_id": "c", "reason": SA.UNASSIGNED_REASON, "status": "unknown"}],
    )
    SA.validate_leaf(spec, ["a", "b", "c"])  # no raise


# --------------------------------------------------------------------------- #
# Roundtrip, no model identity, no hashes
# --------------------------------------------------------------------------- #
def test_roundtrip_and_no_model_or_hash_fields(tmp_path):
    sample_ids = np.array([f"s{i}" for i in range(6)], dtype=object)
    domains = np.array(list("AABBAB"), dtype=object)
    labels = np.array([0, 1, 0, 1, 0, 1])
    spec, eligible = SA.build_tabular_leaf(
        "toy", "random_id", 1, label="random_id",
        train=[0, 1], val=[2], test=[3], source_val=[4], source_test=[5],
        domains=domains, labels=labels, sample_ids=sample_ids, has_target=False,
        group_kind="geography", params={"assembly_seed": 0}, audit_events=[],
    )
    ldir = SA.publish_leaf(tmp_path, spec, eligible)
    assert (ldir / "assignments.csv").exists()
    assert (ldir / "exclusions.csv").exists()
    assert (ldir / "manifest.json").exists()

    manifest = SA.read_manifest(ldir)
    assert "model" not in manifest
    blob = json.dumps(manifest).lower()
    for banned in ("sha", "hash", "digest", "fingerprint", "checksum"):
        assert banned not in blob, f"canonical manifest must not contain {banned!r}"
    assert manifest["holdout"] == "random_id"
    assert set(manifest["partitions"]) == set(SA.PARTITIONS)

    id_map = {str(s): i for i, s in enumerate(sample_ids.tolist())}
    idx = SA.load_split_indices(ldir, id_map)
    assert idx["train"].tolist() == [0, 1]
    assert idx["source_val"].tolist() == [4]
    assert idx["source_test"].tolist() == [5]


def test_unknown_id_on_load_rejected(tmp_path):
    ids = np.array(["a", "b", "c"], dtype=object)
    spec, eligible = SA.build_tabular_leaf(
        "toy", "random_id", 0, label="random_id",
        train=[0], val=[1], test=[2], source_val=[], source_test=[],
        domains=np.array(list("ABC"), dtype=object), labels=np.array([0, 1, 0]),
        sample_ids=ids, has_target=False, group_kind="geography", params={}, audit_events=[],
    )
    ldir = SA.publish_leaf(tmp_path, spec, eligible)
    # current benchmark no longer contains 'c'
    with pytest.raises(SA.SplitArtifactError, match="unknown id 'c'"):
        SA.load_split_indices(ldir, {"a": 0, "b": 1})


def test_malformed_assignments_header_rejected(tmp_path):
    ldir = tmp_path / "leaf"
    ldir.mkdir()
    (ldir / "manifest.json").write_text("{}")
    (ldir / "assignments.csv").write_text("wrong,header\nx,train\n")
    with pytest.raises(SA.SplitArtifactError, match="malformed assignments header"):
        SA.read_assignments(ldir)


# --------------------------------------------------------------------------- #
# Completeness marker & overwrite safety
# --------------------------------------------------------------------------- #
def _publish_simple(root, label, train_ids, eligible, *, regime="random_id"):
    spec = _spec(
        {"train": list(train_ids), "val": [], "test": [], "source_val": [], "source_test": []},
        regime=regime, label=label,
        exclusions=[{"stable_id": e, "reason": SA.UNASSIGNED_REASON, "status": "unknown"}
                    for e in eligible if e not in set(train_ids)],
    )
    return SA.publish_leaf(root, spec, eligible)


def test_incomplete_leaf_without_manifest_is_ignored(tmp_path):
    ldir = _publish_simple(tmp_path, "random_id", ["a", "b"], ["a", "b"])
    assert SA.is_complete(ldir)
    (ldir / "manifest.json").unlink()  # simulate crash before manifest was written
    assert not SA.is_complete(ldir)
    with pytest.raises(SA.SplitArtifactError, match="incomplete"):
        SA.load_split_indices(ldir, {"a": 0, "b": 1})


def test_overwrite_replaces_membership(tmp_path):
    ldir = _publish_simple(tmp_path, "random_id", ["a", "b"], ["a", "b", "c"])
    assert SA.read_assignments(ldir)["train"] == ["a", "b"]
    _publish_simple(tmp_path, "random_id", ["c"], ["a", "b", "c"])
    assert SA.read_assignments(ldir)["train"] == ["c"]  # fully replaced, not merged


def test_interrupted_overwrite_of_complete_leaf_reads_incomplete(tmp_path):
    """A crash mid-overwrite must leave an incomplete leaf, never new payload under the old marker."""
    ldir = _publish_simple(tmp_path, "random_id", ["a", "b"], ["a", "b"])
    assert SA.is_complete(ldir)
    # Reproduce the protocol's intermediate state: manifest removed FIRST, payload half-swapped,
    # then a crash before the new manifest is published.
    (ldir / "manifest.json").unlink()
    (ldir / "assignments.csv").write_text("stable_id,partition\nc,train\n")  # new payload, no marker
    assert not SA.is_complete(ldir)  # readers treat it as incomplete
    with pytest.raises(SA.SplitArtifactError, match="incomplete"):
        SA.load_split_indices(ldir, {"c": 2})


def test_fault_injection_interrupted_publish_of_complete_leaf(tmp_path, monkeypatch):
    """Fault-inject the REAL publish_leaf transaction: crash after the old manifest is removed and
    after one payload replacement. The leaf must end with NO canonical manifest, and load must
    refuse it. Regresses if manifest is written before payloads or the old manifest is not removed.
    """
    ldir = _publish_simple(tmp_path, "random_id", ["a", "b"], ["a", "b"])  # complete v1
    assert SA.is_complete(ldir)

    real_replace = os.replace

    def flaky_replace(src, dst):
        dstp = Path(dst)
        # fail on the IN-LOCK exclusions.csv swap: manifest already unlinked, assignments already swapped
        if dstp.name == "exclusions.csv" and dstp.parent == ldir:
            raise RuntimeError("injected crash mid-publish")
        return real_replace(src, dst)

    monkeypatch.setattr(SA.os, "replace", flaky_replace)
    spec = _spec({"train": ["c"], "val": [], "test": [], "source_val": [], "source_test": []},
                 regime="random_id", label="random_id")
    with pytest.raises(RuntimeError, match="injected crash"):
        SA.publish_leaf(tmp_path, spec, ["c"], overwrite=True)

    assert not (ldir / "manifest.json").exists(), "old manifest must have been removed and new one not written"
    assert not SA.is_complete(ldir)
    with pytest.raises(SA.SplitArtifactError, match="incomplete"):
        SA.load_split_indices(ldir, {"c": 0})


# --------------------------------------------------------------------------- #
# Runtime enforces the SAME invariants as generation over an on-disk leaf
# --------------------------------------------------------------------------- #
def _write_raw_leaf(root, *, benchmark, regime, seed, holdout, assignments, exclusions, manifest_over=None):
    ldir = SA.leaf_dir(root, benchmark, regime, seed, holdout)
    ldir.mkdir(parents=True, exist_ok=True)
    with open(ldir / "assignments.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["stable_id", "partition", "domain"])
        for row in assignments:
            sid, part = row[0], row[1]
            w.writerow([sid, part, row[2] if len(row) > 2 else "d"])
    with open(ldir / "exclusions.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["stable_id", "reason", "status"])
        for row in exclusions:
            w.writerow(row)
    manifest = {
        "schema_version": SA.SCHEMA_VERSION, "benchmark": benchmark, "regime": regime, "seed": seed,
        "holdout": holdout, "holdout_dirname": SA.holdout_dirname(holdout), "target_unit": "sample",
        "domain_basis": "geography", "group_kind": "geography", "has_target": False, "params": {},
        "partitions": {}, "n_excluded": len(exclusions), "exclusion_reason_counts": {}, "audit": [],
    }
    if manifest_over:
        manifest.update(manifest_over)
    (ldir / "manifest.json").write_text(json.dumps(manifest))
    return ldir


def test_load_rejects_partition_overlap_on_disk(tmp_path):
    ldir = _write_raw_leaf(tmp_path, benchmark="toy", regime="random_id", seed=0, holdout="random_id",
                           assignments=[("a", "train"), ("a", "test")], exclusions=[])
    with pytest.raises(SA.SplitArtifactError, match="overlap"):
        SA.load_split_indices(ldir, {"a": 0})


def test_load_rejects_incomplete_accounting_on_disk(tmp_path):
    ldir = _write_raw_leaf(tmp_path, benchmark="toy", regime="random_id", seed=0, holdout="random_id",
                           assignments=[("a", "train")], exclusions=[])
    with pytest.raises(SA.SplitArtifactError, match="incomplete accounting"):
        SA.load_split_indices(ldir, {"a": 0, "b": 1})  # b unaccounted in the CURRENT population


def test_load_rejects_unknown_excluded_id_on_disk(tmp_path):
    ldir = _write_raw_leaf(tmp_path, benchmark="toy", regime="random_id", seed=0, holdout="random_id",
                           assignments=[("a", "train")],
                           exclusions=[("zzz", SA.UNASSIGNED_REASON, "unknown")])
    with pytest.raises(SA.SplitArtifactError, match="unknown excluded id"):
        SA.load_split_indices(ldir, {"a": 0})


def test_load_rejects_manifest_path_disagreement(tmp_path):
    ldir = _write_raw_leaf(tmp_path, benchmark="toy", regime="random_id", seed=0, holdout="random_id",
                           assignments=[("a", "train")], exclusions=[],
                           manifest_over={"benchmark": "other-benchmark"})
    with pytest.raises(SA.SplitArtifactError, match="disagrees with canonical path"):
        SA.load_split_indices(ldir, {"a": 0})


def test_load_rejects_requested_identity_mismatch(tmp_path):
    ldir = _write_raw_leaf(tmp_path, benchmark="toy", regime="random_id", seed=0, holdout="random_id",
                           assignments=[("a", "train")], exclusions=[])
    with pytest.raises(SA.SplitArtifactError, match="requested holdout"):
        SA.load_split_indices(ldir, {"a": 0}, holdout="a-different-holdout")


def test_load_accepts_a_wellformed_on_disk_leaf(tmp_path):
    ldir = _write_raw_leaf(tmp_path, benchmark="toy", regime="random_id", seed=0, holdout="random_id",
                           assignments=[("a", "train"), ("b", "test")],
                           exclusions=[("c", SA.UNASSIGNED_REASON, "unknown")])
    idx = SA.load_split_indices(ldir, {"a": 0, "b": 1, "c": 2},
                                benchmark="toy", regime="random_id", seed=0, holdout="random_id")
    assert idx["train"].tolist() == [0] and idx["test"].tolist() == [1]


# --------------------------------------------------------------------------- #
# Schema-version parser contract (one schema, no historical-version fallbacks) + blank domains
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("ver,match", [
    ("1", "must be an integer"),   # a string, not an int
    (1.0, "must be an integer"),   # a float, not an int
    (True, "must be an integer"),  # bool is an int subclass -- must still be rejected
    (0, "unsupported schema_version 0"),
    (2, "unsupported schema_version 2"),
])
def test_leaf_manifest_rejects_a_bad_schema_version(tmp_path, ver, match):
    ldir = _write_raw_leaf(tmp_path, benchmark="toy", regime="random_id", seed=0, holdout="random_id",
                           assignments=[("a", "train")], exclusions=[],
                           manifest_over={"schema_version": ver})
    with pytest.raises(SA.SplitArtifactError, match=match):
        SA.read_manifest(ldir)


def test_leaf_manifest_rejects_a_missing_schema_version(tmp_path):
    ldir = _write_raw_leaf(tmp_path, benchmark="toy", regime="random_id", seed=0, holdout="random_id",
                           assignments=[("a", "train")], exclusions=[])
    m = json.loads((ldir / "manifest.json").read_text())
    del m["schema_version"]
    (ldir / "manifest.json").write_text(json.dumps(m))
    with pytest.raises(SA.SplitArtifactError, match="missing schema_version"):
        SA.read_manifest(ldir)


def test_generation_json_schema_version_is_enforced(tmp_path):
    root = tmp_path / "splits"
    _publish_simple(root, "random_id", ["a", "b"], ["a", "b"])  # a real, complete leaf under toy/random_id/0
    d = SA.regime_seed_dir(root, "toy", "random_id", 0)
    good = {"schema_version": SA.SCHEMA_VERSION, "benchmark": "toy", "regime": "random_id", "seed": 0,
            "requested_holdouts": ["random_id"], "yielded_holdouts": ["random_id"],
            "dropped_holdouts": [], "audit_events": [], "regime_problems": []}
    (d / "generation.json").write_text(json.dumps(good))
    assert SA.list_leaves(root, "toy", "random_id", 0) == ["random_id"]  # baseline: a valid version passes

    (d / "generation.json").write_text(json.dumps({**good, "schema_version": 2}))
    with pytest.raises(SA.SplitArtifactError, match="unsupported schema_version 2"):
        SA.list_leaves(root, "toy", "random_id", 0)

    (d / "generation.json").write_text(json.dumps({k: v for k, v in good.items() if k != "schema_version"}))
    with pytest.raises(SA.SplitArtifactError, match="missing schema_version"):
        SA.list_leaves(root, "toy", "random_id", 0)


@pytest.mark.parametrize("domain", ["", "   ", "\t"])
def test_blank_or_whitespace_domain_in_assignments_is_rejected(tmp_path, domain):
    ldir = SA.leaf_dir(tmp_path, "toy", "random_id", 0, "random_id")
    ldir.mkdir(parents=True, exist_ok=True)
    with open(ldir / "assignments.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["stable_id", "partition", "domain"])
        w.writerow(["a", "train", "kenya"])
        w.writerow(["b", "test", domain])
    # both readers funnel through the same row parser, so both must reject it
    with pytest.raises(SA.SplitArtifactError, match="blank/whitespace-only domain for id 'b'"):
        SA.read_assignments(ldir)
    with pytest.raises(SA.SplitArtifactError, match="blank/whitespace-only domain for id 'b'"):
        SA.read_domains(ldir)


# --------------------------------------------------------------------------- #
# Filesystem-safe holdout names (exact label preserved)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("label", ["kenya", "lem-brazil", "latvia_vs_estonia", "fold_5", "a/b c", "d:e", "%weird"])
def test_holdout_dirname_roundtrips_and_is_fs_safe(label):
    d = SA.holdout_dirname(label)
    assert "/" not in d and " " not in d
    assert SA.holdout_label(d) == label


def test_slashy_label_dir_is_encoded_but_manifest_keeps_exact_label(tmp_path):
    label = "region/with space"
    spec = _spec(
        {"train": ["a"], "val": [], "test": [], "source_val": [], "source_test": []},
        regime="geographic_ood", label=label,
        exclusions=[],
    )
    ldir = SA.publish_leaf(tmp_path, spec, ["a"])
    assert "/" not in ldir.name and " " not in ldir.name
    assert SA.read_manifest(ldir)["holdout"] == label
    assert SA.read_manifest(ldir)["holdout_dirname"] == ldir.name


# --------------------------------------------------------------------------- #
# Exclusion reasons/status: proven when available, else unassigned/unknown; never guessed
# --------------------------------------------------------------------------- #
def test_exclusion_reason_status_from_proof_only():
    sample_ids = np.array(["p0", "p1", "u0", "x0"], dtype=object)
    # p0 purged (proof: audit event index 0); u0 has unknown domain (proof: domains); x0 just unassigned
    domains = np.array(["A", "A", "unknown", "A"], dtype=object)
    labels = np.array([0, 1, 0, 0])
    audit = [{"kind": "purge", "purged_train_indices": [0]}]
    spec, eligible = SA.build_tabular_leaf(
        "toy", "geographic_ood", 0, label="A",
        train=[], val=[1], test=[], source_val=[], source_test=[],
        domains=domains, labels=labels, sample_ids=sample_ids, has_target=True,
        group_kind="geography", params={}, audit_events=audit,
    )
    by_id = {e["stable_id"]: (e["reason"], e["status"]) for e in spec.exclusions}
    assert by_id["p0"] == ("purged_near_ood", "proven")
    assert by_id["u0"] == ("unknown_domain", "proven")
    assert by_id["x0"] == (SA.UNASSIGNED_REASON, "unknown")
    # p1 is assigned (val), so it must NOT appear in exclusions
    assert "p1" not in by_id


# --------------------------------------------------------------------------- #
# Dense (PASTIS) leaves are patch-level; stable ids are patch ids
# --------------------------------------------------------------------------- #
def test_dense_leaf_is_patch_level(tmp_path):
    from evals.regimes.base import DenseSplit

    cfg = DenseSplit(
        label="fold_5", train_folds={1, 2, 3}, val_folds={4}, test_folds={5},
        train_patches={10, 11}, val_patches={20}, test_patches={30},
        source_val_patches=set(), source_test_patches=set(), has_target=True, group_kind="geography",
    )
    dense_cache = dict(
        all_patch_ids=[10, 11, 20, 30, 99],  # 99 has no coords -> proven no_coords exclusion
        fold_of={10: 1, 11: 2, 20: 4, 30: 5, 99: 3},
        class_sets={10: {0, 1}, 11: {1}, 20: {0}, 30: {0, 2}, 99: {0}},
        patch_latlon={10: (1.0, 2.0), 11: (1.1, 2.0), 20: (3.0, 4.0), 30: (5.0, 6.0), 99: (np.nan, np.nan)},
    )
    spec, eligible = SA.build_dense_leaf(
        "pastis", "official", 0, cfg=cfg, bench=SimpleNamespace(), params={"assembly_seed": 0},
        audit_events=[], **dense_cache,
    )
    ldir = SA.publish_leaf(tmp_path, spec, eligible)
    assigns = SA.read_assignments(ldir)
    assert assigns["train"] == ["10", "11"] and assigns["test"] == ["30"]
    # patch 99 excluded, proven no_coords
    excl = {e["stable_id"]: (e["reason"], e["status"]) for e in SA.read_exclusions(ldir)}
    assert excl["99"] == ("no_coords", "proven")
    assert SA.read_manifest(ldir)["target_unit"] == "patch"
