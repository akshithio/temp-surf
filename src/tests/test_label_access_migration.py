"""The narrow label-access migration: derives label_access.csv + splits.json metadata from FROZEN
assignments, and can never write an assignments.csv.

The central guarantee here is negative. Once the assignment splits are canonical, "regenerate the
derived artifacts" must not be able to turn into "re-derive split membership" -- not even into a
byte-identical rewrite, because a rewrite re-derives frozen scientific inputs from live code and data
and would only stay correct by coincidence. So the headline test makes the assignments writer raise on
contact and proves the migration still completes.
"""

from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path

import pytest

from evals import split_artifacts as SA

_REPO = Path(__file__).resolve().parents[2]


def _load_migration():
    spec = importlib.util.spec_from_file_location(
        "migrate_label_access", _REPO / "tools" / "migrate_label_access.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _write_assignments_csv(path: Path, src, pool, test) -> None:
    """Write a leaf's assignments.csv DIRECTLY, so building the fixture never touches the writer the
    headline test is about to disable."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(SA.CSV_HEADER)
        for part, ids in (("source_train", src), ("target_label_pool", pool), ("target_test", test)):
            for sid in ids:
                w.writerow([sid, part, SA.STATUS_ASSIGNED, "dom", ""])
        # a purged row that must NEVER re-enter a population through a derived artifact
        w.writerow(["purged_0", "source_train", SA.STATUS_PURGED, "dom", SA.REASON_PURGED_NEAR_OOD])


def _tree(tmp_path: Path, regions=("kenya", "togo", "sudan"), seeds=(0, 1, 2), n_source=200, n_pool=120):
    """A frozen splits tree: one assignments.csv per (region, seed) plus the canonical log."""
    root = tmp_path / "splits"
    entries, classes = [], {}
    for region in regions:
        for seed in seeds:
            src = [f"{region}_s{i}" for i in range(n_source)]
            pool = [f"{region}_p{i}" for i in range(n_pool)]
            test = [f"{region}_t{i}" for i in range(8)]
            rel = f"cropharvest/geographic_ood/{seed}/{region}/assignments.csv"
            _write_assignments_csv(root / rel, src, pool, test)
            entries.append({
                "benchmark": "cropharvest", "regime": "geographic_ood", "seed": seed,
                "holdout": region, "assignments_csv": rel, "sha256": "x" * 64,
                "supports_target_labels": True, "target_role": "headline",
            })
            for i, sid in enumerate([*src, *pool]):
                classes[sid] = frozenset({str(i % 2)})
    logs = tmp_path / "logs" / "splits.json"
    logs.parent.mkdir(parents=True, exist_ok=True)
    logs.write_text(json.dumps({
        "code_revision": "abc123oldassignments", "generation_timestamp": "2026-01-01T00:00:00+00:00",
        "run_seeds": [0, 1, 2], "inputs": {"data_fingerprint": "orig"}, "split_config": {"frozen": True},
        "leaves": entries,
    }, indent=2))
    return root, logs, classes


def _configure(mod, monkeypatch, root, logs, classes, *, dry_run=False):
    monkeypatch.setattr(mod, "SPLITS_ROOT", root)
    monkeypatch.setattr(mod, "LOGS_PATH", logs)
    monkeypatch.setattr(mod, "BENCHMARKS", ["cropharvest"])
    monkeypatch.setattr(mod, "DRY_RUN", dry_run)
    monkeypatch.setattr(mod, "_class_map", lambda _b: classes)
    monkeypatch.setattr(mod, "_tree_is_dirty", lambda: "")


# --------------------------------------------------------------------------- #
# The guarantee
# --------------------------------------------------------------------------- #
def test_migration_completes_with_the_assignments_writer_disabled(tmp_path, monkeypatch):
    """THE regression test: make write_assignments raise on contact, and the migration still finishes.

    This is what makes "never touches split membership" a property of the code rather than something a
    human has to verify against a hash afterwards."""
    mod = _load_migration()
    root, logs, classes = _tree(tmp_path)

    def _boom(*a, **k):
        raise AssertionError("write_assignments must never be reachable from the label-access migration")

    monkeypatch.setattr(SA, "write_assignments", _boom)
    _configure(mod, monkeypatch, root, logs, classes)
    assert mod.main() == 0
    assert len(list(root.rglob("label_access.csv"))) == 9


def test_assignments_bytes_are_untouched(tmp_path, monkeypatch):
    """Belt to the writer-disabled brace: every assignments.csv is byte-identical afterwards."""
    mod = _load_migration()
    root, logs, classes = _tree(tmp_path)
    before = {p: p.read_bytes() for p in sorted(root.rglob("assignments.csv"))}
    _configure(mod, monkeypatch, root, logs, classes)
    assert mod.main() == 0
    after = {p: p.read_bytes() for p in sorted(root.rglob("assignments.csv"))}
    assert before == after and len(before) == 9


def test_purged_rows_never_re_enter_a_population(tmp_path, monkeypatch):
    """A purged id was deliberately removed from the experiment; a derived artifact must not
    resurrect it. Only ``status == assigned`` rows are populations."""
    mod = _load_migration()
    root, logs, classes = _tree(tmp_path)
    _configure(mod, monkeypatch, root, logs, classes)
    entry = json.loads(logs.read_text())["leaves"][0]
    pops = mod._populations(root, entry)
    assert "purged_0" not in pops["source_train"]
    assert len(pops["source_train"]) == 200


# --------------------------------------------------------------------------- #
# Provenance
# --------------------------------------------------------------------------- #
def test_original_assignment_provenance_is_preserved_and_label_access_recorded_separately(
    tmp_path, monkeypatch
):
    """The old assignments were NOT generated by this revision, and the log must not claim they were.
    The original provenance survives verbatim; the new code revision lands in its own block."""
    mod = _load_migration()
    root, logs, classes = _tree(tmp_path)
    _configure(mod, monkeypatch, root, logs, classes)
    monkeypatch.setattr(mod, "_code_revision", lambda: "newrev999")
    assert mod.main() == 0

    out = json.loads(logs.read_text())
    assert out["code_revision"] == "abc123oldassignments", "assignment provenance was overwritten"
    assert out["generation_timestamp"] == "2026-01-01T00:00:00+00:00"
    assert out["inputs"] == {"data_fingerprint": "orig"} and out["split_config"] == {"frozen": True}
    la = out["label_access_provenance"]
    assert la["code_revision"] == "newrev999"
    assert la["derived_from_assignments"] is True and la["assignments_regenerated"] is False
    assert la["allocation_percents"] == list(SA.ALLOCATION_PERCENTS)


def test_dry_run_writes_nothing(tmp_path, monkeypatch):
    """A dry run must leave the canonical tree and log byte-identical."""
    mod = _load_migration()
    root, logs, classes = _tree(tmp_path)
    log_before = logs.read_bytes()
    _configure(mod, monkeypatch, root, logs, classes, dry_run=True)
    assert mod.main() == 0
    assert logs.read_bytes() == log_before
    assert list(root.rglob("label_access.csv")) == []


def test_log_is_written_last_so_a_failure_leaves_it_intact(tmp_path, monkeypatch):
    """If artifact construction raises, the canonical log must be untouched -- a half-updated log
    pointing at artifacts that do not exist is worse than no update at all."""
    mod = _load_migration()
    root, logs, classes = _tree(tmp_path)
    log_before = logs.read_bytes()
    _configure(mod, monkeypatch, root, logs, classes)
    monkeypatch.setattr(mod._LA, "finalize", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    with pytest.raises(RuntimeError, match="boom"):
        mod.main()
    assert logs.read_bytes() == log_before


# --------------------------------------------------------------------------- #
# Region-level outcome, end to end through the migration
# --------------------------------------------------------------------------- #
def test_an_ineligible_region_is_dropped_and_leaves_no_stale_order(tmp_path, monkeypatch):
    """A region excluded at any seed keeps no frozen order on disk and no pointer in the log --
    otherwise a reader could consume an allocation order for a region that is not allocation-eligible."""
    mod = _load_migration()
    root, logs, classes = _tree(tmp_path, regions=("kenya", "togo", "sudan", "mali"))
    # shrink mali's seed-1 target pool below the max additive count, in the FROZEN assignments
    rel = root / "cropharvest/geographic_ood/1/mali/assignments.csv"
    kept = [r for r in SA.read_assignments_csv(rel)
            if not (r["partition"] == "target_label_pool" and int(r["stable_id"].split("_p")[1]) >= 16)]
    with rel.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(SA.CSV_HEADER)
        for r in kept:
            w.writerow([r[c] for c in SA.CSV_HEADER])
    # pre-seed a stale order for mali so we can prove it is removed
    for seed in (0, 1, 2):
        stale = root / f"cropharvest/geographic_ood/{seed}/mali/label_access.csv"
        stale.parent.mkdir(parents=True, exist_ok=True)
        stale.write_text("stale\n")
    entries = json.loads(logs.read_text())
    for e in entries["leaves"]:
        if e["holdout"] == "mali":
            e["label_access_csv"] = f"cropharvest/geographic_ood/{e['seed']}/mali/label_access.csv"
            e["label_access_sha256"] = "y" * 64
    logs.write_text(json.dumps(entries, indent=2))

    _configure(mod, monkeypatch, root, logs, classes)
    assert mod.main() == 0

    out = json.loads(logs.read_text())
    mali = [e for e in out["leaves"] if e["holdout"] == "mali"]
    assert len(mali) == 3
    for e in mali:
        assert e["label_access"]["excluded"] is True
        assert "label_access_csv" not in e and "label_access_sha256" not in e
        assert not (root / f"cropharvest/geographic_ood/{e['seed']}/mali/label_access.csv").exists()
    survivors = {e["holdout"] for e in out["leaves"] if e.get("label_access_csv")}
    assert survivors == {"kenya", "togo", "sudan"}


def test_migrated_orders_use_the_new_schema_and_a_shared_budget(tmp_path, monkeypatch):
    """The migrated artifacts are the NEW nested-source-order schema, and every region shares one B_d."""
    mod = _load_migration()
    root, logs, classes = _tree(tmp_path)
    _configure(mod, monkeypatch, root, logs, classes)
    assert mod.main() == 0
    for p in root.rglob("label_access.csv"):
        assert p.read_text().splitlines()[0] == ",".join(SA.LABEL_ACCESS_HEADER)
    out = json.loads(logs.read_text())
    budgets = {e["label_access"]["benchmark_budget"] for e in out["leaves"] if e.get("label_access_csv")}
    assert budgets == {120}, budgets
