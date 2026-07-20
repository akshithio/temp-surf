"""Schema-v2 spatial_cluster_ood: coordinate-only spherical-K-means cells, five fixed cells each
rotated once as the target, purge-before-partition, has_target=True / supports_target_labels=False.

This regime is a SPLIT-SENSITIVITY analysis, not a second deployment setting: the held-out cell is
scored zero-shot in full, so target_test is the complete cell and target_label_pool is always empty.

Semantic coverage (no real data, no model): exactly five fixed cells and five rotations, boundaries
identical across run seeds, seed-varying source subdivisions at fixed counts, deterministic
label-independent assignment, exactly-once accounting, purge-before-partition, fail-closed
coordinates, whole-cell zero-shot target_test, source_test as an untouched within-source reference,
PASTIS patch atomicity, and artifact round-trips that preserve the frozen cluster IDs without
re-clustering.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from evals import split_artifacts as SA
from evals import split_spec
from evals.regimes import base as RB
from evals.regimes import spatial_cluster_ood as sc
from tests import splitfix

# five far-apart coordinate blobs -> five clean spherical-K-means cells; well beyond any purge radius
_FAR_CENTERS = [(0.5, 37.0), (8.0, 1.0), (9.0, 40.0), (-12.0, -55.0), (43.0, 68.0)]


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #
def _tab_bench(per=24, benchmark="cropharvest", centers=_FAR_CENTERS, seed=0):
    """Tabular bench with ``len(centers)`` far-apart two-class blobs (cells never touch -> no purge)."""
    rng = np.random.default_rng(seed)
    groups, y, latlon, sids = [], [], [], []
    for ci, (la, lo) in enumerate(centers):
        for i in range(per):
            groups.append(f"native{ci}")  # native regions the cell construction MUST ignore
            y.append(i % 2)
            latlon.append((la + rng.normal(0, 0.03), lo + rng.normal(0, 0.03)))
            sids.append(f"c{ci}_{i}")
    bench = SimpleNamespace(
        name=benchmark, groups=np.asarray(groups, dtype=object), labels=np.asarray(y, dtype=np.int64),
        latlon=np.asarray(latlon, dtype=float), sample_ids=np.asarray(sids, dtype=object),
    )
    bench_mod = SimpleNamespace(BENCHMARK=benchmark, make_targets=lambda b: (b.labels, b.groups))
    return bench, bench_mod


def _line_bench(per=30):
    """Five cells spaced 0.4 deg (~44 km) apart in latitude so, under CropHarvest's 50 km purge, each
    cell's immediate neighbour(s) fall inside the purge radius of the target -- and NON-neighbours
    (0.8 deg ~ 89 km) do not. Deterministic coordinates (no RNG) so the purged set is exact."""
    groups, y, latlon, sids = [], [], [], []
    for ci in range(5):
        lat = 0.4 * ci
        for i in range(per):
            groups.append(f"native{ci}")
            y.append(i % 2)
            latlon.append((lat, 0.0005 * i))  # tiny lon spread for distinctness; lat drives the cells
            sids.append(f"c{ci}_{i}")
    bench = SimpleNamespace(
        name="cropharvest", groups=np.asarray(groups, dtype=object), labels=np.asarray(y, dtype=np.int64),
        latlon=np.asarray(latlon, dtype=float), sample_ids=np.asarray(sids, dtype=object),
    )
    bench_mod = SimpleNamespace(BENCHMARK="cropharvest", make_targets=lambda b: (b.labels, b.groups))
    return bench, bench_mod


def _pastis_bench(per_group=8):
    """PASTIS-like bench: five far-apart patch groups -> five patch-level cells. Patches carry a
    published fold and a Sentinel tile that the cell construction MUST ignore (it uses only centroids)."""
    centers = [(45.0, -1.0), (46.0, 30.0), (10.0, 40.0), (-20.0, -55.0), (60.0, 100.0)]
    patches, pid = [], 100
    for gi, (la, lo) in enumerate(centers):
        for k in range(per_group):
            patches.append(SimpleNamespace(patch_id=pid, fold=(pid % 5) + 1, tile=f"T{gi}", latlon=(la + k * 0.01, lo + k * 0.01)))
            pid += 1
    pids = [p.patch_id for p in patches]
    class_sets = {p.patch_id: {p.patch_id % 4, 10 + (p.patch_id % 3)} for p in patches}
    bench = SimpleNamespace(
        patches=patches,
        patch_ids=lambda folds=None: [p.patch_id for p in patches if folds is None or p.fold in folds],
        patch_class_sets=lambda ids=None: {int(p): class_sets[int(p)] for p in (ids if ids is not None else pids)},
        patch_tiles={p.patch_id: p.tile for p in patches},
        patch_latlon={p.patch_id: p.latlon for p in patches},
    )
    bench_mod = SimpleNamespace(BENCHMARK="pastis")
    return bench, bench_mod


# --------------------------------------------------------------------------- #
# Route capabilities + five fixed cells / five rotations
# --------------------------------------------------------------------------- #
def test_spatial_declares_target_geography_but_no_target_labels():
    """Split-sensitivity only: spatial cells are scored zero-shot. geographic_ood is the sole regime
    carrying the target-label-access suite."""
    assert RB.route_capabilities(sc) == (True, False)


def test_exactly_five_fixed_cells_and_five_target_rotations():
    bench, bench_mod = _tab_bench()
    cells = sc.sample_domains(bench, bench_mod)
    assert sorted(set(cells.tolist())) == list(sc._CELL_NAMES)  # exactly cluster_00..cluster_04
    assert len(sc._CELL_NAMES) == sc.N_CLUSTERS == 5
    labels = [s.label for s in sc.iter_source_target_splits(bench, bench_mod, 0)]
    assert labels == list(sc._CELL_NAMES)  # each cell rotated once as the target, in canonical order


def test_cell_naming_is_ascending_mean_latitude_then_longitude():
    bench, bench_mod = _tab_bench()
    cells = sc.sample_domains(bench, bench_mod)
    centers = [
        (float(bench.latlon[cells == name, 0].mean()), float(bench.latlon[cells == name, 1].mean()))
        for name in sc._CELL_NAMES
    ]
    assert centers == sorted(centers)  # cluster_00..04 are sorted by (mean lat, mean lon)


# --------------------------------------------------------------------------- #
# Seed behavior: boundaries frozen, subdivisions vary at fixed counts
# --------------------------------------------------------------------------- #
def test_cell_boundaries_are_identical_across_run_seeds():
    bench, bench_mod = _tab_bench()
    full_cell = {}
    for seed in (0, 1, 2):
        for s in sc.iter_source_target_splits(bench, bench_mod, seed):
            cell = frozenset(s.target_label_pool.tolist()) | frozenset(s.target_test.tolist())
            full_cell.setdefault(s.label, set()).add(cell)
    # the target cell (pool | test) is exactly the same set of rows for every run seed
    assert all(len(variants) == 1 for variants in full_cell.values())
    # and it matches the seedless sample_domains cell membership
    cells = sc.sample_domains(bench, bench_mod)
    for name, variants in full_cell.items():
        assert next(iter(variants)) == set(np.flatnonzero(cells == name).tolist())


def test_seed_subdivisions_vary_but_counts_are_fixed():
    bench, bench_mod = _tab_bench()
    a = {s.label: s for s in sc.iter_source_target_splits(bench, bench_mod, 0)}
    a2 = {s.label: s for s in sc.iter_source_target_splits(bench, bench_mod, 0)}
    b = {s.label: s for s in sc.iter_source_target_splits(bench, bench_mod, 1)}
    for lab in a:  # same seed -> identical membership (deterministic)
        assert np.array_equal(a[lab].source_train, a2[lab].source_train)
        assert np.array_equal(a[lab].target_test, a2[lab].target_test)
    # membership varies across seeds while every partition SIZE is fixed
    assert any(not np.array_equal(a[lab].source_train, b[lab].source_train) for lab in a)
    for lab in a:
        for part in ("source_train", "source_val", "source_test", "target_label_pool", "target_test"):
            assert len(getattr(a[lab], part)) == len(getattr(b[lab], part))


def test_cells_are_deterministic_and_label_independent():
    bench, bench_mod = _tab_bench()
    base = sc.sample_domains(bench, bench_mod)
    assert np.array_equal(base, sc.sample_domains(bench, bench_mod))  # deterministic
    # permuting / replacing the labels does NOT change any cell assignment (cells use coordinates only)
    flipped = SimpleNamespace(**{**bench.__dict__, "labels": 1 - bench.labels})
    assert np.array_equal(base, sc.sample_domains(flipped, bench_mod))
    constant = SimpleNamespace(**{**bench.__dict__, "labels": np.zeros_like(bench.labels)})
    assert np.array_equal(base, sc.sample_domains(constant, bench_mod))


# --------------------------------------------------------------------------- #
# Exactly-once accounting + partition exactness
# --------------------------------------------------------------------------- #
def test_every_eligible_item_appears_exactly_once_per_split():
    bench, bench_mod = _tab_bench()
    n = len(bench.labels)
    for s in sc.iter_source_target_splits(bench, bench_mod, 0):
        parts = [s.source_train, s.source_val, s.source_test, s.target_label_pool, s.target_test]
        flat = np.concatenate(parts)
        assert len(flat) == len(set(flat.tolist()))       # pairwise disjoint (no item in two partitions)
        # no purge fires on the far-apart fixture, so the five partitions cover the whole population
        assert set(flat.tolist()) == set(range(n))
        tr, va, te = split_spec.source_partition_sizes(len(s.source_train) + len(s.source_val) + len(s.source_test))
        assert (len(s.source_train), len(s.source_val), len(s.source_test)) == (tr, va, te)


def test_whole_cell_is_zero_shot_target_test_with_no_label_pool():
    """No part of a spatial cell is ever trainable: the label pool is empty and target_test is the
    COMPLETE cell, so a target-label route cannot be constructed here even by accident."""
    bench, bench_mod = _tab_bench()
    cells = sc.sample_domains(bench, bench_mod)
    for s in sc.iter_source_target_splits(bench, bench_mod, 0):
        cell_rows = set(np.flatnonzero(cells == s.label).tolist())
        assert s.target_label_pool.size == 0
        assert set(s.target_test.tolist()) == cell_rows
        assert s.has_target is True and s.supports_target_labels is False
        assert s.target_role == RB.TARGET_ROLE_HEADLINE


def test_source_test_is_a_distinct_within_source_reference():
    """source_test is the untouched within-source reference: a real partition, disjoint from the
    training/calibration source and from the target. (The runtime never trains/tunes on it -- pinned
    by test_erm_characterization.test_tabular_source_test_scope_uses_exact_partition_ids_not_trained_on.)"""
    bench, bench_mod = _tab_bench()
    for s in sc.iter_source_target_splits(bench, bench_mod, 0):
        st = set(s.source_test.tolist())
        assert st                                            # non-empty within-source reference
        assert st.isdisjoint(s.source_train.tolist())
        assert st.isdisjoint(s.source_val.tolist())
        assert st.isdisjoint(s.target_label_pool.tolist())
        assert st.isdisjoint(s.target_test.tolist())


# --------------------------------------------------------------------------- #
# Purge before partition
# --------------------------------------------------------------------------- #
def test_purge_removes_source_within_radius_before_partitioning():
    bench, bench_mod = _line_bench()
    cells = sc.sample_domains(bench, bench_mod)
    RB.clear_split_audit_events()
    events = RB.SPLIT_AUDIT_EVENTS
    windows, prev, splits = {}, 0, {}
    for s in sc.iter_source_target_splits(bench, bench_mod, 0):
        windows[s.label] = list(events[prev:len(events)])
        prev = len(events)
        splits[s.label] = s
    sp = splits["cluster_00"]
    purge_events = [e for e in windows["cluster_00"] if e["kind"] == "purge"]
    assert purge_events, "the adjacent cell within 50 km should have triggered a purge for cluster_00"
    purged = set().union(*(set(e["purged_indices"]) for e in purge_events))
    # cluster_01 (~44 km away) is purged; cluster_02+ (~89 km) are not
    assert purged == set(np.flatnonzero(cells == "cluster_01").tolist())
    # purge happened BEFORE partitioning: purged rows are in NO source partition
    src = set(sp.source_train.tolist()) | set(sp.source_val.tolist()) | set(sp.source_test.tolist())
    assert src.isdisjoint(purged)
    # every cell still rotates as a valid target (none dropped/merged to recover class support)
    assert list(splits) == list(sc._CELL_NAMES)


def test_no_stratification_fallback_is_ever_emitted():
    bench, bench_mod = _tab_bench()
    RB.clear_split_audit_events()
    list(sc.iter_source_target_splits(bench, bench_mod, 0))
    assert not any(e["kind"] == "stratification_fallback" for e in RB.SPLIT_AUDIT_EVENTS)


# --------------------------------------------------------------------------- #
# Fail-closed coordinates (never silently retained / dropped / skipped)
# --------------------------------------------------------------------------- #
def test_invalid_coordinate_is_a_hard_error_tabular():
    bench, bench_mod = _tab_bench()
    bench.latlon[3] = [np.nan, np.nan]  # one sample loses its coordinate
    with pytest.raises(ValueError, match="non-finite coordinates") as exc:
        list(sc.iter_source_target_splits(bench, bench_mod, 0))
    assert "cropharvest/spatial_cluster_ood" in str(exc.value)  # benchmark context in the message


def test_missing_benchmark_coordinates_is_a_hard_error_tabular():
    bench, bench_mod = _tab_bench()
    bench.latlon = None
    with pytest.raises(ValueError, match="no geographic coordinates"):
        list(sc.iter_source_target_splits(bench, bench_mod, 0))


def test_invalid_patch_coordinate_is_a_hard_error_dense():
    bench, bench_mod = _pastis_bench()
    bench.patch_latlon[bench.patches[0].patch_id] = (np.nan, np.nan)
    with pytest.raises(ValueError, match="non-finite coordinates"):
        list(sc.iter_dense_source_target_splits(bench, bench_mod, 0))


# --------------------------------------------------------------------------- #
# PASTIS dense: five patch-level cells, atomicity, cell-not-fold domain
# --------------------------------------------------------------------------- #
def test_pastis_dense_five_cells_and_patch_atomicity():
    bench, bench_mod = _pastis_bench()
    all_patches = set(bench.patch_ids(None))
    dsplits = list(sc.iter_dense_source_target_splits(bench, bench_mod, 0))
    assert [d.label for d in dsplits] == list(sc._CELL_NAMES)
    for d in dsplits:
        parts = [
            d.source_train_patches, d.source_val_patches, d.source_test_patches,
            d.target_label_pool_patches, d.target_test_patches,
        ]
        flat = [p for part in parts for p in part]
        # every patch appears in EXACTLY one partition, none split (whole-patch atomicity)
        assert len(flat) == len(set(flat)) == len(all_patches)
        assert d.has_target is True and d.supports_target_labels is False
        assert not d.target_label_pool_patches          # zero-shot: no trainable target patches
        assert d.target_role == RB.TARGET_ROLE_HEADLINE


def test_pastis_patch_domains_record_the_cell_not_the_fold_or_tile():
    bench, bench_mod = _pastis_bench()
    doms = sc.patch_domains(bench, bench_mod)
    assert set(doms.values()) == set(sc._CELL_NAMES)                 # cells, not folds/tiles
    folds = {int(p.patch_id): str(p.fold) for p in bench.patches}
    tiles = {int(p.patch_id): str(p.tile) for p in bench.patches}
    assert all(doms[p] != folds[p] and doms[p] != tiles[p] for p in doms)


# --------------------------------------------------------------------------- #
# Artifact round-trips preserve the frozen cluster IDs + partitions (no re-clustering at load)
# --------------------------------------------------------------------------- #
def test_spatial_tabular_round_trip_preserves_cluster_ids_and_partitions(tmp_path):
    bench, bench_mod = _tab_bench()
    domains = sc.sample_domains(bench, bench_mod)
    y = bench_mod.make_targets(bench)[0]
    root = tmp_path / "splits"
    split = next(iter(sc.iter_source_target_splits(bench, bench_mod, 0)))
    rows, summary = SA.build_tabular_leaf(
        "cropharvest", "spatial_cluster_ood", 0, split=split, domains=domains, labels=y,
        sample_ids=bench.sample_ids, audit_events=[], purge_km=50.0,
    )
    splitfix.freeze(root, [(rows, summary)])
    # the central-log entry records the frozen cell as the domain basis + a headline role
    assert summary["group_kind"] == "spatial_cluster" and summary["target_role"] == RB.TARGET_ROLE_HEADLINE
    # every assigned id's FROZEN cell is stored verbatim in the CSV (source rows keep their own cells;
    # the target rows carry the held-out cell) -- the runtime reads cluster IDs from the CSV, not K-means
    csv_rows = SA.read_assignments_csv(SA.assignments_path(root, "cropharvest", "spatial_cluster_ood", 0, split.label))
    doms = {r["stable_id"]: r["domain"] for r in csv_rows if r["status"] == SA.STATUS_ASSIGNED}
    id_to_cell = {str(sid): str(domains[i]) for i, sid in enumerate(bench.sample_ids.tolist())}
    assert all(doms[sid] == id_to_cell[sid] for sid in doms)
    assert set(doms.values()) <= set(sc._CELL_NAMES)
    target_ids = {str(bench.sample_ids[i]) for i in split.target_test.tolist()}
    assert {doms[t] for t in target_ids} == {split.label}  # target_test IS the held-out cell
    loaded = SA.load_tabular_splits(root, "cropharvest", bench.sample_ids, ["spatial_cluster_ood"], [0])
    ls = loaded[0]
    assert ls.split.has_target is True and ls.split.supports_target_labels is False
    for part, arr in split.as_partitions().items():
        assert set(getattr(ls.split, part).tolist()) == {int(i) for i in arr.tolist()}


def test_spatial_dense_round_trip_preserves_cluster_ids_and_is_not_reclustered(tmp_path, monkeypatch):
    bench, bench_mod = _pastis_bench()
    dsplit = next(iter(sc.iter_dense_source_target_splits(bench, bench_mod, 0)))
    domain_of = {int(k): str(v) for k, v in sc.patch_domains(bench, bench_mod).items()}
    all_pids = [int(p) for p in bench.patch_ids(None)]
    root = tmp_path / "splits"
    rows, summary = SA.build_dense_leaf(
        "pastis", "spatial_cluster_ood", 0, dense_split=dsplit, audit_events=[],
        all_patch_ids=all_pids, domain_of=domain_of,
        class_sets={int(k): set(v) for k, v in bench.patch_class_sets(all_pids).items()},
        patch_latlon=dict(bench.patch_latlon), purge_km=2.0,
    )
    splitfix.freeze(root, [(rows, summary)])
    # loading must NEVER re-run K-means (the cell is frozen in the artifact)
    monkeypatch.setattr(sc, "_cell_labels", lambda *a, **k: (_ for _ in ()).throw(
        RuntimeError("spatial cells must not be recomputed at load")))
    patch_fold = {int(p.patch_id): int(p.fold) for p in bench.patches}
    patch_tile = {int(k): v for k, v in bench.patch_tiles.items()}
    by_seed = SA.load_dense_splits(root, "pastis", patch_fold, patch_tile, ["spatial_cluster_ood"], [0])
    assert by_seed[0][0].split.as_partitions() == dsplit.as_partitions()  # cluster IDs + partitions preserved


def test_every_label_arm_consumes_identical_seed_specific_artifacts(tmp_path):
    """Two independent consumptions of the same seed's artifact return byte-identical partitions (the
    fixed per-seed split every model + label arm shares); a different seed differs in membership."""
    bench, bench_mod = _tab_bench()
    root = tmp_path / "splits"
    built = []
    for seed in (0, 1):
        domains = sc.sample_domains(bench, bench_mod)
        y = bench_mod.make_targets(bench)[0]
        for split in sc.iter_source_target_splits(bench, bench_mod, seed):
            built.append(SA.build_tabular_leaf(
                "cropharvest", "spatial_cluster_ood", seed, split=split, domains=domains, labels=y,
                sample_ids=bench.sample_ids, audit_events=[], purge_km=50.0,
            ))
    splitfix.freeze(root, built)  # one central log covering both seeds' leaves

    a = SA.load_tabular_splits(root, "cropharvest", bench.sample_ids, ["spatial_cluster_ood"], [0])
    a2 = SA.load_tabular_splits(root, "cropharvest", bench.sample_ids, ["spatial_cluster_ood"], [0])
    b = SA.load_tabular_splits(root, "cropharvest", bench.sample_ids, ["spatial_cluster_ood"], [1])
    for x, x2 in zip(a, a2, strict=True):
        assert x.split.as_partitions().keys() == x2.split.as_partitions().keys()
        for part, arr in x.split.as_partitions().items():
            assert np.array_equal(arr, getattr(x2.split, part))  # identical across independent loads
    # same frozen cells, but seed-1 membership differs from seed-0 for at least one partition
    by_label_a = {x.split.label: x.split for x in a}
    by_label_b = {x.split.label: x.split for x in b}
    assert any(
        not np.array_equal(by_label_a[lab].source_train, by_label_b[lab].source_train)
        for lab in by_label_a
    )
