"""Schema-v2 geographic_ood: true leave-one-domain-out with purge-before-partition, has_target=True /
supports_target_labels=True (the held-out region's labels drive the target-label-budget routes).

Semantic coverage: exact target rotation, complete non-target source coverage, purge-before-partition,
no silent fallback, one-class route suppression (zero-shot stress), PASTIS tiles-not-folds, patch
atomicity, and target_test invariance across seeds. No real data, no model.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from evals import split_artifacts as SA
from evals import split_spec
from evals.regimes import base as RB
from evals.regimes import geographic_ood as geo
from tests import splitfix


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #
def _ch_bench(per=40, seed=0):
    """CropHarvest-like: headline localized (kenya/togo/ethiopia/rwanda), one-class supplementary
    (central-asia), and a source-only global collection (geowiki-landcover-2017)."""
    rng = np.random.default_rng(seed)
    centers = {
        "kenya": (0.5, 37.0), "togo": (8.0, 1.0), "ethiopia": (9.0, 40.0), "rwanda": (-2.0, 30.0),
        "central-asia": (43.0, 68.0), "geowiki-landcover-2017": (20.0, 20.0),
    }
    groups, y, latlon = [], [], []
    for dom, (la, lo) in centers.items():
        for i in range(per):
            groups.append(dom)
            y.append(0 if dom == "central-asia" else i % 2)  # central-asia is one-class
            latlon.append((la + rng.normal(0, 0.05), lo + rng.normal(0, 0.05)))
    bench = SimpleNamespace(
        name="cropharvest", groups=np.asarray(groups, dtype=object), labels=np.asarray(y, dtype=np.int64),
        latlon=np.asarray(latlon, dtype=float), sample_ids=np.asarray([f"s{i}" for i in range(len(y))], dtype=object),
    )
    bench_mod = SimpleNamespace(BENCHMARK="cropharvest", make_targets=lambda b: (b.labels, b.groups))
    return bench, bench_mod


def _euro_bench(per=40):
    groups, y, latlon = [], [], []
    centers = {"Estonia": (59.0, 25.0), "Latvia": (57.0, 24.0), "Portugal": (39.5, -8.0)}
    for dom, (la, lo) in centers.items():
        for i in range(per):
            groups.append(dom)
            y.append(i % 4)
            latlon.append((la + (i % 5) * 0.01, lo + (i % 3) * 0.01))
    bench = SimpleNamespace(
        name="eurocropsml", groups=np.asarray(groups, dtype=object), labels=np.asarray(y, dtype=np.int64),
        latlon=np.asarray(latlon, dtype=float), sample_ids=np.asarray([f"e{i}" for i in range(len(y))], dtype=object),
    )
    bench_mod = SimpleNamespace(BENCHMARK="eurocropsml", make_targets=lambda b: (b.labels, b.groups))
    return bench, bench_mod


def _pastis_bench(per_tile=8):
    tiles = list(split_spec.PASTIS.geographic_targets)  # 4 Sentinel tiles
    patches, pid = [], 100
    centers = {t: (45.0 + i, -1.0 + i) for i, t in enumerate(tiles)}
    for tile in tiles:
        la, lo = centers[tile]
        for k in range(per_tile):
            patches.append(SimpleNamespace(patch_id=pid, fold=(pid % 5) + 1, tile=tile, latlon=(la + k * 0.01, lo + k * 0.01)))
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
# Route capabilities + target rotation
# --------------------------------------------------------------------------- #
def test_geographic_declares_target_geography_and_target_labels():
    assert RB.route_capabilities(geo) == (True, True)


def test_exact_target_rotation_headline_plus_supplementary_no_source_only():
    bench, bench_mod = _ch_bench()
    labels = [sp.label for sp in geo.iter_source_target_splits(bench, bench_mod, 0)]
    spec = split_spec.CROPHARVEST
    present = {str(g) for g in bench.groups}
    expected = [t for t in (*spec.geographic_targets, *spec.supplementary_targets) if t in present]
    assert sorted(labels) == sorted(expected)
    # source-only global collections NEVER rotate as targets
    assert "geowiki-landcover-2017" not in labels
    for unit in spec.source_only_units:
        assert unit not in labels


# --------------------------------------------------------------------------- #
# Complete source-domain coverage + partition exactness
# --------------------------------------------------------------------------- #
def test_source_is_complete_non_target_population_and_exact_80_10_10():
    bench, bench_mod = _ch_bench()
    groups = np.asarray(bench.groups).astype(str)
    splits = {sp.label: sp for sp in geo.iter_source_target_splits(bench, bench_mod, 0)}
    sp = splits["kenya"]
    src = np.concatenate([sp.source_train, sp.source_val, sp.source_test])
    # every source sample is non-target (kenya excluded); the source-only collection IS in source
    assert not np.isin(sp.source_train, np.flatnonzero(groups == "kenya")).any()
    assert set(groups[src]) == set(groups.tolist()) - {"kenya"}  # ALL other domains covered
    tr, va, te = split_spec.source_partition_sizes(len(src))
    assert (len(sp.source_train), len(sp.source_val), len(sp.source_test)) == (tr, va, te)


def test_headline_target_is_split_80_20_pool_and_test():
    bench, bench_mod = _ch_bench()
    groups = np.asarray(bench.groups).astype(str)
    sp = {s.label: s for s in geo.iter_source_target_splits(bench, bench_mod, 0)}["togo"]
    target_rows = set(np.flatnonzero(groups == "togo").tolist())
    assert sp.has_target is True and sp.supports_target_labels is True
    pool_n, test_n = split_spec.target_partition_sizes(len(target_rows))
    assert (len(sp.target_label_pool), len(sp.target_test)) == (pool_n, test_n)
    # pool and test are disjoint and together are exactly the target region
    assert set(sp.target_label_pool.tolist()).isdisjoint(sp.target_test.tolist())
    assert set(sp.target_label_pool.tolist()) | set(sp.target_test.tolist()) == target_rows


# --------------------------------------------------------------------------- #
# One-class route suppression (zero-shot stress)
# --------------------------------------------------------------------------- #
def test_one_class_supplementary_target_is_zero_shot_stress():
    bench, bench_mod = _ch_bench()
    groups = np.asarray(bench.groups).astype(str)
    sp = {s.label: s for s in geo.iter_source_target_splits(bench, bench_mod, 0)}["central-asia"]
    assert sp.has_target is True
    assert sp.supports_target_labels is False       # excluded from target-label routes
    assert sp.target_label_pool.size == 0           # no label pool for a one-class stress target
    # the whole region is the zero-shot evaluation set
    assert set(sp.target_test.tolist()) == set(np.flatnonzero(groups == "central-asia").tolist())


# --------------------------------------------------------------------------- #
# Purge-before-partition
# --------------------------------------------------------------------------- #
def test_purge_removes_source_within_radius_before_partitioning():
    bench, bench_mod = _ch_bench()
    groups = np.asarray(bench.groups).astype(str)
    # move rwanda ONTO kenya so the 50 km purge removes rwanda source rows when kenya is the target
    bench.latlon[groups == "rwanda"] = bench.latlon[groups == "kenya"][0] + 1e-4

    # isolate the per-target audit window (each target emits its own purge), like the generator does
    RB.clear_split_audit_events()
    events = RB.SPLIT_AUDIT_EVENTS
    windows, prev = {}, 0
    splits = {}
    for s in geo.iter_source_target_splits(bench, bench_mod, 0):
        windows[s.label] = list(events[prev:len(events)])
        prev = len(events)
        splits[s.label] = s
    sp = splits["kenya"]
    purge_events = [e for e in windows["kenya"] if e["kind"] == "purge"]
    assert purge_events, "co-located rwanda should have triggered a purge for the kenya target"
    purged = set().union(*(set(e["purged_indices"]) for e in purge_events))
    assert purged, "purge recorded no removed rows"
    # purged rows are excluded from EVERY source partition (purge happened BEFORE partitioning)
    src = set(sp.source_train.tolist()) | set(sp.source_val.tolist()) | set(sp.source_test.tolist())
    assert src.isdisjoint(purged)
    # the purged rows are rwanda rows sitting on top of the kenya target
    assert purged.issubset(set(np.flatnonzero(groups == "rwanda").tolist()))


# --------------------------------------------------------------------------- #
# No silent fallback; determinism; seed-varying membership
# --------------------------------------------------------------------------- #
def test_no_stratification_fallback_is_ever_emitted():
    bench, bench_mod = _ch_bench()
    RB.clear_split_audit_events()
    list(geo.iter_source_target_splits(bench, bench_mod, 0))
    assert not any(e["kind"] == "stratification_fallback" for e in RB.SPLIT_AUDIT_EVENTS)


def test_deterministic_in_seed_and_membership_varies_across_seeds():
    bench, bench_mod = _euro_bench()
    a = {s.label: s for s in geo.iter_source_target_splits(bench, bench_mod, 0)}
    a2 = {s.label: s for s in geo.iter_source_target_splits(bench, bench_mod, 0)}
    b = {s.label: s for s in geo.iter_source_target_splits(bench, bench_mod, 1)}
    for lab in a:
        assert np.array_equal(a[lab].source_train, a2[lab].source_train)   # same seed -> identical
        assert np.array_equal(a[lab].target_test, a2[lab].target_test)
    # membership varies across seeds while sizes hold
    assert any(not np.array_equal(a[lab].source_train, b[lab].source_train) for lab in a)
    for lab in a:
        assert len(a[lab].source_train) == len(b[lab].source_train)
        assert len(a[lab].target_test) == len(b[lab].target_test)


def test_target_test_invariant_across_seeds_in_size_and_is_exact_partition():
    """target_test is a deterministic exact 20% of the region for each seed (the fixed evaluation the
    label arms share); its SIZE never varies with the seed."""
    bench, bench_mod = _euro_bench()
    sizes = set()
    for seed in (0, 1, 2):
        sp = {s.label: s for s in geo.iter_source_target_splits(bench, bench_mod, seed)}["Estonia"]
        groups = np.asarray(bench.groups).astype(str)
        region = set(np.flatnonzero(groups == "Estonia").tolist())
        assert set(sp.target_test.tolist()) | set(sp.target_label_pool.tolist()) == region
        sizes.add((len(sp.target_label_pool), len(sp.target_test)))
    assert len(sizes) == 1  # sizes fixed across seeds


# --------------------------------------------------------------------------- #
# PASTIS dense: tiles-not-folds + patch atomicity
# --------------------------------------------------------------------------- #
def test_pastis_dense_lodo_is_over_tiles_not_folds():
    bench, bench_mod = _pastis_bench()
    labels = [d.label for d in geo.iter_dense_source_target_splits(bench, bench_mod, 0)]
    assert sorted(labels) == sorted(str(t) for t in split_spec.PASTIS.geographic_targets)
    # patch_domains records the Sentinel TILE, never the (cache-layout) fold
    doms = geo.patch_domains(bench, bench_mod)
    tiles = {int(p.patch_id): str(p.tile) for p in bench.patches}
    folds = {int(p.patch_id): str(p.fold) for p in bench.patches}
    assert doms == tiles
    assert all(doms[p] != folds[p] for p in doms)


def test_pastis_dense_patches_are_atomic_and_target_is_the_held_out_tile():
    bench, bench_mod = _pastis_bench()
    tile_of = {p.patch_id: p.tile for p in bench.patches}
    all_patches = set(bench.patch_ids(None))
    for d in geo.iter_dense_source_target_splits(bench, bench_mod, 0):
        parts = [
            d.source_train_patches, d.source_val_patches, d.source_test_patches,
            d.target_label_pool_patches, d.target_test_patches,
        ]
        # every patch appears in EXACTLY one partition (atomicity), none split
        flat = [p for part in parts for p in part]
        assert len(flat) == len(set(flat)) == len(all_patches)
        # the target partitions are exactly the held-out tile; source never contains it
        target = d.target_label_pool_patches | d.target_test_patches
        assert {tile_of[p] for p in target} == {d.label}
        source = d.source_train_patches | d.source_val_patches | d.source_test_patches
        assert d.label not in {tile_of[p] for p in source}
        assert d.has_target is True and d.supports_target_labels is True


# --------------------------------------------------------------------------- #
# Artifact round-trip (tabular + dense) -- v2 builders, structural tile check
# --------------------------------------------------------------------------- #
def test_geographic_tabular_round_trip(tmp_path):
    bench, bench_mod = _euro_bench()
    split = next(iter(geo.iter_source_target_splits(bench, bench_mod, 0)))
    domains = geo.sample_domains(bench, bench_mod)
    y = bench_mod.make_targets(bench)[0]
    root = tmp_path / "splits"
    rows, summary = SA.build_tabular_leaf(
        "eurocropsml", "geographic_ood", 0, split=split, domains=domains, labels=y,
        sample_ids=bench.sample_ids, audit_events=[], purge_km=25.0,
    )
    splitfix.freeze(root, [(rows, summary)])
    loaded = SA.load_tabular_splits(root, "eurocropsml", bench.sample_ids, ["geographic_ood"], [0])
    ls = loaded[0]
    assert ls.split.has_target is True and ls.split.supports_target_labels is True
    for part, arr in split.as_partitions().items():
        assert set(getattr(ls.split, part).tolist()) == {int(i) for i in arr.tolist()}


def test_geographic_dense_round_trip_checks_tiles_not_folds(tmp_path):
    bench, bench_mod = _pastis_bench()
    dsplit = next(iter(geo.iter_dense_source_target_splits(bench, bench_mod, 0)))
    domain_of = {int(k): str(v) for k, v in geo.patch_domains(bench, bench_mod).items()}
    all_pids = [int(p) for p in bench.patch_ids(None)]
    root = tmp_path / "splits"
    rows, summary = SA.build_dense_leaf(
        "pastis", "geographic_ood", 0, dense_split=dsplit, audit_events=[],
        all_patch_ids=all_pids, domain_of=domain_of,
        class_sets={int(k): set(v) for k, v in bench.patch_class_sets(all_pids).items()},
        patch_latlon=dict(bench.patch_latlon), purge_km=2.0,
    )
    splitfix.freeze(root, [(rows, summary)])
    patch_fold = {int(p.patch_id): int(p.fold) for p in bench.patches}
    patch_tile = {int(k): v for k, v in bench.patch_tiles.items()}
    by_seed = SA.load_dense_splits(root, "pastis", patch_fold, patch_tile, ["geographic_ood"], [0])
    assert by_seed[0][0].split.as_partitions() == dsplit.as_partitions()

    # structural check is against the TILE (not fold): a shifted tile is refused
    tampered = dict(patch_tile)
    a_target = sorted(dsplit.target_test_patches)[0]
    tampered[a_target] = "T99XXX"
    with pytest.raises(SA.SplitArtifactError, match="structural metadata changed"):
        SA.load_dense_splits(root, "pastis", patch_fold, tampered, ["geographic_ood"], [0])


# --------------------------------------------------------------------------- #
# Defect 1: invalid coordinates fail closed (never silently retained/dropped/skipped)
# --------------------------------------------------------------------------- #
def test_invalid_source_coordinate_is_a_hard_error_tabular():
    bench, bench_mod = _ch_bench()
    groups = np.asarray(bench.groups).astype(str)
    # a SOURCE sample (togo) with a NaN coordinate cannot be distance-checked against the target
    bench.latlon[np.flatnonzero(groups == "togo")[0]] = [np.nan, np.nan]
    with pytest.raises(ValueError, match="non-finite coordinates"):
        list(geo.iter_source_target_splits(bench, bench_mod, 0))


def test_invalid_target_coordinate_is_a_hard_error_tabular():
    bench, bench_mod = _ch_bench()
    groups = np.asarray(bench.groups).astype(str)
    bench.latlon[np.flatnonzero(groups == "kenya")[0]] = [np.nan, np.nan]
    with pytest.raises(ValueError, match="non-finite coordinates") as exc:
        list(geo.iter_source_target_splits(bench, bench_mod, 0))
    assert "cropharvest/geographic_ood" in str(exc.value)  # benchmark/target context in the message


def test_missing_benchmark_coordinates_is_a_hard_error_tabular():
    bench, bench_mod = _ch_bench()
    bench.latlon = None  # no coordinates at all -> the load-bearing purge cannot run
    with pytest.raises(ValueError, match="requires geographic coordinates"):
        list(geo.iter_source_target_splits(bench, bench_mod, 0))


def test_invalid_patch_coordinate_is_a_hard_error_dense():
    bench, bench_mod = _pastis_bench()
    p0 = bench.patches[0].patch_id
    bench.patch_latlon[p0] = (np.nan, np.nan)  # a patch centroid is missing
    with pytest.raises(ValueError, match="non-finite coordinates"):
        list(geo.iter_dense_source_target_splits(bench, bench_mod, 0))


# --------------------------------------------------------------------------- #
# Defect 3: machine-readable headline vs supplementary-stress target role
# --------------------------------------------------------------------------- #
def test_target_role_marks_headline_vs_supplementary_stress():
    bench, bench_mod = _ch_bench()
    splits = {s.label: s for s in geo.iter_source_target_splits(bench, bench_mod, 0)}
    assert splits["kenya"].target_role == RB.TARGET_ROLE_HEADLINE
    assert splits["central-asia"].target_role == RB.TARGET_ROLE_SUPPLEMENTARY_STRESS
    # a stress role must be zero-shot (enforced at construction)
    assert splits["central-asia"].supports_target_labels is False


def test_target_role_survives_the_tabular_artifact_round_trip(tmp_path):
    bench, bench_mod = _ch_bench()
    stress = next(s for s in geo.iter_source_target_splits(bench, bench_mod, 0) if s.label == "central-asia")
    domains = geo.sample_domains(bench, bench_mod)
    y = bench_mod.make_targets(bench)[0]
    root = tmp_path / "splits"
    rows, summary = SA.build_tabular_leaf(
        "cropharvest", "geographic_ood", 0, split=stress, domains=domains, labels=y,
        sample_ids=bench.sample_ids, audit_events=[], purge_km=50.0,
    )
    # the central-log entry records the machine-readable role
    assert summary["target_role"] == RB.TARGET_ROLE_SUPPLEMENTARY_STRESS
    splitfix.freeze(root, [(rows, summary)])
    loaded = SA.load_tabular_splits(root, "cropharvest", bench.sample_ids, ["geographic_ood"], [0])
    assert loaded[0].split.target_role == RB.TARGET_ROLE_SUPPLEMENTARY_STRESS
