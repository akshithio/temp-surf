"""Behavioral parity: preprocessing artifacts reproduce the CURRENT runtime split membership.

The acceptance criterion. For every realizable (benchmark, regime, seed, holdout), the loaded
artifact's train/val/test/source_val/source_test membership must exactly equal what
``regime_base.iter_splits`` / ``segmentation_fold_configs`` produce today. Also proves PASTIS
preprocessing needs no embedding cache, and that the audit/generation machinery records dropped
holdouts and silent stratification fallbacks without changing membership.

Everything runs on compact synthetic fixtures; no real data, no SSH.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from dataio.get_input import get_input
from evals import split_artifacts as SA
from evals.benchmarks import cropharvest as ch
from evals.regimes import base as regime_base
from evals.regimes import spatial_cluster_ood as scood
from utils import cacheutils

TABULAR_REGIMES = ["random_id", "official", "geographic_ood", "spatial_cluster_ood"]

# far-apart centers (lat, lon) so cross-domain purge never fires for the LODO fixture
_CENTERS = {
    "kenya": (0.5, 37.0), "togo": (8.0, 1.0), "ethiopia": (9.0, 40.0),
    "lem-brazil": (-12.0, -55.0), "central-asia": (43.0, 68.0), "rwanda": (-2.0, 30.0),
}
_EURO_CENTERS = {"Estonia": (59.0, 25.0), "Latvia": (57.0, 24.0), "Portugal": (39.5, -8.0)}
_BZ_CENTERS = {"frh01": (48.4, -4.5), "frh02": (48.2, -3.0), "frh03": (48.0, -2.0), "frh04": (47.8, -3.5)}


def _ch_bench(domain_sizes: dict[str, int], seed: int = 0) -> SimpleNamespace:
    rng = np.random.default_rng(seed)
    groups, labels, latlon, sample_ids = [], [], [], []
    for dom, n in domain_sizes.items():
        clat, clon = _CENTERS[dom]
        for i in range(n):
            groups.append(dom)
            labels.append(i % 2)  # balanced two classes per domain
            latlon.append((clat + rng.normal(0, 0.05), clon + rng.normal(0, 0.05)))
            sample_ids.append(f"{dom}_{i}")
    return SimpleNamespace(
        name="cropharvest",
        groups=np.asarray(groups, dtype=object),
        labels=np.asarray(labels, dtype=np.int64),
        latlon=np.asarray(latlon, dtype=float),
        sample_ids=np.asarray(sample_ids, dtype=object),
    )


def _euro_bench(country_sizes: dict[str, int], seed: int = 0) -> SimpleNamespace:
    rng = np.random.default_rng(seed)
    hcat = ["3301010000", "3301020000", "3302000000"]  # three distinct 6-digit prefixes -> 3 classes
    groups, labels, latlon, sids = [], [], [], []
    for country, n in country_sizes.items():
        clat, clon = _EURO_CENTERS[country]
        for i in range(n):
            groups.append(country)
            labels.append(i % len(hcat))  # index into label_names, per euro.make_targets
            latlon.append((clat + rng.normal(0, 0.05), clon + rng.normal(0, 0.05)))
            sids.append(f"{country}_{i}.npz")
    return SimpleNamespace(
        name="eurocropsml", groups=np.asarray(groups, dtype=object), labels=np.asarray(labels, dtype=np.int64),
        latlon=np.asarray(latlon, dtype=float), sample_ids=np.asarray(sids, dtype=object), label_names=hcat,
    )


def _breizh_bench(region_sizes: dict[str, int], seed: int = 0) -> SimpleNamespace:
    rng = np.random.default_rng(seed)
    groups, labels, latlon, sids = [], [], [], []
    for region, n in region_sizes.items():
        clat, clon = _BZ_CENTERS[region]
        for i in range(n):
            groups.append(region)
            labels.append(i % 3)
            latlon.append((clat + rng.normal(0, 0.03), clon + rng.normal(0, 0.03)))
            sids.append(f"{region}:{i}")
    return SimpleNamespace(
        name="breizhcrops", groups=np.asarray(groups, dtype=object), labels=np.asarray(labels, dtype=np.int64),
        latlon=np.asarray(latlon, dtype=float), sample_ids=np.asarray(sids, dtype=object), label_names=None,
    )


def _parity_for_regime(tmp_path, bench, bench_mod, regime, seed) -> list[str]:
    """Publish every yielded split, reload it (with identity checks), and assert index-set parity."""
    y, _groups = bench_mod.make_targets(bench)
    holdouts = regime_base.holdouts_for(bench_mod, regime)
    val_group = regime_base.val_group_for(bench_mod, regime)
    id_map = {str(s): i for i, s in enumerate(bench.sample_ids.tolist())}
    benchmark = bench_mod.BENCHMARK

    labels_seen = []
    for (label, train, val, test, domains, has_target, group_kind, source_val, source_test) in \
            regime_base.iter_splits(regime, bench, y, holdouts, seed, val_group=val_group):
        spec, eligible = SA.build_tabular_leaf(
            benchmark, regime, seed, label=label,
            train=train, val=val, test=test, source_val=source_val, source_test=source_test,
            domains=domains, labels=y, sample_ids=bench.sample_ids, has_target=has_target,
            group_kind=group_kind, params={"assembly_seed": 0}, audit_events=[],
        )
        SA.publish_leaf(tmp_path, spec, eligible)
        ldir = SA.leaf_dir(tmp_path, benchmark, regime, seed, label)
        got = SA.load_split_indices(ldir, id_map, benchmark=benchmark, regime=regime, seed=seed, holdout=label)
        expected = {"train": train, "val": val, "test": test, "source_val": source_val, "source_test": source_test}
        for part, arr in expected.items():
            assert set(got[part].tolist()) == {int(i) for i in np.asarray(arr).tolist()}, \
                f"{benchmark}/{regime}/{label}/{part} membership diverged"
        labels_seen.append(str(label))
    return labels_seen


@pytest.mark.parametrize("regime", TABULAR_REGIMES)
def test_cropharvest_membership_parity_all_regimes(tmp_path, regime):
    bench = _ch_bench({"kenya": 20, "togo": 20, "ethiopia": 20, "lem-brazil": 20})
    labels_seen = _parity_for_regime(tmp_path, bench, ch, regime, seed=0)
    assert labels_seen, f"{regime} yielded no splits on the cropharvest fixture"


@pytest.mark.parametrize("regime", ["random_id", "geographic_ood", "spatial_cluster_ood"])
def test_eurocropsml_membership_parity(tmp_path, regime):
    from evals.benchmarks import eurocropsml as euro
    bench = _euro_bench({"Estonia": 24, "Latvia": 24, "Portugal": 24})
    labels_seen = _parity_for_regime(tmp_path, bench, euro, regime, seed=0)
    assert labels_seen, f"{regime} yielded no splits on the eurocropsml fixture"


def test_eurocropsml_official_metadata_parity(tmp_path):
    """Exercise official.iter_splits's EXACT-metadata branch via realistic bench.official_splits
    (explicit release-style row-index lists), not the make_strict_holdout fallback."""
    from evals.benchmarks import eurocropsml as euro
    bench = _euro_bench({"Estonia": 24, "Latvia": 24, "Portugal": 24})
    est = np.flatnonzero(bench.groups == "Estonia")
    lat = np.flatnonzero(bench.groups == "Latvia")
    por = np.flatnonzero(bench.groups == "Portugal")
    # disjoint, 2+-class train/test row-index lists per official holdout (leftover rows -> exclusions)
    bench.official_splits = {
        "latvia_vs_estonia": {
            "train": lat.tolist(), "val": por[:8].tolist(), "test": est.tolist(),
        },
        "latvia_portugal_vs_estonia": {
            "train": np.concatenate([lat, por[:8]]).tolist(),
            "val": por[8:16].tolist(), "test": est.tolist(),
        },
    }
    labels_seen = _parity_for_regime(tmp_path, bench, euro, "official", seed=0)
    assert labels_seen == list(euro.OFFICIAL_HOLDOUTS), labels_seen  # both metadata holdouts realized


@pytest.mark.parametrize("regime", ["random_id", "official", "geographic_ood", "spatial_cluster_ood"])
def test_breizhcrops_membership_parity(tmp_path, regime):
    from evals.benchmarks import breizhcrops as bz
    bench = _breizh_bench({"frh01": 20, "frh02": 20, "frh03": 20, "frh04": 20})
    labels_seen = _parity_for_regime(tmp_path, bench, bz, regime, seed=0)
    assert labels_seen, f"{regime} yielded no splits on the breizhcrops fixture"


def test_random_id_three_seeds_are_distinct_canonical_instances(tmp_path):
    bench = _ch_bench({"kenya": 30, "togo": 30, "ethiopia": 30})
    test_sets = []
    for seed in (0, 1, 2):
        _parity_for_regime(tmp_path, bench, ch, "random_id", seed)
        ldir = SA.leaf_dir(tmp_path, "cropharvest", "random_id", seed, "random_id")
        test_sets.append(tuple(SA.read_assignments(ldir)["test"]))
    # each seed is THE canonical split for its own seed -- three distinct instances, not a 3x3 grid
    assert len(set(test_sets)) == 3


def test_complete_accounting_holds_for_geographic_ood_with_purge(tmp_path):
    # co-locate a second domain on top of the target so the 50 km purge actually removes train rows
    bench = _ch_bench({"kenya": 20, "togo": 20, "ethiopia": 20, "lem-brazil": 20})
    # move ethiopia onto kenya so kenya-as-target purges ethiopia train rows
    eth = bench.groups == "ethiopia"
    bench.latlon[eth] = bench.latlon[bench.groups == "kenya"][0] + 1e-3
    y = bench.labels
    holdouts = regime_base.holdouts_for(ch, "geographic_ood")
    regime_base.clear_split_audit_events()
    events = regime_base.SPLIT_AUDIT_EVENTS
    prev = 0
    n_purge_events = 0
    for (label, train, val, test, domains, has_target, group_kind, source_val, source_test) in \
            regime_base.iter_splits("geographic_ood", bench, y, holdouts, 0):
        window = list(events[prev:len(events)])
        prev = len(events)
        n_purge_events += sum(1 for e in window if e.get("kind") == "purge")
        spec, eligible = SA.build_tabular_leaf(
            "cropharvest", "geographic_ood", 0, label=label,
            train=train, val=val, test=test, source_val=source_val, source_test=source_test,
            domains=domains, labels=y, sample_ids=bench.sample_ids, has_target=has_target,
            group_kind=group_kind, params={}, audit_events=window,
        )
        # publish validates complete accounting (assignments + exclusions == eligible, disjoint)
        SA.publish_leaf(tmp_path, spec, eligible)
        if any(e["reason"] == "purged_near_ood" for e in spec.exclusions):
            assert all(e["status"] == "proven" for e in spec.exclusions if e["reason"] == "purged_near_ood")
    assert n_purge_events >= 1, "expected the co-located fixture to trigger a purge"


# --------------------------------------------------------------------------- #
# Audit + generation.json coverage (dropped holdout recorded, membership unchanged)
# --------------------------------------------------------------------------- #
def _load_generator():
    path = Path(__file__).resolve().parents[2] / "tools" / "generate_splits.py"
    spec = importlib.util.spec_from_file_location("generate_splits_undertest", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_generation_records_dropped_holdout(tmp_path):
    gen = _load_generator()
    # 'central-asia' has n=5 < min_target_n=10 -> geographic_ood drops it (ineligible_target)
    bench = _ch_bench({"kenya": 20, "togo": 20, "ethiopia": 20, "central-asia": 5})
    leaves, summ = gen.generate_tabular(
        tmp_path, bench, ch, "geographic_ood", 0, audit_only=False, overwrite=True,
    )
    gen_json = json.loads((SA.regime_seed_dir(tmp_path, "cropharvest", "geographic_ood", 0) / "generation.json").read_text())
    dropped_labels = [d["label"] for d in gen_json["dropped_holdouts"]]
    assert "central-asia" in dropped_labels
    assert "central-asia" not in gen_json["yielded_holdouts"]
    assert "central-asia" in gen_json["requested_holdouts"]
    # the valid holdouts still produced leaves
    assert any(lf["holdout"] == "kenya" for lf in leaves)


def test_generation_records_requested_even_with_zero_yield(tmp_path):
    gen = _load_generator()
    # official holds out kenya/lem-brazil/togo BY NAME; a bench whose groups contain none of them
    # -> every official holdout drops -> ZERO leaves, but requested must still identify all three.
    bench = _ch_bench({"ethiopia": 20, "rwanda": 20, "central-asia": 20})
    leaves, _summ = gen.generate_tabular(tmp_path, bench, ch, "official", 0, audit_only=False, overwrite=True)
    gj = json.loads((SA.regime_seed_dir(tmp_path, "cropharvest", "official", 0) / "generation.json").read_text())
    assert leaves == []
    assert gj["yielded_holdouts"] == []
    assert set(gj["requested_holdouts"]) == set(ch.OFFICIAL_HOLDOUTS)  # kenya, lem-brazil, togo
    assert {d["label"] for d in gj["dropped_holdouts"]} == set(ch.OFFICIAL_HOLDOUTS)


def test_stratification_fallback_recorded_without_changing_membership():
    from evals.regimes import random_id as rid

    # a globally singleton class makes random_id's stratified split infeasible -> silent fallback
    y = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 2])  # classes 1 and 2 appear once each
    groups = np.array(["A"] * 10, dtype=object)
    regime_base.clear_split_audit_events()
    base_train, base_val, base_test = rid.make_splits(y, 0)  # same fallback path, for the baseline
    regime_base.clear_split_audit_events()
    splits = list(regime_base.iter_splits("random_id", SimpleNamespace(name="toy", groups=groups), y, None, 0))
    kinds = [e["kind"] for e in regime_base.SPLIT_AUDIT_EVENTS]
    assert "stratification_fallback" in kinds
    # membership unchanged vs calling make_splits directly (the event is behavior-neutral)
    _label, train, val, test, *_ = splits[0]
    assert set(train.tolist()) == set(base_train.tolist())
    assert set(test.tolist()) == set(base_test.tolist())


# --------------------------------------------------------------------------- #
# BreizhCrops: parcel `fid` is additive stable metadata (a real breizh load is ~53 min, so mock it)
# --------------------------------------------------------------------------- #
def test_breizhcrops_fid_is_additive_and_aligned(monkeypatch):
    from evals.benchmarks import breizhcrops as bz

    class _FakeDS:
        def __init__(self, region):
            self.region = region

        def __len__(self):
            return 3

        def __getitem__(self, i):
            x = np.full((4, len(bz.BZ_X_BANDS)), 0.1, dtype=np.float32)
            fid = 1000 * (ord(self.region[-1]) - ord("0")) + i  # unique within+across regions
            return x, i % 2, fid

    fake_pkg = SimpleNamespace(
        BreizhCrops=lambda region, root=None, level=None, load_timeseries=None, verbose=None: _FakeDS(region)
    )
    monkeypatch.setitem(sys.modules, "breizhcrops", fake_pkg)
    monkeypatch.setattr(bz, "BZ_REGIONS", ["frh01", "frh02"])
    monkeypatch.setattr(bz, "_bz_parcel_latlon", lambda ds: {})  # coords not needed here
    monkeypatch.setattr(bz, "_bz_class_names", lambda base: ["c0", "c1"])

    bench = bz.load_benchmark(root=Path("/tmp"), shuffle=True, seed=0)
    n = len(bench.labels)
    assert bench.sample_ids is not None
    assert len(bench.sample_ids) == n == len(bench.groups) == len(bench.latlon)
    assert len(set(bench.sample_ids.tolist())) == n, "stable ids must be unique"
    # additivity: each sample_id is aligned to the SAME shuffled row as groups/labels (region prefix)
    for sid, grp in zip(bench.sample_ids.tolist(), bench.groups.tolist(), strict=True):
        assert sid.startswith(f"{grp}:"), f"sample_id {sid!r} not aligned with group {grp!r}"


# --------------------------------------------------------------------------- #
# PASTIS: cache-free patch discovery == complete-cache discovery (no embedding cache needed)
# --------------------------------------------------------------------------- #
def _make_pastis(base: Path, patches):
    for d in ("DATA_S2", "DATA_S1A", "ANNOTATIONS"):
        (base / d).mkdir(parents=True, exist_ok=True)
    feats = []
    for pid, fold, lon, lat, classes in patches:
        feats.append({
            "type": "Feature",
            "properties": {
                "ID_PATCH": pid, "Fold": fold,
                "dates-S2": {"0": 20190115, "1": 20190215},
                "dates-S1A": {"0": 20190110, "1": 20190210},
            },
            "geometry": {"type": "Polygon", "coordinates": [[
                [lon, lat], [lon + 0.01, lat], [lon + 0.01, lat + 0.01], [lon, lat + 0.01], [lon, lat]
            ]]},
        })
        np.save(base / "DATA_S2" / f"S2_{pid}.npy", np.ones((2, 10, 128, 128), dtype=np.int16))
        np.save(base / "DATA_S1A" / f"S1A_{pid}.npy", np.ones((2, 3, 128, 128), dtype=np.float16))
        target = np.zeros((3, 128, 128), dtype=np.uint8)
        cls = list(classes)
        target[0, :, :] = cls[0]
        if len(cls) > 1:
            target[0, :64, :64] = cls[1]
        if len(cls) > 2:
            target[0, 64:, 64:] = cls[2]
        target[0, 0, 0] = 19  # an ignore pixel in every patch (must be excluded from class sets)
        np.save(base / "ANNOTATIONS" / f"TARGET_{pid}.npy", target)
    (base / "metadata.geojson").write_text(json.dumps({"type": "FeatureCollection", "features": feats}))


def _complete_cache(bench, cache_root: Path):
    for tile_id, fold, _tile, labels in bench.iter_tiles():
        fd = cache_root / f"fold_{fold}"
        fd.mkdir(parents=True, exist_ok=True)
        np.save(fd / f"{tile_id}.npy", np.zeros((max(len(labels), 1), 2), dtype=np.float32))
        np.save(fd / f"{tile_id}.labels.npy", np.asarray(labels, dtype=np.int64))


@pytest.fixture
def pastis_env(tmp_path):
    base = tmp_path / "pastis"
    patches = [
        (1001, 1, -1.0, 46.0, [0, 1]),
        (1002, 1, 2.0, 47.0, [0, 2]),
        (1003, 2, 5.0, 48.0, [1, 2]),
        (1004, 3, -3.0, 45.0, [0, 3]),
        (1005, 4, 8.0, 44.0, [0, 1, 2]),
        (1006, 5, -5.0, 49.0, [0, 1]),
    ]
    _make_pastis(base, patches)
    bench = get_input("pastis", root=tmp_path, shuffle=False)
    cache = tmp_path / "embcache"
    _complete_cache(bench, cache)
    return SimpleNamespace(bench=bench, cache=cache, tmp=tmp_path)


def test_pastis_patch_universe_is_cache_free_and_matches_complete_cache(pastis_env):
    bench, cache = pastis_env.bench, pastis_env.cache
    for folds in ({1}, {1, 2, 3}, {4}, {5}, {1, 2, 3, 4, 5}):
        cache_free = set(bench.patch_ids(folds))
        from_cache = set(cacheutils.dense_fold_patches(cache, folds))
        assert cache_free == from_cache, f"patch universe mismatch for folds {folds}"


def test_pastis_class_sets_are_cache_free_and_matches_complete_cache(pastis_env):
    bench, cache = pastis_env.bench, pastis_env.cache
    all_ids = bench.patch_ids(None)
    cache_free = bench.patch_class_sets(all_ids)
    from_cache = scood._patch_class_sets(cache, {1, 2, 3, 4, 5}, np.asarray(all_ids, dtype=np.int64))
    assert cache_free == from_cache
    assert all(19 not in s for s in cache_free.values()), "ignore index must not appear in class sets"


def test_pastis_official_dense_leaf_roundtrip_is_patch_level(pastis_env):
    bench = pastis_env.bench
    import evals.benchmarks.pastis as pastis_mod

    cfgs = list(regime_base.segmentation_fold_configs(pastis_mod, ["official"], seed=0, emb_dir=None, bench=bench))
    assert len(cfgs) == 1
    _regime, cfg = cfgs[0]
    dense_cache = dict(
        all_patch_ids=[int(p) for p in bench.patch_ids(None)],
        fold_of={int(p.patch_id): int(p.fold) for p in bench.patches},
        class_sets=bench.patch_class_sets(bench.patch_ids(None)),
        patch_latlon={int(k): v for k, v in bench.patch_latlon.items()},
    )
    spec, eligible = SA.build_dense_leaf(
        "pastis", "official", 0, cfg=cfg, bench=bench, params={"assembly_seed": 0},
        audit_events=[], **dense_cache,
    )
    ldir = SA.publish_leaf(pastis_env.tmp / "splits", spec, eligible)
    assigns = SA.read_assignments(ldir)
    # test fold is 5 -> only patch 1006
    assert assigns["test"] == ["1006"]
    # train folds 1,2,3 -> patches 1001..1004 ; val fold 4 -> 1005
    assert set(assigns["train"]) == {"1001", "1002", "1003", "1004"}
    assert assigns["val"] == ["1005"]
    assert SA.read_manifest(ldir)["target_unit"] == "patch"


def _resolve_dense_partitions(cfg, bench) -> dict[str, list[int]]:
    def r(explicit, folds):
        pids = explicit if explicit is not None else bench.patch_ids(set(folds))
        return sorted(int(p) for p in pids)
    return {
        "train": r(cfg.train_patches, cfg.train_folds),
        "val": r(cfg.val_patches, cfg.val_folds),
        "test": r(cfg.test_patches, cfg.test_folds),
        "source_val": sorted(int(p) for p in (cfg.source_val_patches or set())),
        "source_test": sorted(int(p) for p in (cfg.source_test_patches or set())),
    }


@pytest.mark.parametrize("regime", ["random_id", "official", "geographic_ood", "spatial_cluster_ood"])
def test_pastis_dense_all_regimes_parity_and_patch_level(pastis_env, regime):
    """Every supported PASTIS dense regime: cache-free split == complete-cache split, at PATCH level;
    then publish each realized leaf and reload it."""
    import evals.benchmarks.pastis as pastis_mod

    bench, cache = pastis_env.bench, pastis_env.cache
    free = list(regime_base.segmentation_fold_configs(pastis_mod, [regime], seed=0, emb_dir=None, bench=bench))
    cached = list(regime_base.segmentation_fold_configs(pastis_mod, [regime], seed=0, emb_dir=cache, bench=bench))
    # all four PASTIS dense regimes realize on this fixture -- assert so the parity below can't pass
    # vacuously on two empty lists
    assert free, f"PASTIS dense {regime} realized no fold configs on the fixture"
    assert [c.label for _, c in free] == [c.label for _, c in cached]

    dense_cache = dict(
        all_patch_ids=[int(p) for p in bench.patch_ids(None)],
        fold_of={int(p.patch_id): int(p.fold) for p in bench.patches},
        class_sets=bench.patch_class_sets(bench.patch_ids(None)),
        patch_latlon={int(k): v for k, v in bench.patch_latlon.items()},
    )
    root = pastis_env.tmp / f"splits_{regime}"
    published = 0
    for (_r, cf), (_c, cc) in zip(free, cached, strict=True):
        # cache-free patch membership equals complete-cache membership, per partition
        assert _resolve_dense_partitions(cf, bench) == _resolve_dense_partitions(cc, bench)
        spec, eligible = SA.build_dense_leaf(
            "pastis", regime, 0, cfg=cf, bench=bench, params={"assembly_seed": 0},
            audit_events=[], **dense_cache,
        )
        ldir = SA.publish_leaf(root, spec, eligible)
        published += 1
        assigns = SA.read_assignments(ldir)
        # dense assignments are patch IDs (all resolve to real patches)
        all_ids = {str(p) for p in dense_cache["all_patch_ids"]}
        for part in SA.PARTITIONS:
            assert set(assigns[part]).issubset(all_ids)
        # spatial_cluster is recorded with its own domain basis, not hardcoded geography
        if regime == "spatial_cluster_ood":
            assert SA.read_manifest(ldir)["domain_basis"] == "spatial_cluster"
    assert published == len(free) >= 1, f"expected {regime} to publish >=1 dense leaf"


def test_generator_dense_end_to_end_writes_leaves_and_generation(pastis_env):
    import evals.benchmarks.pastis as pastis_mod

    gen = _load_generator()
    root = pastis_env.tmp / "gensplits"
    leaves, _summ = gen.generate_dense(
        root, pastis_env.bench, pastis_mod, "official", 0,
        audit_only=False, overwrite=True, dense_cache=gen._dense_cache(pastis_env.bench),
    )
    assert leaves and leaves[0]["target_unit"] == "patch"
    gj = json.loads((SA.regime_seed_dir(root, "pastis", "official", 0) / "generation.json").read_text())
    assert gj["yielded_holdouts"] == ["fold_5"]
    assert SA.is_complete(SA.leaf_dir(root, "pastis", "official", 0, "fold_5"))


# --------------------------------------------------------------------------- #
# Phase B: runtime CONSUMPTION of canonical splits (integration, temp artifacts)
# --------------------------------------------------------------------------- #
def test_phase_b_tabular_consumption_and_hard_fails(tmp_path):
    gen = _load_generator()
    bench = _ch_bench({"kenya": 20, "togo": 20, "ethiopia": 20, "lem-brazil": 20})
    root = tmp_path / "splits"
    regimes = ["random_id", "geographic_ood"]
    for regime in regimes:
        gen.generate_tabular(root, bench, ch, regime, 0, audit_only=False, overwrite=True)

    specs, consumed = SA.load_tabular_splits(root, "cropharvest", bench.sample_ids, bench, ch, regimes, [0])
    assert specs and consumed and all(c.startswith("cropharvest/") for c in consumed)

    # consumed partitions must equal a direct runtime iter_splits run (parity through consumption)
    y, _g = ch.make_targets(bench)
    direct = {}
    for regime in regimes:
        for (label, tr, va, te, _dom, _ht, _gk, sv, st) in regime_base.iter_splits(
            regime, bench, y, regime_base.holdouts_for(ch, regime), 0, val_group=regime_base.val_group_for(ch, regime)
        ):
            direct[(regime, str(label))] = tuple(
                {int(i) for i in np.asarray(a).tolist()} for a in (tr, va, te, sv, st)
            )
    for (_seed, regime, label, tr, va, te, _dom, _ht, _db, sv, st) in specs:
        got = tuple({int(i) for i in a.tolist()} for a in (tr, va, te, sv, st))
        assert got == direct[(regime, label)], f"consumed {regime}/{label} diverged from runtime"

    # split_ref.json records the consumed regime-level paths + the scope note
    rdir = tmp_path / "results"
    rdir.mkdir()
    SA.write_split_ref(rdir, benchmark="cropharvest", consumed=consumed)
    ref = json.loads((rdir / "split_ref.json").read_text())
    assert ref["consumed_leaves"] == sorted(set(consumed))
    # canonical RELATIVE location + regime-partitions-only scope; no machine-specific absolute root
    assert ref["splits_location"] == "data/splits" and not ref["splits_location"].startswith("/")
    assert ref["scope"] == "regime_partitions_only"
    assert all(c.count("/") == 3 for c in ref["consumed_leaves"])  # <bench>/<regime>/<seed>/<holdout>
    note = ref["scope_note"].lower()
    assert "few-shot" in note and "not stored" in note and "target-budget" in note

    # hard-fail: benchmark not generated
    with pytest.raises(SA.SplitArtifactError, match="no canonical splits"):
        SA.load_tabular_splits(root, "eurocropsml", bench.sample_ids, bench, ch, ["random_id"], [0])
    # hard-fail: a declared-yielded leaf is incomplete
    victim = json.loads((SA.regime_seed_dir(root, "cropharvest", "geographic_ood", 0) / "generation.json").read_text())["yielded_holdouts"][0]
    (SA.leaf_dir(root, "cropharvest", "geographic_ood", 0, victim) / "manifest.json").unlink()
    with pytest.raises(SA.SplitArtifactError, match="incomplete or missing"):
        SA.load_tabular_splits(root, "cropharvest", bench.sample_ids, bench, ch, ["geographic_ood"], [0])
    # hard-fail: manifest identity disagrees with the canonical path
    gen.generate_tabular(root, bench, ch, "geographic_ood", 0, audit_only=False, overwrite=True)
    ldir = SA.leaf_dir(root, "cropharvest", "geographic_ood", 0, victim)
    m = SA.read_manifest(ldir)
    m["benchmark"] = "tampered"
    (ldir / "manifest.json").write_text(json.dumps(m))
    with pytest.raises(SA.SplitArtifactError, match="disagrees with canonical path"):
        SA.load_tabular_splits(root, "cropharvest", bench.sample_ids, bench, ch, ["geographic_ood"], [0])


def test_phase_b_dense_consumption_reconstructs_densesplit(pastis_env):
    import evals.benchmarks.pastis as pastis_mod

    gen = _load_generator()
    root = pastis_env.tmp / "densesplits"
    dc = gen._dense_cache(pastis_env.bench)
    for regime in ["random_id", "official"]:
        gen.generate_dense(root, pastis_env.bench, pastis_mod, regime, 0, audit_only=False, overwrite=True, dense_cache=dc)

    patch_fold = {int(p.patch_id): int(p.fold) for p in pastis_env.bench.patches}
    by_seed, consumed = SA.load_dense_splits(root, "pastis", patch_fold, ["random_id", "official"], [0])
    assert by_seed[0] and consumed
    off = next(cfg for r, cfg in by_seed[0] if r == "official")
    # DenseSplit reconstructed at patch level: fold 5 -> patch 1006
    assert off.test_folds == {5} and off.test_patches == {1006}
    assert off.train_folds == {1, 2, 3} and off.val_folds == {4}
    assert off.has_target is True

    with pytest.raises(SA.SplitArtifactError, match="no canonical splits"):
        SA.load_dense_splits(pastis_env.tmp / "nope", "pastis", patch_fold, ["random_id"], [0])


def test_phase_b_domains_come_from_artifact_not_runtime_assign_domains(tmp_path, monkeypatch):
    """Req 3: tabular consumption reconstructs domains from the artifact and never calls
    assign_domains/KMeans -- spatial_cluster_ood loads fine even if assign_domains is made to raise."""
    gen = _load_generator()
    bench = _ch_bench({"kenya": 20, "togo": 20, "ethiopia": 20, "lem-brazil": 20})
    root = tmp_path / "splits"
    gen.generate_tabular(root, bench, ch, "spatial_cluster_ood", 0, audit_only=False, overwrite=True)

    import evals.regimes.spatial_cluster_ood as sc

    def _boom(*_a, **_k):
        raise RuntimeError("assign_domains must NOT be called during consumption")
    monkeypatch.setattr(sc, "assign_domains", _boom)

    specs, _consumed = SA.load_tabular_splits(root, "cropharvest", bench.sample_ids, bench, ch, ["spatial_cluster_ood"], [0])
    assert specs
    for (_s, _r, _l, _tr, _va, te, dom, _ht, _db, _sv, _st) in specs:
        for i in te.tolist():  # every test sample has its realized cluster domain from the artifact
            assert dom[i] != "__unassigned__"


def test_phase_b_refuses_regime_that_recorded_problems(tmp_path):
    """Req 4: a non-empty regime_problems in generation.json is a hard refusal."""
    gen = _load_generator()
    bench = _ch_bench({"kenya": 20, "togo": 20, "ethiopia": 20})
    root = tmp_path / "splits"
    gen.generate_tabular(root, bench, ch, "random_id", 0, audit_only=False, overwrite=True)
    gp = SA.regime_seed_dir(root, "cropharvest", "random_id", 0) / "generation.json"
    g = json.loads(gp.read_text())
    g["regime_problems"] = [["cropharvest", "random_id", "synthetic problem"]]
    gp.write_text(json.dumps(g))
    with pytest.raises(SA.SplitArtifactError, match="regime problem"):
        SA.load_tabular_splits(root, "cropharvest", bench.sample_ids, bench, ch, ["random_id"], [0])


def test_phase_b_refuses_zero_yield_requested_regime(tmp_path):
    """Req 4: a requested regime that yielded zero leaves is a hard refusal (isolated from the
    regime_problem path -- a clean generation.json that simply produced nothing)."""
    gen = _load_generator()
    bench = _ch_bench({"kenya": 20, "togo": 20})
    root = tmp_path / "splits"
    gen.generate_tabular(root, bench, ch, "random_id", 0, audit_only=False, overwrite=True)  # a real, valid cell
    # a CLEAN (no regime_problems) but empty official cell
    d = SA.regime_seed_dir(root, "cropharvest", "official", 0)
    d.mkdir(parents=True, exist_ok=True)
    (d / "generation.json").write_text(json.dumps({
        "schema_version": 1, "benchmark": "cropharvest", "regime": "official", "seed": 0,
        "requested_holdouts": ["kenya"], "yielded_holdouts": [], "dropped_holdouts": [],
        "audit_events": [], "regime_problems": [],
    }))
    with pytest.raises(SA.SplitArtifactError, match="zero leaves"):
        SA.load_tabular_splits(root, "cropharvest", bench.sample_ids, bench, ch, ["official"], [0])


def test_phase_b_dense_refuses_changed_patch_fold_membership(pastis_env):
    """Req 5: an assigned patch whose current fold is not in its partition's fold set is refused."""
    import evals.benchmarks.pastis as pastis_mod

    gen = _load_generator()
    root = pastis_env.tmp / "densesplits_fold"
    gen.generate_dense(root, pastis_env.bench, pastis_mod, "official", 0,
                       audit_only=False, overwrite=True, dense_cache=gen._dense_cache(pastis_env.bench))
    patch_fold = {int(p.patch_id): int(p.fold) for p in pastis_env.bench.patches}
    patch_fold[1006] = 1  # was fold 5 (test); now claims fold 1, not in test_folds {5}
    with pytest.raises(SA.SplitArtifactError, match="patch-fold membership changed"):
        SA.load_dense_splits(root, "pastis", patch_fold, ["official"], [0])


def test_phase_b_run_manifest_binds_consumed_split_paths(tmp_path):
    """Req 2: the consumed leaf-path set is bound to resume identity -- a changed set, or an existing
    manifest lacking the field, refuses resume (not backfilled); overwrite bypasses."""
    from utils import runstate as RS

    base = {
        "schema": 1, "benchmark": "cropharvest", "model": "raw", "seeds": [0], "regimes": ["random_id"],
        "consumed_splits": {"scope": "regime_partitions_only", "leaves": ["cropharvest/random_id/0/random_id"]},
    }
    RS.publish_run_manifest(tmp_path, base)
    RS.check_run_manifest(tmp_path, base, overwrite_mode=False)  # exact match -> resume ok

    changed = {**base, "consumed_splits": {"scope": "regime_partitions_only",
               "leaves": ["cropharvest/random_id/0/random_id", "cropharvest/official/0/kenya"]}}
    with pytest.raises(RuntimeError, match="consumed_splits"):
        RS.check_run_manifest(tmp_path, changed, overwrite_mode=False)

    # existing manifest LACKS the field -> a run WITH the field is refused, never backfilled
    RS.publish_run_manifest(tmp_path, {k: v for k, v in base.items() if k != "consumed_splits"})
    with pytest.raises(RuntimeError, match="consumed_splits"):
        RS.check_run_manifest(tmp_path, base, overwrite_mode=False)

    RS.check_run_manifest(tmp_path, changed, overwrite_mode=True)  # overwrite bypasses
