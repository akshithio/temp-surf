"""Schema-v2 split round-trip: a frozen artifact reloads to exactly what the regime emitted.

The acceptance criterion. For every realizable (benchmark, regime, seed, holdout), a regime emits an
explicit :class:`~evals.regimes.base.SourceTargetSplit` / ``DenseSourceTargetSplit``; writing it to a
frozen ``assignments.csv`` + the central ``data/logs/splits.json`` and reloading it (with checksum +
complete-accounting checks) must reproduce every one of the five explicit partitions exactly. Also
proves PASTIS split construction needs no embedding cache (the regime reads ``bench.patch_ids`` /
``patch_class_sets``, never the cache) and drives the generator + runtime consumption end to end.

All four regimes are schema v2. random_id / geographic_ood parity is exercised here on the shared
fixtures; official has its own provenance-shaped bench (test_regime_official.py); spatial_cluster_ood
(coordinate-only cells) has its own fixtures in test_regime_spatial.py.

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
from tests import splitfix
from utils import cacheutils

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
    """Freeze every EMITTED schema-v2 split (CSV + central log), reload it, and assert that every
    explicit partition's membership reloads exactly as the regime emitted it (the v2 round-trip).
    Returns the emitted holdout labels in emission order."""
    root = Path(tmp_path) / "splits"
    regime_mod = regime_base.load_regime(regime)
    y, _groups = bench_mod.make_targets(bench)
    domains = np.asarray(regime_mod.sample_domains(bench, bench_mod), dtype=object)
    benchmark = bench_mod.BENCHMARK

    built, emitted, order = [], {}, []
    for split in regime_mod.iter_source_target_splits(bench, bench_mod, seed):
        built.append(SA.build_tabular_leaf(
            benchmark, regime, seed, split=split, domains=domains,
            labels=y, sample_ids=bench.sample_ids, audit_events=[],
        ))
        emitted[str(split.label)] = split
        order.append(str(split.label))
    splitfix.freeze(root, built)

    for ls in SA.load_tabular_splits(root, benchmark, bench.sample_ids, [regime], [seed]):
        split = emitted[ls.split.label]
        for part, arr in split.as_partitions().items():
            assert set(getattr(ls.split, part).tolist()) == {int(i) for i in np.asarray(arr).tolist()}, \
                f"{benchmark}/{regime}/{ls.split.label}/{part} membership diverged"
    return order


# official is excluded here: CropHarvest official is Togo-only over the un-merged togo/togo-eval
# PROVENANCE (not the canonical geographic groups this generic fixture carries), so it needs a
# provenance-shaped bench -- exercised in test_regime_official.py instead.
@pytest.mark.parametrize("regime", ["random_id", "geographic_ood"])
def test_cropharvest_membership_parity_all_regimes(tmp_path, regime):
    bench = _ch_bench({"kenya": 20, "togo": 20, "ethiopia": 20, "lem-brazil": 20})
    labels_seen = _parity_for_regime(tmp_path, bench, ch, regime, seed=0)
    assert labels_seen, f"{regime} yielded no splits on the cropharvest fixture"


@pytest.mark.parametrize("regime", ["random_id", "geographic_ood"])
def test_eurocropsml_membership_parity(tmp_path, regime):
    from evals.benchmarks import eurocropsml as euro
    bench = _euro_bench({"Estonia": 24, "Latvia": 24, "Portugal": 24})
    labels_seen = _parity_for_regime(tmp_path, bench, euro, regime, seed=0)
    assert labels_seen, f"{regime} yielded no splits on the eurocropsml fixture"


def test_eurocropsml_official_metadata_parity(tmp_path):
    """Exercise official.iter_source_target_splits's EXACT-metadata branch via realistic
    bench.official_splits (explicit release-style row-index lists)."""
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


@pytest.mark.parametrize("regime", ["random_id", "official", "geographic_ood"])
def test_breizhcrops_membership_parity(tmp_path, regime):
    from evals.benchmarks import breizhcrops as bz
    bench = _breizh_bench({"frh01": 20, "frh02": 20, "frh03": 20, "frh04": 20})
    labels_seen = _parity_for_regime(tmp_path, bench, bz, regime, seed=0)
    assert labels_seen, f"{regime} yielded no splits on the breizhcrops fixture"


def test_random_id_three_seeds_are_distinct_canonical_instances(tmp_path):
    bench = _ch_bench({"kenya": 30, "togo": 30, "ethiopia": 30})
    root = tmp_path / "splits"
    test_sets = []
    for seed in (0, 1, 2):
        _parity_for_regime(tmp_path, bench, ch, "random_id", seed)
        csv_rows = SA.read_assignments_csv(SA.assignments_path(root, "cropharvest", "random_id", seed, "random_id"))
        # v2: source_test is random_id's in-distribution evaluation partition
        test_sets.append(tuple(sorted(r["stable_id"] for r in csv_rows if r["partition"] == "source_test")))
    # each seed is THE canonical split for its own seed -- three distinct instances, not a 3x3 grid
    assert len(set(test_sets)) == 3


def test_complete_accounting_holds_for_geographic_ood_with_purge(tmp_path):
    from evals.regimes import geographic_ood as geo

    # co-locate a second domain on top of the target so the 50 km purge actually removes source rows
    bench = _ch_bench({"kenya": 20, "togo": 20, "ethiopia": 20, "lem-brazil": 20})
    bench.latlon[bench.groups == "ethiopia"] = bench.latlon[bench.groups == "kenya"][0] + 1e-3
    y = bench.labels
    domains = geo.sample_domains(bench, ch)
    regime_base.clear_split_audit_events()
    events = regime_base.SPLIT_AUDIT_EVENTS
    prev = 0
    n_purge_events, total_eligible = 0, len(bench.sample_ids)
    for split in geo.iter_source_target_splits(bench, ch, 0):
        window = list(events[prev:len(events)])
        prev = len(events)
        n_purge_events += sum(1 for e in window if e.get("kind") == "purge")
        rows, summary = SA.build_tabular_leaf(
            "cropharvest", "geographic_ood", 0, split=split, domains=domains,
            labels=y, sample_ids=bench.sample_ids, audit_events=window, purge_km=50.0,
        )
        # complete accounting: every eligible sample appears exactly once, tagged assigned/purged/excluded
        assert len(rows) == total_eligible == len({r["stable_id"] for r in rows})
        purged = [r for r in rows if r["status"] == SA.STATUS_PURGED]
        assert all(r["reason"] == "purged_near_ood" and not r["partition"] for r in purged)
        assert summary["purge_count"] == len(purged)
    assert n_purge_events >= 1, "expected the co-located fixture to trigger a purge"


def test_random_id_v2_split_is_exact_with_no_fallback():
    """The v2 replacement for the retired silent-fallback behavior: a class distribution that the old
    proportional ``train_test_split`` could not stratify now yields an EXACT-size constrained split
    with NO unstratified fallback and NO audit event (``partition_source`` raises on a genuinely
    infeasible split rather than degrading)."""
    from evals import split_spec
    from evals.regimes import random_id as rid

    y = np.array(([0] * 8 + [1] * 2) * 3, dtype=np.int64)  # 30 rows; class 1 rare (6 total)
    groups = np.array(["A"] * 30, dtype=object)
    bench = SimpleNamespace(labels=y, groups=groups, sample_ids=np.array([f"s{i}" for i in range(30)], dtype=object))
    bench_mod = SimpleNamespace(BENCHMARK="toy", make_targets=lambda b: (b.labels, b.groups))

    regime_base.clear_split_audit_events()
    split = next(iter(rid.iter_source_target_splits(bench, bench_mod, 0)))
    tr, va, te = split_spec.source_partition_sizes(30)
    assert (len(split.source_train), len(split.source_val), len(split.source_test)) == (tr, va, te)
    # exact partition of the whole population, and NOT a single stratification_fallback was emitted
    everything = np.concatenate([split.source_train, split.source_val, split.source_test])
    assert set(everything.tolist()) == set(range(30))
    assert not any(e["kind"] == "stratification_fallback" for e in regime_base.SPLIT_AUDIT_EVENTS)


# --------------------------------------------------------------------------- #
# BreizhCrops: parcel `fid` is additive stable metadata (a real breizh load is ~53 min, so mock it)
# --------------------------------------------------------------------------- #
def test_breizhcrops_fid_is_additive_and_aligned(monkeypatch):
    from evals.benchmarks import breizhcrops as bz

    class _FakeDS:
        def __init__(self, region, transform):
            self.region = region
            self.transform = transform

        def __len__(self):
            return 3

        def __getitem__(self, i):
            x = np.full((4, len(bz.BZ_X_BANDS)), 0.1, dtype=np.float32)
            fid = 1000 * (ord(self.region[-1]) - ord("0")) + i  # unique within+across regions
            return self.transform(x), i % 2, fid

    transforms = []

    def fake_breizhcrops(
        region, root=None, level=None, transform=None, load_timeseries=None, verbose=None
    ):
        transforms.append(transform)
        return _FakeDS(region, transform)

    fake_pkg = SimpleNamespace(
        BreizhCrops=fake_breizhcrops
    )
    monkeypatch.setitem(sys.modules, "breizhcrops", fake_pkg)
    monkeypatch.setattr(bz, "BZ_REGIONS", ["frh01", "frh02"])
    monkeypatch.setattr(bz, "_bz_parcel_latlon", lambda ds: {})  # coords not needed here
    monkeypatch.setattr(bz, "_bz_class_names", lambda base: ["c0", "c1"])

    bench = bz.load_benchmark(root=Path("/tmp"), shuffle=True, seed=0)
    n = len(bench.labels)
    assert bench.sample_ids is not None
    assert all(transform is bz._identity_timeseries for transform in transforms)
    assert all(values.shape == (4, len(bz.BZ_X_BANDS) + 1) for values in bench.native.s2.values)
    np.testing.assert_allclose(bench.native.s2.values[0][..., :-1], 0.1)
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


def _cache_class_sets(emb_dir, folds, patch_ids) -> dict[int, set[int]]:
    """Per-patch class sets read from the cached label tiles -- the cache-based reference that
    cache-free ``bench.patch_class_sets`` is checked against (no regime code needed)."""
    out: dict[int, set[int]] = {int(pid): set() for pid in patch_ids}
    for path in cacheutils._dense_label_paths(emb_dir, set(folds), set(out)):
        pid = int(path.name.split("_", 1)[0])
        out[pid].update(int(v) for v in np.unique(np.load(path, mmap_mode="r")))
    return out


def test_pastis_class_sets_are_cache_free_and_matches_complete_cache(pastis_env):
    bench, cache = pastis_env.bench, pastis_env.cache
    all_ids = bench.patch_ids(None)
    cache_free = bench.patch_class_sets(all_ids)
    from_cache = _cache_class_sets(cache, {1, 2, 3, 4, 5}, np.asarray(all_ids, dtype=np.int64))
    assert cache_free == from_cache
    assert all(19 not in s for s in cache_free.values()), "ignore index must not appear in class sets"


def test_pastis_official_dense_leaf_roundtrip_is_patch_level(pastis_env):
    import evals.benchmarks.pastis as pastis_mod
    from evals.regimes import official

    bench = pastis_env.bench
    root = pastis_env.tmp / "splits"
    dsplit = next(iter(official.iter_dense_source_target_splits(bench, pastis_mod, 0)))
    all_pids = [int(p) for p in bench.patch_ids(None)]
    domain_of = {int(k): str(v) for k, v in official.patch_domains(bench, pastis_mod).items()}
    rows, summary = SA.build_dense_leaf(
        "pastis", "official", 0, dense_split=dsplit, audit_events=[],
        all_patch_ids=all_pids, domain_of=domain_of,
        class_sets={int(k): set(v) for k, v in bench.patch_class_sets(all_pids).items()},
        patch_latlon={int(k): v for k, v in bench.patch_latlon.items()},
    )
    splitfix.freeze(root, [(rows, summary)])
    by_part: dict[str, set[str]] = {}
    for r in rows:
        if r["status"] == SA.STATUS_ASSIGNED:
            by_part.setdefault(r["partition"], set()).add(r["stable_id"])
    # folds 1-3 -> source_train (1001..1004); fold 4 -> source_val (1005); fold 5 -> target_test (1006)
    assert by_part.get("source_train") == {"1001", "1002", "1003", "1004"}
    assert by_part.get("source_val") == {"1005"}
    assert by_part.get("target_test") == {"1006"}
    assert "source_test" not in by_part and "target_label_pool" not in by_part
    assert summary["target_unit"] == "patch"
    assert summary["has_target"] is True and summary["supports_target_labels"] is False


# geographic_ood (tile-LODO) needs real Sentinel tiles, which this synthetic pastis_env fixture does
# not carry (tile=None); it is exercised in test_regime_geographic.py / test_evals.py instead.
@pytest.mark.parametrize("regime", ["random_id", "official"])
def test_pastis_dense_all_regimes_parity_and_patch_level(pastis_env, regime):
    """Each migrated PASTIS dense regime emits patch-level v2 splits (cache-free by construction --
    the regime reads bench.patch_ids/patch_class_sets, never the embedding cache); freeze and reload."""
    import evals.benchmarks.pastis as pastis_mod

    bench = pastis_env.bench
    regime_mod = regime_base.load_regime(regime)
    all_ids = [int(p) for p in bench.patch_ids(None)]
    domain_of = {int(k): str(v) for k, v in regime_mod.patch_domains(bench, pastis_mod).items()}
    class_sets = {int(k): set(v) for k, v in bench.patch_class_sets(all_ids).items()}
    patch_latlon = {int(k): v for k, v in bench.patch_latlon.items()}
    root = pastis_env.tmp / f"splits_{regime}"
    all_ids_s = {str(p) for p in all_ids}
    built = []
    for dsplit in regime_mod.iter_dense_source_target_splits(bench, pastis_mod, 0):
        rows, summary = SA.build_dense_leaf(
            "pastis", regime, 0, dense_split=dsplit, audit_events=[], all_patch_ids=all_ids,
            domain_of=domain_of, class_sets=class_sets, patch_latlon=patch_latlon,
        )
        assert summary["target_unit"] == "patch"
        assert {r["stable_id"] for r in rows if r["status"] == SA.STATUS_ASSIGNED}.issubset(all_ids_s)
        built.append((rows, summary))
    splitfix.freeze(root, built)
    assert built, f"expected {regime} to build >=1 dense leaf"


# --------------------------------------------------------------------------- #
# Generator: build assignments.csv leaves + the one central log (tools/generate_splits.py)
# --------------------------------------------------------------------------- #
def _load_generator():
    path = Path(__file__).resolve().parents[2] / "tools" / "generate_splits.py"
    spec = importlib.util.spec_from_file_location("generate_splits_undertest", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _gen_tabular(gen, root, bench, bench_mod, regime, seed):
    """Drive the generator for one tabular (regime, seed) AND write the central log (the log is a
    single file the runtime discovers leaves from; main() writes it once after all benchmarks)."""
    entries = gen.generate_tabular(root, bench, bench_mod, regime, seed, audit_only=False)
    SA.write_splits_log(SA.default_log_path(root), provenance=gen.build_provenance(), entries=entries)
    return entries


def _gen_dense(gen, root, bench, bench_mod, regime, seed):
    entries = gen.generate_dense(root, bench, bench_mod, regime, seed, audit_only=False, dense_cache=gen._dense_cache(bench))
    SA.write_splits_log(SA.default_log_path(root), provenance=gen.build_provenance(), entries=entries)
    return entries


def test_generator_rejects_a_missing_holdout(tmp_path):
    """A configured LODO target absent from the data (a dropped holdout) FAILS generation -- the frozen
    split set must be exactly complete, never silently thinned."""
    gen = _load_generator()
    bench = _ch_bench({"kenya": 20, "togo": 20, "lem-brazil": 20})  # most geographic targets absent
    with pytest.raises(SA.SplitArtifactError, match="missing"):
        gen.generate_tabular(tmp_path / "splits", bench, ch, "geographic_ood", 0, audit_only=True)


def test_generator_rejects_a_duplicate_holdout(tmp_path, monkeypatch):
    """Two leaves for the same holdout fail generation (one leaf per expected holdout)."""
    from evals.regimes import random_id
    gen = _load_generator()
    bench = _ch_bench({"kenya": 20, "togo": 20})
    real = next(iter(random_id.iter_source_target_splits(bench, ch, 0)))
    monkeypatch.setattr(random_id, "iter_source_target_splits", lambda *a, **k: iter([real, real]))
    with pytest.raises(SA.SplitArtifactError, match="duplicate"):
        gen.generate_tabular(tmp_path / "splits", bench, ch, "random_id", 0, audit_only=True)


def test_generator_rejects_an_unexpected_holdout(tmp_path, monkeypatch):
    """A realized holdout not in the expected set fails generation."""
    from evals.regimes import base as RB
    from evals.regimes import random_id
    gen = _load_generator()
    bench = _ch_bench({"kenya": 20, "togo": 20})
    real = next(iter(random_id.iter_source_target_splits(bench, ch, 0)))
    bonus = RB.SourceTargetSplit(
        label="atlantis", source_train=real.source_train, source_val=real.source_val,
        source_test=real.source_test, has_target=False, supports_target_labels=False,
    )
    monkeypatch.setattr(random_id, "iter_source_target_splits", lambda *a, **k: iter([real, bonus]))
    with pytest.raises(SA.SplitArtifactError, match="unexpected"):
        gen.generate_tabular(tmp_path / "splits", bench, ch, "random_id", 0, audit_only=True)


def test_generator_tabular_end_to_end_writes_leaf_and_log(tmp_path):
    gen = _load_generator()
    root = tmp_path / "splits"
    bench = _ch_bench({"kenya": 20, "togo": 20, "ethiopia": 20, "lem-brazil": 20})
    entries = _gen_tabular(gen, root, bench, ch, "random_id", 0)
    assert entries and entries[0]["target_unit"] == "sample" and entries[0]["holdout"] == "random_id"
    assert len(entries[0]["sha256"]) == 64
    assert SA.assignments_path(root, "cropharvest", "random_id", 0, "random_id").is_file()
    log = SA.read_splits_log(SA.default_log_path(root))
    assert "generation_timestamp" in log and log["run_seeds"] == [0, 1, 2] and log["leaves"]


def test_generator_dense_end_to_end_writes_leaf_and_log(pastis_env):
    import evals.benchmarks.pastis as pastis_mod

    gen = _load_generator()
    root = pastis_env.tmp / "gensplits"
    entries = _gen_dense(gen, root, pastis_env.bench, pastis_mod, "random_id", 0)
    assert entries and entries[0]["target_unit"] == "patch" and entries[0]["holdout"] == "random_patch"
    assert SA.assignments_path(root, "pastis", "random_id", 0, "random_patch").is_file()


def test_generator_official_tabular_end_to_end(tmp_path):
    from evals.benchmarks import eurocropsml as euro
    gen = _load_generator()
    root = tmp_path / "splits"
    bench = _euro_bench({"Estonia": 24, "Latvia": 24, "Portugal": 24})
    est = np.flatnonzero(bench.groups == "Estonia")
    lat = np.flatnonzero(bench.groups == "Latvia")
    por = np.flatnonzero(bench.groups == "Portugal")
    bench.official_splits = {
        "latvia_vs_estonia": {"train": lat.tolist(), "val": por[:8].tolist(), "test": est.tolist(),
                              "target_train": est[:5].tolist()},
        "latvia_portugal_vs_estonia": {"train": np.concatenate([lat, por[:8]]).tolist(),
                                       "val": por[8:16].tolist(), "test": est.tolist(),
                                       "target_train": est[:5].tolist()},
    }
    entries = _gen_tabular(gen, root, bench, euro, "official", 0)
    assert [e["holdout"] for e in entries] == list(euro.OFFICIAL_HOLDOUTS)
    assert all(e["target_unit"] == "sample" for e in entries)
    for anchor in euro.OFFICIAL_HOLDOUTS:
        assert SA.assignments_path(root, "eurocropsml", "official", 0, anchor).is_file()


# --------------------------------------------------------------------------- #
# Phase B: runtime CONSUMPTION of frozen leaves (discovery + checksum from the central log)
# --------------------------------------------------------------------------- #
def test_phase_b_tabular_consumption_and_hard_fails(tmp_path):
    from evals.regimes import random_id

    gen = _load_generator()
    bench = _ch_bench({"kenya": 20, "togo": 20, "ethiopia": 20, "lem-brazil": 20})
    root = tmp_path / "splits"
    _gen_tabular(gen, root, bench, ch, "random_id", 0)

    loaded = SA.load_tabular_splits(root, "cropharvest", bench.sample_ids, ["random_id"], [0])
    assert loaded
    # consumed partitions must equal what the regime emitted (parity through consumption)
    emitted = next(iter(random_id.iter_source_target_splits(bench, ch, 0)))
    ls = loaded[0]
    for part, arr in emitted.as_partitions().items():
        assert set(getattr(ls.split, part).tolist()) == {int(i) for i in np.asarray(arr).tolist()}, \
            f"consumed random_id/{part} diverged from the emitted split"

    # hard-fail: a benchmark with no leaves in the log
    with pytest.raises(SA.SplitArtifactError, match="zero leaves"):
        SA.load_tabular_splits(root, "eurocropsml", bench.sample_ids, ["random_id"], [0])
    # hard-fail: the frozen CSV changed after the log recorded its checksum
    csv_path = SA.assignments_path(root, "cropharvest", "random_id", 0, "random_id")
    csv_path.write_bytes(csv_path.read_bytes() + b"tampered_extra_row,source_train,assigned,d,\n")
    with pytest.raises(SA.SplitArtifactError, match="checksum mismatch"):
        SA.load_tabular_splits(root, "cropharvest", bench.sample_ids, ["random_id"], [0])
    # hard-fail: no log at all (a splits root whose parent has no logs/ sibling)
    with pytest.raises(SA.SplitArtifactError, match="no split log"):
        SA.load_tabular_splits(tmp_path / "elsewhere" / "splits", "cropharvest", bench.sample_ids, ["random_id"], [0])


def test_phase_b_dense_consumption_reconstructs_dense_split(pastis_env):
    import evals.benchmarks.pastis as pastis_mod

    gen = _load_generator()
    root = pastis_env.tmp / "densesplits"
    _gen_dense(gen, root, pastis_env.bench, pastis_mod, "random_id", 0)

    patch_fold = {int(p.patch_id): int(p.fold) for p in pastis_env.bench.patches}
    patch_tile = {int(k): v for k, v in pastis_env.bench.patch_tiles.items()}
    by_seed = SA.load_dense_splits(root, "pastis", patch_fold, patch_tile, ["random_id"], [0])
    assert by_seed[0]
    ld = by_seed[0][0]
    assert ld.regime == "random_id"
    d = ld.split
    # DenseSourceTargetSplit reconstructed at PATCH level: source-only, disjoint + complete over all patches
    assert d.has_target is False and d.supports_target_labels is False
    all_patches = {int(p) for p in pastis_env.bench.patch_ids(None)}
    assert d.source_train_patches | d.source_val_patches | d.source_test_patches == all_patches
    assert not d.target_label_pool_patches and not d.target_test_patches

    with pytest.raises(SA.SplitArtifactError, match="no split log"):
        SA.load_dense_splits(pastis_env.tmp / "elsewhere" / "splits", "pastis", patch_fold, patch_tile, ["random_id"], [0])


def test_phase_b_domains_come_from_artifact_not_runtime_kmeans(tmp_path, monkeypatch):
    """Tabular consumption reconstructs the per-sample domain from the CSV and never rebuilds the
    spatial cells -- spatial_cluster_ood loads fine even if the KMeans cell construction is patched to
    raise (the runtime must NEVER re-run KMeans)."""
    gen = _load_generator()
    # five far-apart domains -> five clean coordinate cells
    bench = _ch_bench({"kenya": 20, "togo": 20, "ethiopia": 20, "lem-brazil": 20, "central-asia": 20})
    root = tmp_path / "splits"
    _gen_tabular(gen, root, bench, ch, "spatial_cluster_ood", 0)

    import evals.regimes.spatial_cluster_ood as sc

    def _boom(*_a, **_k):
        raise RuntimeError("spatial cell construction (KMeans) must NOT be called during consumption")
    monkeypatch.setattr(sc, "_cell_labels", _boom)

    loaded = SA.load_tabular_splits(root, "cropharvest", bench.sample_ids, ["spatial_cluster_ood"], [0])
    assert loaded
    for lt in loaded:
        for i in lt.split.target_test.tolist():  # every test sample carries its realized cluster from the CSV
            assert lt.domains[i] != "__unassigned__"


def test_phase_b_refuses_zero_yield_requested_regime(tmp_path):
    """A requested regime that has no leaves in the central log is a hard refusal."""
    gen = _load_generator()
    bench = _ch_bench({"kenya": 20, "togo": 20})
    root = tmp_path / "splits"
    _gen_tabular(gen, root, bench, ch, "random_id", 0)  # the log holds only random_id leaves
    with pytest.raises(SA.SplitArtifactError, match="zero leaves"):
        SA.load_tabular_splits(root, "cropharvest", bench.sample_ids, ["official"], [0])


def test_phase_b_dense_refuses_changed_patch_fold_membership(pastis_env):
    """official dense: an assigned patch whose CURRENT published fold no longer matches the fold
    frozen as its domain is refused (the release folds must not have shifted)."""
    import evals.benchmarks.pastis as pastis_mod

    gen = _load_generator()
    root = pastis_env.tmp / "densesplits_fold"
    _gen_dense(gen, root, pastis_env.bench, pastis_mod, "official", 0)
    patch_fold = {int(p.patch_id): int(p.fold) for p in pastis_env.bench.patches}
    patch_tile = {int(k): v for k, v in pastis_env.bench.patch_tiles.items()}
    patch_fold[1006] = 1  # was fold 5 (target_test); now claims fold 1 -> published fold shifted
    with pytest.raises(SA.SplitArtifactError, match="structural metadata changed"):
        SA.load_dense_splits(root, "pastis", patch_fold, patch_tile, ["official"], [0])
