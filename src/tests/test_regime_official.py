"""Schema-v2 official regime: each benchmark's exact release split, has_target=True /
supports_target_labels=False (target geography, NO target-label access -> target_label_pool always
empty, zero-shot on target_test).

Per-benchmark exact definitions + the seed contract (item 7): EuroCropsML anchors, BreizhCrops
regions, and PASTIS folds are FIXED across seeds; only the CropHarvest Togo source train/val
subdivision varies by seed (with sizes and class marginals held identical). No real data, no model.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from evals import split_artifacts as SA
from evals import split_spec
from evals.benchmarks import breizhcrops as bz
from evals.benchmarks import cropharvest as ch
from evals.benchmarks import eurocropsml as euro
from evals.benchmarks import pastis as pastis_mod
from evals.regimes import base as RB
from evals.regimes import official
from tests import splitfix


# --------------------------------------------------------------------------- #
# Fixtures -- minimal benches the real benchmark modules' official metadata drives
# --------------------------------------------------------------------------- #
def _ch_official_bench():
    """CropHarvest: 100 `togo` source (60 crop / 40 non-crop) + 30 `togo-eval` target (20/10)."""
    sids, y, groups = [], [], []
    i = 0
    for prov, n_crop, n_non in (("togo", 60, 40), ("togo-eval", 20, 10)):
        for cls, count in ((1, n_crop), (0, n_non)):
            for _ in range(count):
                sids.append(f"{i}_{prov}.h5")
                y.append(cls)
                groups.append("togo")
                i += 1
    return SimpleNamespace(
        labels=np.asarray(y, dtype=np.int64), groups=np.asarray(groups, dtype=object),
        sample_ids=np.asarray(sids, dtype=object),
    )


def _euro_official_bench():
    n = 60  # 20 per country
    groups = np.asarray([["Estonia", "Latvia", "Portugal"][i // 20] for i in range(n)], dtype=object)
    est = np.flatnonzero(groups == "Estonia")
    lat = np.flatnonzero(groups == "Latvia")
    por = np.flatnonzero(groups == "Portugal")
    official_splits = {
        "latvia_vs_estonia": {
            "train": lat.tolist(), "val": por[:8].tolist(), "test": est.tolist(),
            "target_train": est[:5].tolist(),  # deliberately ignored by v2 official
        },
        "latvia_portugal_vs_estonia": {
            "train": np.concatenate([lat, por[:8]]).tolist(), "val": por[8:16].tolist(),
            "test": est.tolist(), "target_train": est[:5].tolist(),
        },
    }
    return SimpleNamespace(
        groups=groups, labels=np.asarray([i % 3 for i in range(n)], dtype=np.int64),
        sample_ids=np.asarray([f"s{i}" for i in range(n)], dtype=object), official_splits=official_splits,
    )


def _breizh_official_bench():
    groups, labels, sids = [], [], []
    i = 0
    for region, n in (("frh01", 20), ("frh02", 20), ("frh03", 15), ("frh04", 15)):
        for _ in range(n):
            groups.append(region)
            labels.append(i % 3)
            sids.append(f"{region}:{i}")
            i += 1
    return SimpleNamespace(
        groups=np.asarray(groups, dtype=object), labels=np.asarray(labels, dtype=np.int64),
        sample_ids=np.asarray(sids, dtype=object),
    )


def _pastis_official_bench():
    patches, pid = [], 100
    for fold in range(1, 6):
        for _ in range(4):  # 4 patches per fold -> 20 patches
            patches.append(SimpleNamespace(patch_id=pid, fold=fold, tile=f"T3{pid % 4}", latlon=(float(pid), float(-pid))))
            pid += 1
    pids = [p.patch_id for p in patches]
    class_sets = {p.patch_id: {p.patch_id % 4, 10 + (p.patch_id % 3)} for p in patches}
    return SimpleNamespace(
        patches=patches,
        patch_ids=lambda folds=None: [p.patch_id for p in patches if folds is None or p.fold in folds],
        patch_class_sets=lambda ids=None: {int(p): class_sets[int(p)] for p in (ids if ids is not None else pids)},
        patch_tiles={p.patch_id: p.tile for p in patches},
        patch_latlon={p.patch_id: p.latlon for p in patches},
    )


# --------------------------------------------------------------------------- #
# Route capabilities
# --------------------------------------------------------------------------- #
def test_official_declares_target_geography_without_target_labels():
    assert RB.route_capabilities(official) == (True, False)


# --------------------------------------------------------------------------- #
# CropHarvest -- Togo only; seed-varying source subdivision, fixed marginals + target_test
# --------------------------------------------------------------------------- #
def test_cropharvest_official_is_togo_only():
    assert ch.OFFICIAL_HOLDOUTS == ["togo"]  # Kenya / lem-brazil removed
    assert ch.OFFICIAL_PROVENANCE == {"source": "togo", "target": "togo-eval"}


def test_cropharvest_official_togo_split_shape_and_route():
    bench = _ch_official_bench()
    prov = ch.provenance_groups(bench)
    split = next(iter(official.iter_source_target_splits(bench, ch, 0)))
    assert split.label == "togo"
    assert split.has_target is True and split.supports_target_labels is False
    assert split.source_test.size == 0 and split.target_label_pool.size == 0
    train_n, val_n = split_spec.official_source_train_val_sizes(100)
    assert (len(split.source_train), len(split.source_val)) == (train_n, val_n) == (90, 10)
    # target_test is exactly the togo-eval provenance (fixed release evaluation)
    assert set(split.target_test.tolist()) == set(np.flatnonzero(prov == "togo-eval").tolist())
    # source partitions are drawn ONLY from the togo provenance pool
    src_pool = set(np.flatnonzero(prov == "togo").tolist())
    assert set(split.source_train.tolist()) | set(split.source_val.tolist()) == src_pool


def test_cropharvest_official_seed_varies_but_marginals_and_target_are_fixed():
    bench = _ch_official_bench()
    y = bench.labels
    prov = ch.provenance_groups(bench)
    splits = {s: next(iter(official.iter_source_target_splits(bench, ch, s))) for s in (0, 1, 2)}

    def val_marginal(sp):
        return int((y[sp.source_val] == 1).sum()), int((y[sp.source_val] == 0).sum())

    # identical sizes + class marginals across seeds (crop 6 / non-crop 4 by the proportional quota)
    assert len({val_marginal(splits[s]) for s in (0, 1, 2)}) == 1
    assert val_marginal(splits[0]) == (6, 4)
    assert len({(len(splits[s].source_train), len(splits[s].source_val)) for s in (0, 1, 2)}) == 1
    # ONLY the source subdivision varies by seed
    assert len({frozenset(splits[s].source_val.tolist()) for s in (0, 1, 2)}) == 3
    # target_test is fixed across seeds (the release evaluation never varies)
    assert len({frozenset(splits[s].target_test.tolist()) for s in (0, 1, 2)}) == 1
    assert set(splits[0].target_test.tolist()) == set(np.flatnonzero(prov == "togo-eval").tolist())


# --------------------------------------------------------------------------- #
# EuroCropsML -- both anchors, exact release membership, seed-invariant
# --------------------------------------------------------------------------- #
def test_eurocropsml_official_anchors_exact_and_seed_invariant():
    bench = _euro_official_bench()
    assert [sp.label for sp in official.iter_source_target_splits(bench, euro, 0)] == list(euro.OFFICIAL_HOLDOUTS)
    per_seed = {s: {sp.label: sp for sp in official.iter_source_target_splits(bench, euro, s)} for s in (0, 1, 2)}
    for anchor in euro.OFFICIAL_HOLDOUTS:
        spec = bench.official_splits[anchor]
        sp0 = per_seed[0][anchor]
        assert set(sp0.source_train.tolist()) == set(spec["train"])   # release train -> source_train
        assert set(sp0.source_val.tolist()) == set(spec["val"])       # release val -> source_val
        assert set(sp0.target_test.tolist()) == set(spec["test"])     # release test -> target_test
        assert sp0.source_test.size == 0
        assert sp0.target_label_pool.size == 0                        # finetune target_train IGNORED
        assert sp0.has_target is True and sp0.supports_target_labels is False
        for s in (1, 2):  # identical membership across seeds
            sp = per_seed[s][anchor]
            assert np.array_equal(sp.source_train, sp0.source_train)
            assert np.array_equal(sp.source_val, sp0.source_val)
            assert np.array_equal(sp.target_test, sp0.target_test)


# --------------------------------------------------------------------------- #
# BreizhCrops -- FRH01+FRH02 / FRH03 / FRH04, fixed across seeds
# --------------------------------------------------------------------------- #
def test_breizhcrops_official_regions_exact_and_seed_invariant():
    bench = _breizh_official_bench()
    g = bench.groups
    splits = [next(iter(official.iter_source_target_splits(bench, bz, s))) for s in (0, 1, 2)]
    for sp in splits:
        assert sp.label == "frh04"
        assert set(sp.source_train.tolist()) == set(np.flatnonzero(np.isin(g, ["frh01", "frh02"])).tolist())
        assert set(sp.source_val.tolist()) == set(np.flatnonzero(g == "frh03").tolist())
        assert set(sp.target_test.tolist()) == set(np.flatnonzero(g == "frh04").tolist())
        assert sp.source_test.size == 0 and sp.target_label_pool.size == 0
        assert sp.has_target is True and sp.supports_target_labels is False
    assert np.array_equal(splits[0].source_train, splits[1].source_train)
    assert np.array_equal(splits[1].source_train, splits[2].source_train)
    assert np.array_equal(splits[0].target_test, splits[2].target_test)


# --------------------------------------------------------------------------- #
# PASTIS -- folds 1-3 / 4 / 5, patch-level, fixed across seeds
# --------------------------------------------------------------------------- #
def test_pastis_official_folds_patch_level_and_seed_invariant():
    bench = _pastis_official_bench()
    fold_of = {p.patch_id: p.fold for p in bench.patches}
    splits = [next(iter(official.iter_dense_source_target_splits(bench, pastis_mod, s))) for s in (0, 1, 2)]
    for d in splits:
        assert d.label == "fold_5"
        assert d.has_target is True and d.supports_target_labels is False
        assert d.source_train_patches == frozenset(p for p, f in fold_of.items() if f in {1, 2, 3})
        assert d.source_val_patches == frozenset(p for p, f in fold_of.items() if f == 4)
        assert d.target_test_patches == frozenset(p for p, f in fold_of.items() if f == 5)
        assert not d.source_test_patches and not d.target_label_pool_patches
    assert splits[0].as_partitions() == splits[1].as_partitions() == splits[2].as_partitions()


def test_pastis_official_patch_domains_are_published_folds():
    bench = _pastis_official_bench()
    doms = official.patch_domains(bench, pastis_mod)
    assert doms == {int(p.patch_id): str(p.fold) for p in bench.patches}


def test_official_sample_domains_are_native_groups():
    bench = _breizh_official_bench()
    assert np.array_equal(official.sample_domains(bench, bz), np.asarray(bench.groups, dtype=object))


# --------------------------------------------------------------------------- #
# Artifact round-trip -- build -> publish -> load reproduces the official split
# --------------------------------------------------------------------------- #
def test_cropharvest_official_round_trip(tmp_path):
    bench = _ch_official_bench()
    split = next(iter(official.iter_source_target_splits(bench, ch, 0)))
    domains = official.sample_domains(bench, ch)
    y, _g = ch.make_targets(bench)
    root = tmp_path / "splits"
    rows, summary = SA.build_tabular_leaf(
        "cropharvest", "official", 0, split=split, domains=domains, labels=y,
        sample_ids=bench.sample_ids, audit_events=[],
    )
    splitfix.freeze(root, [(rows, summary)])
    loaded = SA.load_tabular_splits(root, "cropharvest", bench.sample_ids, ["official"], [0])
    assert len(loaded) == 1
    ls = loaded[0]
    assert ls.split.has_target is True and ls.split.supports_target_labels is False
    for part, arr in split.as_partitions().items():
        assert set(getattr(ls.split, part).tolist()) == {int(i) for i in arr.tolist()}


def test_pastis_official_dense_round_trip_checks_published_folds(tmp_path):
    bench = _pastis_official_bench()
    dsplit = next(iter(official.iter_dense_source_target_splits(bench, pastis_mod, 0)))
    domain_of = {int(k): str(v) for k, v in official.patch_domains(bench, pastis_mod).items()}
    all_pids = [int(p) for p in bench.patch_ids(None)]
    root = tmp_path / "splits"
    rows, summary = SA.build_dense_leaf(
        "pastis", "official", 0, dense_split=dsplit, audit_events=[],
        all_patch_ids=all_pids, domain_of=domain_of,
        class_sets={int(k): set(v) for k, v in bench.patch_class_sets(all_pids).items()},
        patch_latlon=dict(bench.patch_latlon),
    )
    splitfix.freeze(root, [(rows, summary)])
    patch_fold = {int(p.patch_id): int(p.fold) for p in bench.patches}
    patch_tile = {int(k): v for k, v in bench.patch_tiles.items()}
    by_seed = SA.load_dense_splits(root, "pastis", patch_fold, patch_tile, ["official"], [0])
    assert by_seed[0]
    d = by_seed[0][0].split
    assert d.as_partitions() == dsplit.as_partitions()

    # structural check: if a patch's CURRENT published fold no longer matches the frozen domain, refuse
    tampered = dict(patch_fold)
    a_test_patch = sorted(dsplit.target_test_patches)[0]
    tampered[a_test_patch] = 1  # fold 5 -> claims fold 1
    with pytest.raises(SA.SplitArtifactError, match="structural metadata changed"):
        SA.load_dense_splits(root, "pastis", tampered, patch_tile, ["official"], [0])
