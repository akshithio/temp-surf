"""Schema-v2 random_id vertical slice: the in-distribution regime emits explicit source/target
splits, and those splits survive the artifact round-trip unchanged.

random_id is the within-population reference: the whole eligible pool is the source, partitioned
EXACTLY 80/10/10 into source_train/source_val/source_test with deterministic constrained
stratification (no fallback); there is no target region, so both target partitions are empty and
``has_target`` / ``supports_target_labels`` are both False. These tests exercise the tabular
(``partition_source``) and dense (``multilabel_assign``) emitters and the
regime -> build -> publish -> load round-trip. No real benchmark data, no model, no disk cache.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from evals import split_artifacts as SA
from evals import split_spec
from evals.regimes import base as RB
from evals.regimes import random_id
from tests import splitfix

# --------------------------------------------------------------------------- #
# Fakes -- the minimal bench / bench_mod surface the regime touches.
# --------------------------------------------------------------------------- #
_TAB_N = 60  # 3 classes x 2 regions x 10 -> every (class, region) cell has 10 rows (clean quotas)


def _tab_bench():
    y = np.array([c for c in range(3) for _ in range(20)], dtype=np.int64)
    groups = np.array(
        [("A" if j < 10 else "B") for _c in range(3) for j in range(20)], dtype=object
    )
    sample_ids = np.array([f"s{i}" for i in range(_TAB_N)], dtype=object)
    bench = SimpleNamespace(labels=y, groups=groups, sample_ids=sample_ids)
    bench_mod = SimpleNamespace(BENCHMARK="cropharvest", make_targets=lambda b: (b.labels, b.groups))
    return bench, bench_mod


_DENSE_N = 30


def _dense_bench():
    patches = [
        SimpleNamespace(patch_id=i, fold=(i % 5) + 1, tile=f"T3{i % 4}", latlon=(float(i), float(-i)))
        for i in range(_DENSE_N)
    ]
    class_sets = {p.patch_id: {p.patch_id % 4, 10 + (p.patch_id % 3)} for p in patches}
    bench = SimpleNamespace(
        patches=patches,
        patch_ids=lambda folds=None: [
            p.patch_id for p in patches if folds is None or p.fold in folds
        ],
        patch_class_sets=lambda pids=None: {
            int(p): class_sets[int(p)]
            for p in (pids if pids is not None else [q.patch_id for q in patches])
        },
        patch_tiles={p.patch_id: p.tile for p in patches},  # @property dict on the real bench
        patch_latlon={p.patch_id: p.latlon for p in patches},
    )
    bench_mod = SimpleNamespace(BENCHMARK="pastis")
    return bench, bench_mod


# --------------------------------------------------------------------------- #
# Route capabilities (declared once, fail-closed elsewhere)
# --------------------------------------------------------------------------- #
def test_random_id_declares_source_only_route():
    assert RB.route_capabilities(random_id) == (False, False)


# --------------------------------------------------------------------------- #
# Tabular emitter -- partition_source
# --------------------------------------------------------------------------- #
def test_tabular_source_only_shape_and_route():
    bench, bench_mod = _tab_bench()
    splits = list(random_id.iter_source_target_splits(bench, bench_mod, seed=0))
    assert len(splits) == 1
    s = splits[0]
    assert isinstance(s, RB.SourceTargetSplit)
    assert s.label == "random_id"
    assert s.has_target is False and s.supports_target_labels is False
    assert s.target_label_pool.size == 0 and s.target_test.size == 0

    train, val, test = split_spec.source_partition_sizes(_TAB_N)
    assert (len(s.source_train), len(s.source_val), len(s.source_test)) == (train, val, test)

    # exact partition of the whole population: disjoint and complete
    everything = np.concatenate([s.source_train, s.source_val, s.source_test])
    assert np.array_equal(np.sort(everything), np.arange(_TAB_N))


def test_tabular_stratifies_class_and_region():
    """source_val's class and region marginals hit floor/ceil of the proportional quota (no drift)."""
    bench, bench_mod = _tab_bench()
    y, groups = bench_mod.make_targets(bench)
    s = next(iter(random_id.iter_source_target_splits(bench, bench_mod, seed=1)))
    _, val_n, _ = split_spec.source_partition_sizes(_TAB_N)
    # every clean cell contributes exactly its quota; each class (20) -> 2 in val, each region (30) -> 3.
    val_classes = np.asarray(y)[s.source_val]
    val_regions = np.asarray(groups)[s.source_val]
    assert len(s.source_val) == val_n
    assert {int(c): int((val_classes == c).sum()) for c in (0, 1, 2)} == {0: 2, 1: 2, 2: 2}
    assert {r: int((val_regions == r).sum()) for r in ("A", "B")} == {"A": 3, "B": 3}


def test_tabular_deterministic_in_seed():
    bench, bench_mod = _tab_bench()
    a = next(iter(random_id.iter_source_target_splits(bench, bench_mod, seed=2)))
    b = next(iter(random_id.iter_source_target_splits(bench, bench_mod, seed=2)))
    for part in ("source_train", "source_val", "source_test"):
        assert np.array_equal(getattr(a, part), getattr(b, part))


def test_sample_domains_are_the_native_regions():
    bench, bench_mod = _tab_bench()
    _y, groups = bench_mod.make_targets(bench)
    assert np.array_equal(random_id.sample_domains(bench, bench_mod), np.asarray(groups, dtype=object))


# --------------------------------------------------------------------------- #
# Dense emitter -- multilabel_assign (patch-level)
# --------------------------------------------------------------------------- #
def test_dense_source_only_shape_and_route():
    bench, bench_mod = _dense_bench()
    splits = list(random_id.iter_dense_source_target_splits(bench, bench_mod, seed=0))
    assert len(splits) == 1
    d = splits[0]
    assert isinstance(d, RB.DenseSourceTargetSplit)
    assert d.label == "random_patch"
    assert d.has_target is False and d.supports_target_labels is False
    assert not d.target_label_pool_patches and not d.target_test_patches

    train, val, test = split_spec.source_partition_sizes(_DENSE_N)
    assert (len(d.source_train_patches), len(d.source_val_patches), len(d.source_test_patches)) == (
        train, val, test,
    )
    everything = d.source_train_patches | d.source_val_patches | d.source_test_patches
    assert everything == set(range(_DENSE_N))  # every patch placed exactly once (disjoint + complete)


def test_dense_deterministic_in_seed():
    bench, bench_mod = _dense_bench()
    a = next(iter(random_id.iter_dense_source_target_splits(bench, bench_mod, seed=3)))
    b = next(iter(random_id.iter_dense_source_target_splits(bench, bench_mod, seed=3)))
    assert a.as_partitions() == b.as_partitions()


# --------------------------------------------------------------------------- #
# Artifact round-trip -- regime -> build -> publish -> load
# --------------------------------------------------------------------------- #
def _publish_tabular(root, bench, bench_mod, split, seed):
    y, _g = bench_mod.make_targets(bench)
    domains = random_id.sample_domains(bench, bench_mod)
    rows, summary = SA.build_tabular_leaf(
        bench_mod.BENCHMARK, "random_id", seed,
        split=split, domains=domains, labels=y, sample_ids=bench.sample_ids, audit_events=[],
    )
    splitfix.freeze(root, [(rows, summary)])


def test_tabular_round_trip_preserves_partitions(tmp_path):
    bench, bench_mod = _tab_bench()
    seed = 0
    root = tmp_path / "splits"
    split = next(iter(random_id.iter_source_target_splits(bench, bench_mod, seed)))
    _publish_tabular(root, bench, bench_mod, split, seed)

    loaded = SA.load_tabular_splits(root, bench_mod.BENCHMARK, bench.sample_ids, ["random_id"], [seed])
    assert len(loaded) == 1
    ls = loaded[0]
    assert ls.seed == seed and ls.regime == "random_id"
    assert ls.split.has_target is False and ls.split.supports_target_labels is False
    # Membership is preserved exactly; loaded arrays are in canonical stable-id order (a row-set, not
    # an ordered sequence -- the runtime indexes embeddings with them), so compare as sets.
    for part in ("source_train", "source_val", "source_test"):
        assert set(getattr(ls.split, part).tolist()) == set(getattr(split, part).tolist())
    assert ls.split.target_label_pool.size == 0 and ls.split.target_test.size == 0
    # per-sample domain array reconstructed from the artifact matches the native regions
    _y, groups = bench_mod.make_targets(bench)
    assert np.array_equal(ls.domains, np.asarray(groups, dtype=object))


def test_dense_round_trip_preserves_patch_sets(tmp_path):
    bench, bench_mod = _dense_bench()
    seed = 0
    root = tmp_path / "splits"
    dsplit = next(iter(random_id.iter_dense_source_target_splits(bench, bench_mod, seed)))
    domain_of = {int(k): str(v) for k, v in random_id.patch_domains(bench, bench_mod).items()}
    all_pids = [int(p) for p in bench.patch_ids(None)]
    rows, summary = SA.build_dense_leaf(
        bench_mod.BENCHMARK, "random_id", seed, dense_split=dsplit, audit_events=[],
        all_patch_ids=all_pids, domain_of=domain_of,
        class_sets={int(k): set(v) for k, v in bench.patch_class_sets(all_pids).items()},
        patch_latlon=dict(bench.patch_latlon),
    )
    splitfix.freeze(root, [(rows, summary)])

    patch_fold = {int(p.patch_id): int(p.fold) for p in bench.patches}
    patch_tile = {int(k): v for k, v in bench.patch_tiles.items()}
    by_seed = SA.load_dense_splits(root, bench_mod.BENCHMARK, patch_fold, patch_tile, ["random_id"], [seed])
    assert list(by_seed) == [seed]
    ld = by_seed[seed][0]
    assert ld.regime == "random_id" and ld.split.has_target is False
    assert ld.split.as_partitions() == dsplit.as_partitions()


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_tabular_round_trip_all_seeds(tmp_path, seed):
    bench, bench_mod = _tab_bench()
    root = tmp_path / "splits"
    split = next(iter(random_id.iter_source_target_splits(bench, bench_mod, seed)))
    _publish_tabular(root, bench, bench_mod, split, seed)
    loaded = SA.load_tabular_splits(root, bench_mod.BENCHMARK, bench.sample_ids, ["random_id"], [seed])
    assert set(loaded[0].split.source_train.tolist()) == set(split.source_train.tolist())


# --------------------------------------------------------------------------- #
# PASTIS domain metadata: the recorded per-patch domain is the Sentinel TILE, not the fold
# --------------------------------------------------------------------------- #
def test_dense_patch_domains_are_tiles_not_folds():
    bench, bench_mod = _dense_bench()
    doms = random_id.patch_domains(bench, bench_mod)
    tiles = {int(p.patch_id): str(p.tile) for p in bench.patches}
    folds = {int(p.patch_id): str(p.fold) for p in bench.patches}
    assert doms == tiles                                   # recorded domain == Sentinel tile
    assert all(doms[pid] != folds[pid] for pid in doms)    # tile and fold genuinely differ per patch
    assert doms != folds


def test_dense_recorded_domains_in_published_leaf_are_tiles(tmp_path):
    bench, bench_mod = _dense_bench()
    seed = 0
    root = tmp_path / "splits"
    dsplit = next(iter(random_id.iter_dense_source_target_splits(bench, bench_mod, seed)))
    domain_of = {int(k): str(v) for k, v in random_id.patch_domains(bench, bench_mod).items()}
    all_pids = [int(p) for p in bench.patch_ids(None)]
    rows, summary = SA.build_dense_leaf(
        bench_mod.BENCHMARK, "random_id", seed, dense_split=dsplit, audit_events=[],
        all_patch_ids=all_pids, domain_of=domain_of,
        class_sets={int(k): set(v) for k, v in bench.patch_class_sets(all_pids).items()},
        patch_latlon=dict(bench.patch_latlon),
    )
    splitfix.freeze(root, [(rows, summary)])
    csv_rows = SA.read_assignments_csv(SA.assignments_path(root, bench_mod.BENCHMARK, "random_id", seed, "random_patch"))
    recorded = {r["stable_id"]: r["domain"] for r in csv_rows if r["status"] == SA.STATUS_ASSIGNED}
    tiles = {str(p.patch_id): str(p.tile) for p in bench.patches}
    folds = {str(p.patch_id): str(p.fold) for p in bench.patches}
    assert recorded, "no per-patch domains were recorded"
    for pid_s, dom in recorded.items():
        assert dom == tiles[pid_s]      # every recorded domain is the patch's tile
        assert dom != folds[pid_s]      # never the fold
    assert summary["group_kind"] == "geography"  # group_kind preserved in the central-log entry


# --------------------------------------------------------------------------- #
# Seed behavior: same seed reproduces membership; seeds 0/1/2 are distinct; sizes+marginals stable
# --------------------------------------------------------------------------- #
def test_tabular_seeds_012_distinct_membership_stable_sizes_and_marginals():
    bench, bench_mod = _tab_bench()
    y, groups = bench_mod.make_targets(bench)
    y = np.asarray(y)
    groups = np.asarray(groups)
    splits = {s: next(iter(random_id.iter_source_target_splits(bench, bench_mod, s))) for s in (0, 1, 2)}

    def sizes(sp):
        return tuple(len(getattr(sp, p)) for p in ("source_train", "source_val", "source_test"))

    def marginals(sp, part):
        idx = getattr(sp, part)
        cls = tuple(int((y[idx] == c).sum()) for c in (0, 1, 2))
        reg = tuple(int((groups[idx] == r).sum()) for r in ("A", "B"))
        return cls, reg

    assert len({sizes(splits[s]) for s in (0, 1, 2)}) == 1              # identical sizes
    for part in ("source_train", "source_val", "source_test"):
        assert len({marginals(splits[s], part) for s in (0, 1, 2)}) == 1, f"{part} marginals drifted"
    assert len({frozenset(splits[s].source_test.tolist()) for s in (0, 1, 2)}) == 3  # distinct membership


def test_dense_seeds_012_distinct_membership_stable_sizes():
    bench, bench_mod = _dense_bench()
    splits = {s: next(iter(random_id.iter_dense_source_target_splits(bench, bench_mod, s))) for s in (0, 1, 2)}
    parts = ("source_train_patches", "source_val_patches", "source_test_patches")
    assert len({tuple(len(getattr(splits[s], p)) for p in parts) for s in (0, 1, 2)}) == 1  # identical sizes
    # the complete patch partition differs across all three seeds (the fixture is large enough to permit it)
    full = {s: tuple(frozenset(getattr(splits[s], p)) for p in parts) for s in (0, 1, 2)}
    assert len(set(full.values())) == 3
