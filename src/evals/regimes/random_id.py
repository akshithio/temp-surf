"""In-distribution random split regime (schema v2).

The within-population reference: no target region, so the whole eligible population is split into the
exact-size source partitions (source_train/source_val/source_test = 80/10/10) with deterministic
constrained stratification and NO fallback. Target partitions are empty; ``has_target`` and
``supports_target_labels`` are both False.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from evals import partition, split_spec
from evals.regimes.base import DenseSourceTargetSplit, SourceTargetSplit

NAME = "random_id"
GROUP_KIND = "geography"
HAS_TARGET = False  # train and test share regions -> no target region to sweep
SUPPORTS_TARGET_LABELS = False  # source-only: no target region, so no target-label routes


def _source_sizes(n: int) -> list[tuple[str, int]]:
    train, val, test = split_spec.source_partition_sizes(n)
    return [("source_train", train), ("source_val", val), ("source_test", test)]


# --------------------------------------------------------------------------- #
# Tabular
# --------------------------------------------------------------------------- #
def iter_source_target_splits(bench, bench_mod, seed: int) -> Iterator[SourceTargetSplit]:
    """One in-distribution split: the whole population is the source, partitioned exactly 80/10/10 by
    class and region (constrained stratification, no fallback). No target partitions."""
    y, groups = bench_mod.make_targets(bench)
    assign = partition.partition_source(
        [str(c) for c in np.asarray(y).tolist()],
        [str(r) for r in np.asarray(groups).tolist()],
        _source_sizes(len(y)),
        int(seed),
    )
    yield SourceTargetSplit(
        label="random_id",
        source_train=assign["source_train"], source_val=assign["source_val"], source_test=assign["source_test"],
        has_target=False, supports_target_labels=False, group_kind=GROUP_KIND,
    )


def sample_domains(bench, bench_mod) -> np.ndarray:
    """Per-sample domain basis (the native region groups) recorded for worst-group scoring."""
    _y, groups = bench_mod.make_targets(bench)
    return np.asarray(groups, dtype=object)


# --------------------------------------------------------------------------- #
# Dense (PASTIS) -- patch-level iterative multilabel over class-presence vectors
# --------------------------------------------------------------------------- #
def iter_dense_source_target_splits(bench, bench_mod, seed: int) -> Iterator[DenseSourceTargetSplit]:
    del bench_mod
    patches = [int(p) for p in bench.patch_ids(None)]
    class_sets = bench.patch_class_sets(patches)
    label_sets = [sorted(class_sets.get(p, set())) for p in patches]
    assign = partition.multilabel_assign(label_sets, _source_sizes(len(patches)), int(seed))

    def pset(name: str) -> frozenset[int]:
        return frozenset(patches[i] for i in assign[name].tolist())

    yield DenseSourceTargetSplit(
        label="random_patch",
        source_train_patches=pset("source_train"), source_val_patches=pset("source_val"),
        source_test_patches=pset("source_test"),
        has_target=False, supports_target_labels=False, group_kind=GROUP_KIND,
    )


def patch_domains(bench, bench_mod) -> dict[int, str]:
    """Per-patch domain basis: the patch's canonical Sentinel TILE (group_kind='geography').

    The tile is PASTIS's geographic unit -- the same basis geographic_ood holds out -- so worst-group
    scoring and the recorded per-patch domains are by tile even though random_id never sweeps one out.
    It is NOT the published fold (a cache-layout / cross-validation artifact that spans all tiles).
    """
    del bench_mod
    return {int(pid): str(tile) for pid, tile in bench.patch_tiles.items()}
