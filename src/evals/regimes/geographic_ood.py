"""Geographic leave-one-domain-out regime (schema v2).

Deployment-style geographic OOD: each frozen geographic unit is held out ONCE as the target; every
other eligible sample is the source. The source is purged of everything within ``purge_km`` of the
ENTIRE target FIRST, then the purged source is partitioned exactly 80/10/10
(source_train/source_val/source_test) and the target 80/20 (target_label_pool/target_test), both by
the shared deterministic partitioners. ``has_target=True`` and ``supports_target_labels=True``: the
target's own labels drive the target-label-budget routes -- few-shot draws come ONLY from
target_label_pool and every budget is scored on the fixed target_test.

Per-benchmark frozen geography (from :mod:`evals.split_spec`):
  * CropHarvest -- canonical localized regions rotate as headline targets; GeoWiki / Croplands are
    global collections that stay source-only (never a target); the one-class supplementary regions
    are held out as zero-shot STRESS targets (supports_target_labels=False, empty target_label_pool),
    excluded from the target-label routes. 50 km.
  * EuroCropsML -- Estonia / Latvia / Portugal LODO, the other two jointly source. 25 km.
  * BreizhCrops -- FRH01-FRH04 LODO, the other three jointly source. 5 km.
  * PASTIS (dense) -- Sentinel TILE LODO (never the published folds); patch-level multilabel source
    (80/10/10) and target (80/20) partitioning; whole patches, never pixels. 2 km.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
from sklearn.neighbors import BallTree

from evals import partition, split_spec
from evals.regimes.base import (
    TARGET_ROLE_HEADLINE,
    TARGET_ROLE_SUPPLEMENTARY_STRESS,
    DenseSourceTargetSplit,
    SourceTargetSplit,
    emit_split_audit_event,
)

NAME = "geographic_ood"
GROUP_KIND = "geography"
HAS_TARGET = True
SUPPORTS_TARGET_LABELS = True  # the held-out region's labels drive the target-label-budget routes
EARTH_RADIUS_KM = 6371.0088

_UNKNOWN = ("unknown", "nan")


def _spec(bench_mod) -> split_spec.BenchmarkSpec:
    return split_spec.ALL_SPECS[bench_mod.BENCHMARK]


def _target_rotation(spec: split_spec.BenchmarkSpec) -> tuple[str, ...]:
    """Every frozen geographic unit that is held out once as a target: the headline localized units
    plus the (one-class) supplementary stress units. Source-only global collections are NOT here."""
    return tuple(str(t) for t in spec.geographic_targets) + tuple(str(t) for t in spec.supplementary_targets)


# --------------------------------------------------------------------------- #
# Domain metadata
# --------------------------------------------------------------------------- #
def sample_domains(bench, bench_mod) -> np.ndarray:
    """Per-sample geographic domain -- the canonical unit LODO leaves out (region / country)."""
    del bench_mod
    return np.asarray(bench.groups, dtype=object)


def patch_domains(bench, bench_mod) -> dict[int, str]:
    """Per-patch geographic domain: the canonical Sentinel TILE (NOT the published fold). This is the
    unit tile-LODO holds out and the basis the runtime structural check validates against."""
    del bench_mod
    return {int(pid): str(tile) for pid, tile in bench.patch_tiles.items()}


def requested_targets(bench, bench_mod) -> list[str]:
    """The frozen LODO target rotation this regime DECLARES it will attempt (headline + supplementary),
    independent of what the data contains -- an absent target is still declared here and recorded as a
    dropped_holdout at generation, so it can never be a silent gap."""
    del bench
    return list(_target_rotation(_spec(bench_mod)))


# --------------------------------------------------------------------------- #
# Purge (whole source vs entire target, BEFORE partitioning)
# --------------------------------------------------------------------------- #
def _latlon(coords) -> np.ndarray | None:
    if coords is None:
        return None
    arr = np.asarray(coords, dtype=float)
    return arr if arr.ndim == 2 and arr.shape[1] == 2 else None


def _purge(source: np.ndarray, target: np.ndarray, latlon: np.ndarray | None, radius_km: float, *, where: str):
    """Remove every source item within ``radius_km`` of ANY target item (haversine). Records exactly
    which items were purged so the artifact builder can attach a PROVEN ``purged_near_ood`` reason."""
    source = np.asarray(source, dtype=np.int64)
    if radius_km <= 0 or latlon is None or len(source) == 0 or len(target) == 0:
        return np.sort(source)
    tgt_valid = np.asarray(target)[np.isfinite(latlon[target]).all(axis=1)]
    src_valid_mask = np.isfinite(latlon[source]).all(axis=1)
    if len(tgt_valid) == 0 or not src_valid_mask.any():
        return np.sort(source)
    tree = BallTree(np.deg2rad(latlon[tgt_valid]), metric="haversine")
    dist = tree.query(np.deg2rad(latlon[source[src_valid_mask]]), k=1, return_distance=True)[0].ravel()
    keep = np.ones(len(source), dtype=bool)
    keep[np.flatnonzero(src_valid_mask)] = dist > (radius_km / EARTH_RADIUS_KM)
    purged = source[~keep]
    if len(purged):
        emit_split_audit_event(
            "purge", where=where, reference="target", radius_km=float(radius_km),
            n_purged=int(len(purged)), purged_indices=[int(i) for i in purged.tolist()],
        )
    return np.sort(source[keep])


def _require_finite_coords(rows, latlon: np.ndarray | None, *, benchmark: str, target: str, kind: str) -> None:
    """Fail closed: every ASSIGNED source/target item MUST have finite coordinates. The purge is
    load-bearing for the holdout's meaning, so a missing coordinate is a hard generation error (with
    benchmark/target context) -- an item lacking a valid coordinate is never silently retained,
    dropped, or skipped."""
    if latlon is None:
        raise ValueError(
            f"{benchmark}/{NAME} target {target!r}: {kind} requires geographic coordinates, "
            f"but the benchmark provides none -- refuse to build (the source-target purge cannot run)"
        )
    rows = np.asarray(rows, dtype=np.int64)
    if len(rows) == 0:
        return
    bad = rows[~np.isfinite(latlon[rows]).all(axis=1)]
    if len(bad):
        raise ValueError(
            f"{benchmark}/{NAME} target {target!r}: {len(bad)} {kind} item(s) have non-finite "
            f"coordinates (first at index {int(bad[0])}); the purge cannot verify their distance to "
            f"the target region -- refuse to build"
        )


def _source_sizes(n: int) -> list[tuple[str, int]]:
    train, val, test = split_spec.source_partition_sizes(n)
    return [("source_train", train), ("source_val", val), ("source_test", test)]


# --------------------------------------------------------------------------- #
# Tabular LODO
# --------------------------------------------------------------------------- #
def iter_source_target_splits(bench, bench_mod, seed: int) -> Iterator[SourceTargetSplit]:
    spec = _spec(bench_mod)
    benchmark = bench_mod.BENCHMARK
    groups = np.asarray(sample_domains(bench, bench_mod), dtype=object).astype(str)
    y = np.asarray(bench_mod.make_targets(bench)[0])
    latlon = _latlon(getattr(bench, "latlon", None))
    present = set(groups.tolist())

    for target in _target_rotation(spec):
        if target not in present:
            emit_split_audit_event("dropped_holdout", regime=NAME, holdout=target, reason="absent_from_data")
            continue
        target_rows = np.flatnonzero(groups == target)
        # the COMPLETE eligible non-target population is the source (unknown/nan domains excluded)
        source_rows = np.flatnonzero((groups != target) & ~np.isin(groups, _UNKNOWN))
        # fail closed: every assigned source/target item must have a valid coordinate before the purge
        _require_finite_coords(target_rows, latlon, benchmark=benchmark, target=target, kind="target")
        _require_finite_coords(source_rows, latlon, benchmark=benchmark, target=target, kind="source")
        # purge the whole source against the ENTIRE target FIRST, then partition
        source_rows = _purge(source_rows, target_rows, latlon, spec.purge_km, where=NAME)
        if len(source_rows) == 0:
            emit_split_audit_event("dropped_holdout", regime=NAME, holdout=target, reason="empty_source_after_purge")
            continue

        src = partition.partition_source(
            [str(c) for c in y[source_rows].tolist()],
            [str(g) for g in groups[source_rows].tolist()],
            _source_sizes(len(source_rows)), int(seed),
        )
        source_train = np.sort(source_rows[src["source_train"]])
        source_val = np.sort(source_rows[src["source_val"]])
        source_test = np.sort(source_rows[src["source_test"]])

        n_target_classes = int(len(np.unique(y[target_rows])))
        # one-class regions (and the declared supplementary stress units) get NO target-label route:
        # zero-shot SUPPLEMENTARY STRESS only -- target_label_pool empty, supports_target_labels=False,
        # and target_role marks them so headline equal-region aggregation excludes them.
        if target in spec.supplementary_targets or n_target_classes < 2:
            yield SourceTargetSplit(
                label=target, source_train=source_train, source_val=source_val, source_test=source_test,
                target_label_pool=np.empty(0, dtype=np.int64), target_test=np.sort(target_rows),
                domain=target, has_target=True, supports_target_labels=False, group_kind=GROUP_KIND,
                target_role=TARGET_ROLE_SUPPLEMENTARY_STRESS,
            )
            continue

        pool_n, test_n = split_spec.target_partition_sizes(len(target_rows))
        tgt = partition.partition_target([str(c) for c in y[target_rows].tolist()], pool_n, test_n, int(seed))
        yield SourceTargetSplit(
            label=target, source_train=source_train, source_val=source_val, source_test=source_test,
            target_label_pool=np.sort(target_rows[tgt["target_label_pool"]]),
            target_test=np.sort(target_rows[tgt["target_test"]]),
            domain=target, has_target=True, supports_target_labels=True, group_kind=GROUP_KIND,
            target_role=TARGET_ROLE_HEADLINE,
        )


# --------------------------------------------------------------------------- #
# Dense (PASTIS) LODO over Sentinel tiles -- patch-level multilabel
# --------------------------------------------------------------------------- #
def iter_dense_source_target_splits(bench, bench_mod, seed: int) -> Iterator[DenseSourceTargetSplit]:
    spec = _spec(bench_mod)
    benchmark = bench_mod.BENCHMARK
    all_patches = [int(p) for p in bench.patch_ids(None)]
    tile_of = {int(pid): str(tile) for pid, tile in bench.patch_tiles.items()}
    class_sets = bench.patch_class_sets(all_patches)
    patch_latlon = bench.patch_latlon
    pids_arr = np.asarray(all_patches, dtype=np.int64)
    # a patch missing coordinates becomes (nan, nan) so the fail-closed check below catches it
    latlon = _latlon([patch_latlon.get(p, (np.nan, np.nan)) for p in all_patches])
    pos = {p: i for i, p in enumerate(all_patches)}  # patch id -> row in latlon
    present = {tile_of[p] for p in all_patches}

    for tile in (str(t) for t in spec.geographic_targets):
        if tile not in present:
            emit_split_audit_event("dropped_holdout", regime=NAME, holdout=tile, reason="absent_from_data")
            continue
        target_patches = [p for p in all_patches if tile_of[p] == tile]
        source_patches = [p for p in all_patches if tile_of[p] != tile and tile_of[p] not in _UNKNOWN]
        tgt_rows = np.asarray([pos[p] for p in target_patches], dtype=np.int64)
        src_rows = np.asarray([pos[p] for p in source_patches], dtype=np.int64)
        # fail closed: every assigned source/target patch must have a valid centroid before the purge
        _require_finite_coords(tgt_rows, latlon, benchmark=benchmark, target=tile, kind="target patch")
        _require_finite_coords(src_rows, latlon, benchmark=benchmark, target=tile, kind="source patch")
        # purge source PATCHES against the entire target tile (patch-centroid haversine), then partition
        kept_rows = _purge(src_rows, tgt_rows, latlon, spec.purge_km, where=NAME)
        source_patches = [int(pids_arr[r]) for r in kept_rows.tolist()]
        if not source_patches:
            emit_split_audit_event("dropped_holdout", regime=NAME, holdout=tile, reason="empty_source_after_purge")
            continue

        src = partition.multilabel_assign(
            [sorted(class_sets.get(p, set())) for p in source_patches], _source_sizes(len(source_patches)), int(seed),
        )
        pool_n, test_n = split_spec.target_partition_sizes(len(target_patches))
        tgt = partition.multilabel_assign(
            [sorted(class_sets.get(p, set())) for p in target_patches],
            [("target_label_pool", pool_n), ("target_test", test_n)], int(seed),
        )

        def pset(idx, base):  # noqa: ANN001 -- map partitioner indices back to patch ids
            return frozenset(base[i] for i in idx.tolist())

        yield DenseSourceTargetSplit(
            label=tile,
            source_train_patches=pset(src["source_train"], source_patches),
            source_val_patches=pset(src["source_val"], source_patches),
            source_test_patches=pset(src["source_test"], source_patches),
            target_label_pool_patches=pset(tgt["target_label_pool"], target_patches),
            target_test_patches=pset(tgt["target_test"], target_patches),
            has_target=True, supports_target_labels=True, group_kind=GROUP_KIND,
            target_role=TARGET_ROLE_HEADLINE,  # every Sentinel tile is a headline multiclass target
        )
