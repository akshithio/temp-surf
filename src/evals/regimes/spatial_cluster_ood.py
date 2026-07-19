"""Spatial-cluster OOD regime (schema v2).

Coordinate-only spatial cells: exactly five spherical K-means Voronoi cells built from lat/lon
ALONE, each rotated ONCE as the target (every other cell is the source). This is the
deployment-style geographic OOD that does not depend on any curated region/tile basis -- the cells
partition the benchmark's own footprint.

Determinism contract (mirrors :mod:`evals.split_spec`):
  * cell boundaries are FROZEN at :data:`~evals.split_spec.CLUSTER_SEED` and never vary with the run
    seed -- the same five cells for run seeds 0, 1, 2;
  * labels, native regions, published folds, class support, and the run seed are NEVER consulted to
    construct or modify a cell;
  * the run seed varies ONLY the source subdivision, the target pool/test membership, and later
    label draws.

Per rotation the source is purged of everything within the benchmark's ``purge_km`` of the ENTIRE
target cell FIRST, then the purged source is partitioned exactly 80/10/10
(source_train/source_val/source_test) and the target 80/20 (target_label_pool/target_test) by the
shared deterministic partitioners. ``has_target=True`` and ``supports_target_labels=True``: every
cell is a headline target whose own labels drive the target-label-budget routes (few-shot draws come
ONLY from target_label_pool, every budget scored on the fixed target_test). A cell is never merged,
expanded, dropped, or altered to improve class support -- if a real-data cell cannot support the
declared 80/10/10 + 80/20 route, generation fails explicitly (the shared partitioners raise).

Per-benchmark purge radii (from :mod:`evals.split_spec`): CropHarvest 50 km, EuroCropsML 25 km,
BreizhCrops 5 km, PASTIS-R 2 km. PASTIS clusters at the ORIGINAL patch level over EPSG:4326 patch
centroids (whole patches, never pixels or published folds); the cell assignment is stored in the
dense manifest and the runtime NEVER re-runs K-means.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
from sklearn.cluster import KMeans

from evals import partition, split_spec
from evals.regimes.base import (
    TARGET_ROLE_HEADLINE,
    DenseSourceTargetSplit,
    SourceTargetSplit,
)

# The purge, coordinate coercion, and source-sizing are the SAME load-bearing semantics as
# geographic_ood; importing them keeps a single source of truth (no second implementation to drift).
from evals.regimes.geographic_ood import _latlon, _purge, _source_sizes

NAME = "spatial_cluster_ood"
GROUP_KIND = "spatial_cluster"
HAS_TARGET = True
SUPPORTS_TARGET_LABELS = True  # every cell's labels drive the target-label-budget routes
EARTH_RADIUS_KM = 6371.0088

#: Five spherical K-means Voronoi cells, frozen at CLUSTER_SEED (never the run seed), n_init fixed.
N_CLUSTERS: int = split_spec.N_CLUSTERS
CLUSTER_SEED: int = split_spec.CLUSTER_SEED
_KMEANS_N_INIT = 10

#: Canonical cell labels in ascending (mean latitude, then mean longitude) order.
_CELL_NAMES: tuple[str, ...] = tuple(f"cluster_{i:02d}" for i in range(N_CLUSTERS))


# --------------------------------------------------------------------------- #
# Coordinate-only cell construction (fail-closed, seed-independent, label-independent)
# --------------------------------------------------------------------------- #
def _sphere_features(latlon: np.ndarray) -> np.ndarray:
    """Lat/lon (degrees) -> 3D unit-sphere coordinates, so K-means clusters by great-circle geometry
    rather than by raw degrees (which distort badly away from the equator and wrap at the meridian)."""
    lat = np.deg2rad(latlon[:, 0])
    lon = np.deg2rad(latlon[:, 1])
    return np.column_stack([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)])


def _require_all_finite(latlon: np.ndarray | None, *, benchmark: str, unit: str) -> np.ndarray:
    """Fail closed: spatial cells require a finite coordinate for EVERY eligible item. A missing
    coordinate is a hard generation error (with benchmark context) -- never silently retained,
    dropped, or skipped, because the cell an item lands in is defined by that coordinate."""
    if latlon is None:
        raise ValueError(
            f"{benchmark}/{NAME}: the benchmark provides no geographic coordinates, but spatial cells "
            f"require a finite coordinate for every eligible {unit} -- refuse to build"
        )
    finite = np.isfinite(latlon).all(axis=1)
    if not finite.all():
        bad = np.flatnonzero(~finite)
        raise ValueError(
            f"{benchmark}/{NAME}: {len(bad)} eligible {unit}(s) have non-finite coordinates (first at "
            f"index {int(bad[0])}); spatial cells require a finite coordinate for every eligible "
            f"{unit} -- refuse to build"
        )
    return latlon


def _cell_labels(coords, *, benchmark: str, unit: str) -> np.ndarray:
    """Deterministic five-cell spherical-K-means Voronoi assignment over ALL rows of ``coords``.

    Fail-closed on any non-finite coordinate. Uses exactly :data:`N_CLUSTERS` cells with
    ``random_state=CLUSTER_SEED`` and ``n_init=10``; consults NOTHING but the coordinates (no labels,
    regions, folds, class support, or run seed). Cells are renamed ``cluster_00``..``cluster_04`` by
    ascending (mean latitude, then mean longitude) of their members, so the naming is a deterministic
    function of geometry alone. Returns an object array of cell labels aligned to ``coords``.
    """
    latlon = _require_all_finite(_latlon(coords), benchmark=benchmark, unit=unit)
    n = len(latlon)
    n_distinct = len(np.unique(np.round(latlon, 9), axis=0))
    if n < N_CLUSTERS or n_distinct < N_CLUSTERS:
        raise ValueError(
            f"{benchmark}/{NAME}: need at least {N_CLUSTERS} distinct located {unit}s to form "
            f"{N_CLUSTERS} spatial cells (have {n_distinct} distinct of {n}) -- refuse to build"
        )
    raw = KMeans(n_clusters=N_CLUSTERS, random_state=CLUSTER_SEED, n_init=_KMEANS_N_INIT).fit_predict(
        _sphere_features(latlon)
    )
    present = sorted({int(c) for c in raw})
    if len(present) != N_CLUSTERS:
        raise ValueError(
            f"{benchmark}/{NAME}: K-means produced {len(present)} non-empty cells, not {N_CLUSTERS} "
            f"-- the {unit} coordinates cannot support {N_CLUSTERS} spatial cells; refuse to build"
        )

    def _center(cluster: int) -> tuple[float, float]:
        members = latlon[raw == cluster]
        return float(members[:, 0].mean()), float(members[:, 1].mean())

    remap = {old: new for new, old in enumerate(sorted(present, key=_center))}
    return np.asarray([f"cluster_{remap[int(c)]:02d}" for c in raw], dtype=object)


# --------------------------------------------------------------------------- #
# Domain metadata (the fixed cell of every sample / patch; recorded once, loaded from the artifact)
# --------------------------------------------------------------------------- #
def sample_domains(bench, bench_mod) -> np.ndarray:
    """Per-sample spatial cell (the frozen Voronoi cell from lat/lon). Identical across run seeds;
    fail-closed on any non-finite coordinate. Recorded once for worst-cluster scoring."""
    return _cell_labels(getattr(bench, "latlon", None), benchmark=bench_mod.BENCHMARK, unit="sample")


def patch_domains(bench, bench_mod) -> dict[int, str]:
    """Per-patch spatial cell from the EPSG:4326 patch centroid. A patch missing a centroid becomes
    (nan, nan) so the fail-closed check catches it. Stored in the dense manifest; the runtime loads
    the cell from the artifact and NEVER re-runs K-means."""
    all_patches = [int(p) for p in bench.patch_ids(None)]
    patch_latlon = bench.patch_latlon
    cells = _cell_labels(
        [patch_latlon.get(p, (np.nan, np.nan)) for p in all_patches],
        benchmark=bench_mod.BENCHMARK, unit="patch",
    )
    return {p: str(cells[i]) for i, p in enumerate(all_patches)}


# --------------------------------------------------------------------------- #
# Tabular: rotate all five cells (purge whole source vs entire target, THEN partition)
# --------------------------------------------------------------------------- #
def iter_source_target_splits(bench, bench_mod, seed: int) -> Iterator[SourceTargetSplit]:
    benchmark = bench_mod.BENCHMARK
    spec = split_spec.ALL_SPECS[benchmark]
    y = np.asarray(bench_mod.make_targets(bench)[0])
    latlon = _latlon(getattr(bench, "latlon", None))
    cells = _cell_labels(getattr(bench, "latlon", None), benchmark=benchmark, unit="sample")

    for target in _CELL_NAMES:
        target_rows = np.flatnonzero(cells == target)
        source_rows = np.flatnonzero(cells != target)  # every OTHER cell is the source
        # purge the whole source against the ENTIRE target cell FIRST, then partition
        source_rows = _purge(source_rows, target_rows, latlon, spec.purge_km, where=NAME)
        if len(source_rows) == 0:
            raise ValueError(
                f"{benchmark}/{NAME} target {target!r}: the source is empty after the {spec.purge_km} km "
                f"purge -- the cell cannot support the declared route; refuse to build (never merged/expanded)"
            )

        # the region marginal is the CELL (source spans the four non-target cells)
        src = partition.partition_source(
            [str(c) for c in y[source_rows].tolist()],
            [str(g) for g in cells[source_rows].tolist()],
            _source_sizes(len(source_rows)), int(seed),
        )
        source_train = np.sort(source_rows[src["source_train"]])
        source_val = np.sort(source_rows[src["source_val"]])
        source_test = np.sort(source_rows[src["source_test"]])

        pool_n, test_n = split_spec.target_partition_sizes(len(target_rows))
        tgt = partition.partition_target([str(c) for c in y[target_rows].tolist()], pool_n, test_n, int(seed))
        yield SourceTargetSplit(
            label=target, source_train=source_train, source_val=source_val, source_test=source_test,
            target_label_pool=np.sort(target_rows[tgt["target_label_pool"]]),
            target_test=np.sort(target_rows[tgt["target_test"]]),
            domain=target, has_target=True, supports_target_labels=True, group_kind=GROUP_KIND,
            target_role=TARGET_ROLE_HEADLINE,  # every spatial cell is a headline target
        )


# --------------------------------------------------------------------------- #
# Dense (PASTIS): patch-level spatial cells -- multilabel source/target partitioning, patch atomicity
# --------------------------------------------------------------------------- #
def iter_dense_source_target_splits(bench, bench_mod, seed: int) -> Iterator[DenseSourceTargetSplit]:
    benchmark = bench_mod.BENCHMARK
    spec = split_spec.ALL_SPECS[benchmark]
    all_patches = [int(p) for p in bench.patch_ids(None)]
    class_sets = bench.patch_class_sets(all_patches)
    patch_latlon = bench.patch_latlon
    pids_arr = np.asarray(all_patches, dtype=np.int64)
    coords = [patch_latlon.get(p, (np.nan, np.nan)) for p in all_patches]
    latlon = _latlon(coords)
    cells = _cell_labels(coords, benchmark=benchmark, unit="patch")
    pos = {p: i for i, p in enumerate(all_patches)}  # patch id -> row in latlon

    for target in _CELL_NAMES:
        target_patches = [p for p, c in zip(all_patches, cells, strict=True) if c == target]
        source_patches = [p for p, c in zip(all_patches, cells, strict=True) if c != target]
        tgt_rows = np.asarray([pos[p] for p in target_patches], dtype=np.int64)
        src_rows = np.asarray([pos[p] for p in source_patches], dtype=np.int64)
        # purge source PATCHES against the entire target cell (patch-centroid haversine), then partition
        kept_rows = _purge(src_rows, tgt_rows, latlon, spec.purge_km, where=NAME)
        source_patches = [int(pids_arr[r]) for r in kept_rows.tolist()]
        if not source_patches:
            raise ValueError(
                f"{benchmark}/{NAME} target {target!r}: the source is empty after the {spec.purge_km} km "
                f"purge -- the cell cannot support the declared route; refuse to build"
            )

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
            label=target,
            source_train_patches=pset(src["source_train"], source_patches),
            source_val_patches=pset(src["source_val"], source_patches),
            source_test_patches=pset(src["source_test"], source_patches),
            target_label_pool_patches=pset(tgt["target_label_pool"], target_patches),
            target_test_patches=pset(tgt["target_test"], target_patches),
            has_target=True, supports_target_labels=True, group_kind=GROUP_KIND,
            target_role=TARGET_ROLE_HEADLINE,  # every spatial cell is a headline multiclass target
        )
