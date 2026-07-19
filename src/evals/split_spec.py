"""Authoritative TERRA split specification: sizing formulas and per-benchmark policy inputs.

This is the single source of truth for HOW every regime partitions a benchmark. It carries no data
and constructs no splits -- it defines the deterministic sizing rules and the frozen policy inputs
(population exclusions, canonical region merges, target eligibility, purge radii, cluster config,
official anchors) that ``tools/generate_splits.py`` and the regime code consume. The realized
per-region counts and purge counts are OUTPUTS computed from data against these rules; the golden
consistency tests pin them.

Sizing rules (exact-size, deterministic -- no proportion-based ``train_test_split``):

    source 80/10/10:  source_val = source_test = ceil(0.1 * N_source);  source_train = remainder
    target 80/20:     target_test = ceil(0.2 * N_target);               target_label_pool = remainder
    official 90/10:   val = floor(0.1 * N_pool) (= N // 10);             train = remainder

Every run seed in :data:`RUN_SEEDS` yields the SAME partition sizes; only membership varies for
partitions that contain a randomized draw. Spatial-cluster cell boundaries are fixed with
:data:`CLUSTER_SEED`; run seeds vary only the source/target subdivisions inside the fixed cells.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# --------------------------------------------------------------------------- #
# Global determinism knobs
# --------------------------------------------------------------------------- #
#: The three deterministic evaluation replicates. Each run seed determines split membership,
#: label-budget draws, and probe randomness (via operation-specific derived sub-seeds). There are
#: three complete pipeline repetitions, NOT nine combinations of independent seeds.
RUN_SEEDS: tuple[int, ...] = (0, 1, 2)

#: Spatial-cluster cell boundaries are frozen at this seed and never vary with the run seed.
CLUSTER_SEED: int = 0
#: Number of spherical K-means Voronoi cells for spatial_cluster_ood (every cell rotates as target).
N_CLUSTERS: int = 5


# --------------------------------------------------------------------------- #
# Exact-size partition formulas (the numerical heart -- shared by generator + tests)
# --------------------------------------------------------------------------- #
def source_partition_sizes(n_source: int) -> tuple[int, int, int]:
    """Exact-size 80/10/10 source split.

    ``source_val = source_test = ceil(0.1 * n_source)``; ``source_train`` is the remainder. Returns
    ``(train, val, test)``. Raises if the population is too small to give a non-empty train.
    """
    if n_source < 0:
        raise ValueError(f"n_source must be non-negative, got {n_source}")
    val = math.ceil(n_source / 10)
    test = math.ceil(n_source / 10)
    train = n_source - val - test
    if train <= 0:
        raise ValueError(f"n_source={n_source} too small for an exact 80/10/10 split (train={train})")
    return train, val, test


def target_partition_sizes(n_target: int) -> tuple[int, int]:
    """Exact-size 80/20 target split.

    ``target_test = ceil(0.2 * n_target)``; ``target_label_pool`` is the remainder. Returns
    ``(label_pool, test)``.
    """
    if n_target < 0:
        raise ValueError(f"n_target must be non-negative, got {n_target}")
    test = math.ceil(n_target / 5)
    pool = n_target - test
    if pool < 0:
        raise ValueError(f"n_target={n_target} too small for an 80/20 target split")
    return pool, test


def official_source_train_val_sizes(n_pool: int) -> tuple[int, int]:
    """CropHarvest ``official`` 90/10 source subdivision (distinct from the 80/10/10 rule).

    ``val = floor(0.1 * n_pool) = n_pool // 10``; ``train`` is the remainder. There is no source
    test partition -- ``official`` has a fixed release evaluation set. Returns ``(train, val)``.
    """
    if n_pool < 0:
        raise ValueError(f"n_pool must be non-negative, got {n_pool}")
    val = n_pool // 10
    train = n_pool - val
    return train, val


# --------------------------------------------------------------------------- #
# Per-benchmark policy inputs
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class BenchmarkSpec:
    """Frozen policy inputs for one benchmark's split construction (no data, no realized counts)."""

    benchmark: str
    #: exact population after the frozen exclusions below (parcels / examples / patches)
    population: int
    #: source<->target distance purge radius, in km, applied BEFORE source partitioning
    purge_km: float
    #: headline target units rotated by geographic_ood (each is left out once)
    geographic_targets: tuple[str, ...]
    #: geographic units that stay source-only (never a headline target)
    source_only_units: tuple[str, ...] = ()
    #: supplementary one-class targets (source-only stress; excluded from headline mean/worst)
    supplementary_targets: tuple[str, ...] = ()
    #: stable-id provenance (what an artifact stores per unit); documented, not consumed here
    id_kind: str = "sample"  # "sample" | "patch"
    notes: str = ""


# ---- CropHarvest ---------------------------------------------------------- #
#: Malformed file excluded from the staged release before every regime.
CROPHARVEST_EXCLUDED_FILES: tuple[str, ...] = ("1619_central-asia.h5",)

#: THE single source of truth for CropHarvest provenance -> canonical region merges: ordered
#: ``(substring, canonical_region)`` rules applied to the lowercased provenance name. Provenance
#: (22 source datasets) is retained SEPARATELY for the official Togo split; the canonical region
#: (17) is the geographic unit for random/geographic/spatial regimes. ``evals.benchmarks.cropharvest``
#: consumes this via :func:`cropharvest_canonical_region`; it holds no merge rules of its own.
CROPHARVEST_REGION_MERGES: tuple[tuple[str, str], ...] = (
    ("togo", "togo"),          # togo + togo-eval provenance -> togo
    ("brazil", "lem-brazil"),  # Brazil sources -> lem-brazil
    ("kenya", "kenya"),        # Kenya sources -> kenya
    ("rwanda", "rwanda"),      # Rwanda sources -> rwanda
    ("mali", "mali"),          # mali + mali-non-crop + ... -> mali
)


def cropharvest_canonical_region(dataset: str) -> str:
    """Canonical CropHarvest region (17) from a source-collection provenance name (the SoT mapping).

    First matching :data:`CROPHARVEST_REGION_MERGES` rule wins; an unmatched provenance keeps its
    (lowercased) name. Used to build ``bench.groups`` and to reason about eligibility.
    """
    name = str(dataset).lower()
    for needle, region in CROPHARVEST_REGION_MERGES:
        if needle in name:
            return region
    return name

#: The 17 canonical CropHarvest regions.
CROPHARVEST_CANONICAL_REGIONS: tuple[str, ...] = (
    "central-asia", "croplands", "ethiopia", "geowiki-landcover-2017", "ile-de-france",
    "kenya", "lem-brazil", "mali", "martinique-france", "reunion-france", "rwanda",
    "sudan", "tanzania", "togo", "uganda", "usa-kern", "zimbabwe",
)

#: Globally distributed collections: always source, never a headline target.
CROPHARVEST_GLOBAL_COLLECTIONS: tuple[str, ...] = ("croplands", "geowiki-landcover-2017")

CROPHARVEST = BenchmarkSpec(
    benchmark="cropharvest",
    population=67_692,
    purge_km=50.0,
    geographic_targets=(
        "ethiopia", "ile-de-france", "kenya", "lem-brazil", "mali", "martinique-france",
        "reunion-france", "rwanda", "sudan", "togo", "usa-kern",
    ),
    supplementary_targets=("central-asia", "tanzania", "uganda", "zimbabwe"),
    source_only_units=CROPHARVEST_GLOBAL_COLLECTIONS,
    id_kind="sample",
    notes="official regime = Togo only (is_crop labels do not reproduce Kenya maize / Brazil coffee).",
)

#: CropHarvest official: Togo provenance source pool (subdivided 90/10 per seed) + fixed release
#: evaluation set. Class counts are release-defined and fixed across seeds. The 1,145/127 source
#: train/val MEMBERSHIP varies by seed while preserving these class counts; the eval set is identical
#: across seeds.
CROPHARVEST_OFFICIAL = {
    "source_pool": 1_272,                                     # togo provenance (crop + non-crop)
    "source_pool_classes": {"crop": 684, "non_crop": 588},   # 616+68 crop, 529+59 non-crop
    "source_train": 1_145,
    "source_train_classes": {"crop": 616, "non_crop": 529},
    "source_val": 127,
    "source_val_classes": {"crop": 68, "non_crop": 59},
    "eval_test": 306,                                         # togo-eval, identical across seeds
    "eval_test_classes": {"crop": 106, "non_crop": 200},
}


# ---- EuroCropsML ---------------------------------------------------------- #
#: Seven six-digit HCAT classes with < 10 global examples (22 parcels total) removed before every
#: regime, fixing the current random split's silent loss of stratification.
EUROCROPS_REMOVED_CLASSES: dict[str, int] = {
    "330112": 2, "330117": 2, "330125": 9, "330126": 1, "330130": 3, "330310": 2, "330403": 3,
}
EUROCROPS_RAW_POPULATION: int = 706_683
EUROCROPS_N_CLASSES_AFTER: int = 40

EUROCROPML = BenchmarkSpec(
    benchmark="eurocropsml",
    population=706_661,   # 706,683 - 22 removed
    purge_km=25.0,
    geographic_targets=("Estonia", "Latvia", "Portugal"),
    id_kind="sample",
    notes="frozen 40-class mask applied before all regimes; singleton target class -> target test only.",
)

#: Country freezes after the 40-class mask.
EUROCROPS_COUNTRY_POPULATION: dict[str, int] = {
    "Estonia": 175_892, "Latvia": 431_138, "Portugal": 99_631,
}

#: Release official splits (exact source-only anchors, identical across seeds).
EUROCROPS_OFFICIAL = {
    "Latvia->Estonia": {"train": 172_989, "val": 43_248, "test": 35_179},
    "Latvia+Portugal->Estonia": {"train": 239_531, "val": 59_884, "test": 35_179},
}


# ---- BreizhCrops ---------------------------------------------------------- #
BREIZHCROPS = BenchmarkSpec(
    benchmark="breizhcrops",
    population=608_263,
    purge_km=5.0,
    geographic_targets=("frh01", "frh02", "frh03", "frh04"),
    id_kind="sample",
    notes="all 9 published classes retained; regional singleton -> target test, reported not deleted.",
)

#: Release official anchor: FRH01+FRH02 train, FRH03 val, FRH04 test (identical across seeds).
BREIZHCROPS_OFFICIAL = {"train": 319_258, "val": 166_391, "test": 122_614}


# ---- PASTIS-R ------------------------------------------------------------- #
#: Sentinel tile patch counts. Tiles are the geographic unit (every published fold spans all four
#: tiles, so folds cannot serve as geographic units). Requires EPSG:2154 -> EPSG:4326 transform.
PASTIS_TILE_PATCHES: dict[str, int] = {
    "T30UXV": 531, "T31TFJ": 623, "T31TFM": 723, "T32ULU": 556,
}

PASTIS = BenchmarkSpec(
    benchmark="pastis",
    population=2_433,
    purge_km=2.0,
    geographic_targets=("T30UXV", "T31TFJ", "T31TFM", "T32ULU"),
    id_kind="patch",
    notes="allocation over patch IDs only; pixels streamed after; iterative multilabel stratification.",
)

#: Release official anchor: folds 1-3 train, fold 4 val, fold 5 test (identical across seeds).
PASTIS_OFFICIAL = {"train": 1_455, "val": 482, "test": 496}
#: Void segmentation class ignored when building patch class-presence vectors.
PASTIS_VOID_CLASS: int = 19


ALL_SPECS: dict[str, BenchmarkSpec] = {
    s.benchmark: s for s in (CROPHARVEST, EUROCROPML, BREIZHCROPS, PASTIS)
}
