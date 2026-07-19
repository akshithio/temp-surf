"""PASTIS-R crop-type semantic segmentation."""

from __future__ import annotations

import json
import os
import re
from collections.abc import Iterable
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np
from pyproj import Transformer  # projected PASTIS coordinates require a real CRS transform (declared directly)

from dataio.get_input import (
    Benchmark,
    ModalitySeries,
    NativeSeries,
    _synthetic_month_doy,
)
from evals.metrics import score_segmentation_streamed
from evals.probes import fit_probe_multiclass
from utils import perfutils as perf

BENCHMARK = "pastis"
LABEL_KIND = "segmentation"
TRAIN_FOLDS = {1, 2, 3}
VAL_FOLDS = {4}
TEST_FOLDS = {5}
HOLDOUTS = [5]
OFFICIAL_HOLDOUTS = [5]
GEOGRAPHIC_HOLDOUTS = [1, 2, 3, 4, 5]
SPLIT_REGIMES = ["random_id", "official", "geographic_ood", "spatial_cluster_ood"]
# spatial_cluster_ood: coordinate-only spherical-K-means cells over patch centroids (5 cells,
# purge_km from split_spec); no benchmark-specific override -- see evals.regimes.spatial_cluster_ood.
IGNORE_INDEX = 19

# ANNOTATIONS/TARGET_<id>.npy (20 classes: 0=background, 1-18 crops, 19=void).
PASTIS_S2_BANDS = ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"]
PASTIS_S1_BANDS = ["VV", "VH", "VV/VH"]
PASTIS_N_CLASSES = 20
PASTIS_TIMESTEPS = 12
PASTIS_TILE_SIZE = 64


@dataclass(frozen=True)
class PastisPatch:
    """One PASTIS-R patch."""

    patch_id: int
    fold: int
    s2_path: Path
    s1_path: Path
    target_path: Path
    s2_months: np.ndarray
    s1_months: np.ndarray
    latlon: tuple[float, float]
    #: Sentinel-2 granule tile (canonical ``T##XXX``), the geographic unit for tile-LODO. ``None``
    #: only if the metadata carries no ``TILE`` property.
    tile: str | None = None


@dataclass(frozen=True)
class PastisTile:
    """One native-cadence 64x64 PASTIS-R tile."""

    s2: np.ndarray
    s1: np.ndarray
    s2_months: np.ndarray
    s1_months: np.ndarray
    s2_mask: np.ndarray
    s1_mask: np.ndarray
    labels: np.ndarray
    valid: np.ndarray
    fold: int
    latlon: tuple[float, float]
    s2_only: bool = False

    @property
    def height(self) -> int:
        return int(self.labels.shape[0])

    @property
    def width(self) -> int:
        return int(self.labels.shape[1])

    def pixel_benchmark(self) -> Benchmark:
        bench = _pastis_pixel_benchmark(
            self.s2,
            self.s1,
            self.s2_months,
            self.s1_months,
            self.labels.reshape(-1),
            self.valid.reshape(-1),
            self.fold,
            self.latlon,
        )
        # S2-only view for pixel models (Presto, TESSERA, raw): route through the tabular
        # ``Benchmark.s2_only`` contract, which makes S1 a *structurally absent* modality and zeroes
        # coordinates --- not a zero-valued "present" S1. Matches EuroCropsML/BreizhCrops exactly.
        return bench.s2_only() if self.s2_only else bench


@dataclass(frozen=True)
class PastisBenchmark:
    """Lazy PASTIS-R release descriptor."""

    name: str
    label_kind: str
    patches: tuple[PastisPatch, ...]
    tile_size: int = 64
    ignore_index: int = IGNORE_INDEX
    data_quality: dict[str, Any] = field(default_factory=dict)
    s2_only_mode: bool = False

    def s2_only(self) -> PastisBenchmark:
        """Common-input S2-only view: make Sentinel-1 a *structurally absent* modality (not a
        zero-valued "present" one) on every tile, and zero coordinates, so cross-model differences
        can't be attributed to S1/coordinate access. Enforced per model, not by munging data:
        dense S1+S2 fusers (Galileo) mask the S1 group ``MISSING`` for every timestep, and pixel
        models (Presto/TESSERA/raw) route ``pixel_benchmark`` through :meth:`Benchmark.s2_only`.
        Dense S2-only encoders (OlmoEarth, AgriFM) already ignore S1, so this is a no-op for them.
        The caller isolates its cache/results via a ``__s2only`` tag."""
        return replace(self, s2_only_mode=True)

    @property
    def n_samples(self) -> int:
        tiles_per_axis = 128 // self.tile_size
        return len(self.patches) * tiles_per_axis * tiles_per_axis

    @property
    def groups(self) -> np.ndarray:
        tiles_per_patch = (128 // self.tile_size) ** 2
        return np.repeat([patch.fold for patch in self.patches], tiles_per_patch).astype(np.int64)

    @property
    def patch_latlon(self) -> dict[int, tuple[float, float]]:
        return {patch.patch_id: patch.latlon for patch in self.patches}

    @property
    def patch_tiles(self) -> dict[int, str | None]:
        """Patch id -> canonical Sentinel tile (``T##XXX``). The geographic unit for tile-LODO."""
        return {patch.patch_id: patch.tile for patch in self.patches}

    def tiles(self) -> list[str]:
        """Sorted distinct Sentinel tiles present (excluding patches with no tile metadata)."""
        return sorted({p.tile for p in self.patches if p.tile is not None})

    @property
    def latlon(self) -> np.ndarray:
        tiles_per_patch = (128 // self.tile_size) ** 2
        return np.repeat([patch.latlon for patch in self.patches], tiles_per_patch, axis=0).astype(np.float32)

    def patch_ids(self, folds: set[int] | None = None) -> list[int]:
        """Cache-free patch universe: original patch IDs in ``folds`` (or all folds), sorted.

        Equivalent to ``cacheutils.dense_fold_patches(emb_dir, folds)`` on a *complete* cache, but
        derived from the benchmark descriptor instead of cached tile filenames -- so split
        preprocessing never needs an embedding cache. Splitting stays at the original-patch level:
        tiles and pixels are never split units.
        """
        want = None if folds is None else {int(f) for f in folds}
        return sorted(
            int(p.patch_id) for p in self.patches if want is None or int(p.fold) in want
        )

    def patch_class_sets(self, patch_ids: Iterable[int] | None = None) -> dict[int, set[int]]:
        """Cache-free per-patch class sets from the raw ANNOTATIONS/TARGET arrays.

        The set of non-ignore class labels present in each patch. A patch's cached label tiles store
        exactly ``labels[labels != ignore_index]`` per tile, so the union over its tiles equals the
        unique non-ignore labels of the whole target here -- identical to the complete-cache result
        (pinned by test_split_parity.test_pastis_class_sets_are_cache_free_and_matches_complete_cache).
        """
        want = None if patch_ids is None else {int(p) for p in patch_ids}
        out: dict[int, set[int]] = {}
        for patch in self.patches:
            pid = int(patch.patch_id)
            if want is not None and pid not in want:
                continue
            target = np.asarray(np.load(patch.target_path, mmap_mode="r")[0], dtype=np.int64)
            classes = target[target != self.ignore_index]
            out[pid] = {int(c) for c in np.unique(classes)}
        return out

    def iter_tiles(self, folds: set[int] | None = None, cache_root: Path | None = None, overwrite: bool = False):
        """Yield native-cadence tiles."""
        for patch in self.patches:
            if folds is not None and patch.fold not in folds:
                continue
            tile_coords = [
                (row, col)
                for row in range(0, 128, self.tile_size)
                for col in range(0, 128, self.tile_size)
            ]
            if cache_root is not None and not overwrite:
                fold_dir = cache_root / f"fold_{patch.fold}"
                tile_coords = [
                    (row, col)
                    for row, col in tile_coords
                    if not (
                        (fold_dir / f"{patch.patch_id}_{row // self.tile_size}_{col // self.tile_size}.npy").exists()
                        and (fold_dir / f"{patch.patch_id}_{row // self.tile_size}_{col // self.tile_size}.labels.npy").exists()
                    )
                ]
                if not tile_coords:
                    continue
            s2 = np.load(patch.s2_path, mmap_mode="r")  # (T_s2, 10, 128, 128) native cadence
            s1 = np.load(patch.s1_path, mmap_mode="r")  # (T_s1, 3, 128, 128)
            target = np.load(patch.target_path, mmap_mode="r")[0]
            s2_ones = np.ones(s2.shape[0], dtype=np.float32)
            s1_ones = np.ones(s1.shape[0], dtype=np.float32)

            for row, col in tile_coords:
                ys = slice(row, row + self.tile_size)
                xs = slice(col, col + self.tile_size)
                labels = np.asarray(target[ys, xs], dtype=np.int64)
                valid = labels != self.ignore_index
                tile_id = f"{patch.patch_id}_{row // self.tile_size}_{col // self.tile_size}"
                tile = PastisTile(
                    s2=np.asarray(s2[:, :, ys, xs], dtype=np.float32),
                    s1=np.asarray(s1[:, :, ys, xs], dtype=np.float32),
                    s2_months=patch.s2_months,
                    s1_months=patch.s1_months,
                    s2_mask=s2_ones,
                    s1_mask=s1_ones,
                    labels=labels,
                    valid=valid,
                    fold=patch.fold,
                    latlon=patch.latlon,
                    s2_only=self.s2_only_mode,
                )
                yield tile_id, patch.fold, tile, labels[valid]


# --------------------------------------------------------------------------- #
# Internal helpers
# --------------------------------------------------------------------------- #


def _pastis_months(dates_field) -> np.ndarray:
    """dates-S2 {idx: YYYYMMDD} -> 0-indexed calendar month per timestep."""
    d = dates_field if isinstance(dates_field, dict) else json.loads(dates_field)
    items = sorted(d.items(), key=lambda kv: int(kv[0]))
    return np.array([((int(v) // 100) % 100) - 1 for _, v in items], dtype=np.int64)


#: EPSG:2154 (Lambert-93) -> EPSG:4326 (WGS84) transformer, built lazily (the transformer object is
#: heavy to construct, though pyproj itself is imported directly at module top).
_L93_TO_WGS84: Transformer | None = None


def _l93_to_wgs84() -> Transformer:
    global _L93_TO_WGS84
    if _L93_TO_WGS84 is None:
        # always_xy: input (easting, northing) -> output (lon, lat)
        _L93_TO_WGS84 = Transformer.from_crs("EPSG:2154", "EPSG:4326", always_xy=True)
    return _L93_TO_WGS84


def _geometry_latlon(geometry: dict[str, Any] | None) -> tuple[float, float]:
    """Patch representative point as ``(lat, lon)`` in EPSG:4326.

    PASTIS metadata geometries are EPSG:2154 (Lambert-93) easting/northing. The previous loader
    treated them as lon/lat, so every real patch fell outside ``[-180,180]x[-90,90]`` and became
    ``NaN``. Here the projected centroid is transformed to WGS84 first. Coordinates already in valid
    lon/lat (e.g. synthetic fixtures) are kept as-is. A geometry with no coordinates returns
    ``(nan, nan)`` (a MISSING coordinate, hard-failed later by :func:`assert_geographic_ready`); a
    transform that fails or yields out-of-range lat/lon is a hard error here, never a silent NaN.
    """
    coords: list[tuple[float, float]] = []

    def walk(value) -> None:
        if not isinstance(value, list | tuple) or not value:
            return
        if len(value) >= 2 and all(isinstance(v, int | float) for v in value[:2]):
            coords.append((float(value[0]), float(value[1])))
            return
        for child in value:
            walk(child)

    walk((geometry or {}).get("coordinates", []))
    if not coords:
        return (np.nan, np.nan)  # missing geometry -- caught by assert_geographic_ready, not silent
    # Bounding-box center: an unbiased representative point (a polygon ring repeats its first vertex,
    # which would skew a plain mean). For PASTIS's square patches this is the exact center.
    xs = [c[0] for c in coords]
    ys = [c[1] for c in coords]
    x = (min(xs) + max(xs)) / 2.0  # easting or lon
    y = (min(ys) + max(ys)) / 2.0  # northing or lat
    # Already WGS84 lon/lat (synthetic fixtures declare geometries directly in degrees).
    if -180.0 <= x <= 180.0 and -90.0 <= y <= 90.0:
        return (y, x)
    # Otherwise Lambert-93 easting/northing -> transform the projected centroid to lon/lat. A failed
    # or out-of-range transform is a hard error (unusable projected coordinate), never a silent NaN.
    try:
        lon, lat = _l93_to_wgs84().transform(x, y)
    except Exception as exc:
        raise ValueError(f"PASTIS EPSG:2154->4326 transform failed for easting/northing ({x}, {y}): {exc}") from exc
    if not (-90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0):
        raise ValueError(f"PASTIS transform produced out-of-range lat/lon ({lat}, {lon}) from ({x}, {y})")
    return (float(lat), float(lon))


_TILE_RE = re.compile(r"^T\d{2}[A-Z]{3}$")  # canonical Sentinel-2 granule tile, e.g. T31TFM


def _canonical_tile(raw: Any) -> str | None:
    """Normalize a metadata ``TILE`` value to canonical ``T##XXX`` (e.g. ``30UXV`` -> ``T30UXV``).

    Returns ``None`` for an absent/blank value (a MISSING tile, hard-failed later by
    :func:`assert_geographic_ready`). A present-but-malformed value raises -- it is invalid metadata,
    not a geographic-ready state.
    """
    if raw is None:
        return None
    s = str(raw).strip().upper()
    if not s:
        return None
    if not s.startswith("T"):
        s = "T" + s
    if not _TILE_RE.match(s):
        raise ValueError(f"PASTIS tile {raw!r} is not a canonical Sentinel granule tile (T##XXX)")
    return s


def assert_geographic_ready(bench: PastisBenchmark) -> None:
    """Hard-fail if any patch lacks a valid Sentinel tile or a finite coordinate.

    Tile-LODO geographic split generation MUST call this: a patch with missing/invalid tile or
    coordinate metadata cannot be placed in a geographic unit and must never be silently dropped.
    """
    no_tile = [p.patch_id for p in bench.patches if p.tile is None]
    bad_coord = [
        p.patch_id for p in bench.patches
        if not np.all(np.isfinite(np.asarray(p.latlon, dtype=float)))
    ]
    problems: list[str] = []
    if no_tile:
        problems.append(f"{len(no_tile)} patch(es) have no Sentinel tile (e.g. {no_tile[:5]})")
    if bad_coord:
        problems.append(f"{len(bad_coord)} patch(es) have non-finite coordinates (e.g. {bad_coord[:5]})")
    if problems:
        raise ValueError("PASTIS is not geographic-ready: " + "; ".join(problems))


def assert_frozen_tile_universe(bench: PastisBenchmark, expected: dict[str, int] | None = None) -> None:
    """Validate tile assignments against the FROZEN tile universe + counts before geographic/spatial
    split construction. Rejects missing (``None``), unexpected (out-of-universe), duplicated patch
    IDs, and per-tile counts that differ from the expected frozen counts (default
    :data:`split_spec.PASTIS_TILE_PATCHES`). Any mismatch is a hard error, never a silent drop.
    """
    from collections import Counter

    from evals.split_spec import PASTIS_TILE_PATCHES

    expected = dict(PASTIS_TILE_PATCHES if expected is None else expected)
    problems: list[str] = []

    id_counts = Counter(int(p.patch_id) for p in bench.patches)
    dups = sorted(i for i, c in id_counts.items() if c > 1)
    if dups:
        problems.append(f"duplicate patch id(s): {dups[:5]}")

    no_tile = [p.patch_id for p in bench.patches if p.tile is None]
    if no_tile:
        problems.append(f"{len(no_tile)} patch(es) with no tile (e.g. {no_tile[:5]})")

    actual = Counter(p.tile for p in bench.patches if p.tile is not None)
    unexpected = sorted(set(actual) - set(expected))
    if unexpected:
        problems.append(f"unexpected tile(s) outside the frozen universe: {unexpected}")
    for tile, exp_n in sorted(expected.items()):
        got = actual.get(tile, 0)
        if got != exp_n:
            problems.append(f"tile {tile}: {got} patch(es) != expected {exp_n}")

    if problems:
        raise ValueError("PASTIS frozen tile universe invalid: " + "; ".join(problems))


def _monthly_patch(values: np.ndarray, months: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Aggregate a variable-length ``(T,C,H,W)`` patch into calendar months."""
    if len(months) != values.shape[0]:
        raise ValueError(f"Date count {len(months)} does not match observations {values.shape[0]}")
    out = np.zeros((PASTIS_TIMESTEPS,) + values.shape[1:], dtype=np.float32)
    mask = np.zeros(PASTIS_TIMESTEPS, dtype=np.float32)
    for month in range(PASTIS_TIMESTEPS):
        selected = months == month
        if selected.any():
            out[month] = np.asarray(values[selected], dtype=np.float32).mean(axis=0)
            mask[month] = 1.0
    return out, mask


def _pastis_pixel_benchmark(
    s2: np.ndarray,
    s1: np.ndarray,
    s2_months: np.ndarray,
    s1_months: np.ndarray,
    labels: np.ndarray,
    valid: np.ndarray,
    fold: int,
    latlon: tuple[float, float],
) -> Benchmark:
    _, _, height, width = s2.shape
    t_s2, t_s1 = s2.shape[0], s1.shape[0]
    s2_pixels = s2.transpose(2, 3, 0, 1).reshape(height * width, t_s2, -1)[valid]
    s1_pixels = s1.transpose(2, 3, 0, 1).reshape(height * width, t_s1, -1)[valid].astype(np.float32)
    red = s2_pixels[:, :, PASTIS_S2_BANDS.index("B4")]
    nir = s2_pixels[:, :, PASTIS_S2_BANDS.index("B8")]
    ndvi = np.divide(nir - red, nir + red, out=np.zeros_like(red), where=(nir + red) != 0)
    s2_pixels = np.concatenate([s2_pixels, ndvi[:, :, None]], axis=2).astype(np.float32)
    n = int(valid.sum())
    doy_tbl = _synthetic_month_doy(12)

    def _modality(pixels: np.ndarray, months: np.ndarray, bands: list[str]) -> ModalitySeries:
        months = np.asarray(months, dtype=np.int64)
        doy = doy_tbl[months % 12].astype(np.float32)
        years = np.full(len(months), 2019, dtype=np.int64)
        return ModalitySeries([pixels[i] for i in range(n)], [months] * n, [doy] * n, [years] * n, bands)

    native = NativeSeries(
        s2=_modality(s2_pixels, s2_months, PASTIS_S2_BANDS + ["NDVI"]),
        s1=_modality(s1_pixels, s1_months, list(PASTIS_S1_BANDS)),
        climate=ModalitySeries.absent(n),
    )
    return Benchmark(
        name="pastis",
        label_kind="segmentation",
        native=native,
        labels=labels[valid].astype(np.int64),
        groups=np.full(n, fold, dtype=np.int64),
        latlon=np.repeat(np.asarray([latlon], dtype=np.float32), n, axis=0),
        years=np.full(n, 2019, dtype=np.int64),
    )


def load_benchmark(
    root: Path = Path("data/input/benchmarks"),
    max_samples: int | None = None,
    shuffle: bool = True,
    seed: int = 0,
    folds: list[int] | None = None,
) -> PastisBenchmark:
    """Build a lazy PASTIS-R descriptor using the official five folds.

    S2 and ascending-orbit S1 are monthly aggregated only when a 64x64 tile is
    requested. ``max_samples`` limits source patches, not pixels or tiles.
    """
    base = root / "pastis"
    if not (base / "metadata.geojson").exists():
        raise FileNotFoundError(f"PASTIS metadata not found: {base / 'metadata.geojson'}")
    geo = json.loads((base / "metadata.geojson").read_text())
    rows = [{**feature["properties"], "_latlon": _geometry_latlon(feature.get("geometry"))} for feature in geo["features"]]
    if folds:
        rows = [row for row in rows if int(row["Fold"]) in folds]
    order = np.arange(len(rows))
    if shuffle:
        order = np.random.default_rng(seed).permutation(order)
    if max_samples:
        order = order[:max_samples]

    patches: list[PastisPatch] = []
    missing_records: list[dict[str, Any]] = []
    for i in order:
        r = rows[int(i)]
        pid = int(r["ID_PATCH"])
        s2_path = base / "DATA_S2" / f"S2_{pid}.npy"
        s1_path = base / "DATA_S1A" / f"S1A_{pid}.npy"
        target_path = base / "ANNOTATIONS" / f"TARGET_{pid}.npy"
        if not (s2_path.exists() and s1_path.exists() and target_path.exists()):
            missing_records.append({"patch_id": pid, "reason": "missing s2/s1/target npy"})
            continue
        patches.append(
            PastisPatch(
                patch_id=pid,
                fold=int(r["Fold"]),
                s2_path=s2_path,
                s1_path=s1_path,
                target_path=target_path,
                s2_months=_pastis_months(r["dates-S2"]),
                s1_months=_pastis_months(r["dates-S1A"]),
                latlon=r["_latlon"],
                tile=_canonical_tile(r.get("TILE")),
            )
        )

    if missing_records:
        msg = f"PASTIS: {len(missing_records)}/{len(order)} metadata patches have missing .npy files in {base}"
        if os.environ.get("STRICT_MODE", "").strip().lower() not in ("", "0", "false", "no"):
            raise ValueError(msg + " (STRICT_MODE is set)")
        print(f"   !! {msg} -- those patches are skipped (set STRICT_MODE=True to fail instead)", flush=True)
    if not patches:
        raise ValueError(f"No PASTIS patches parsed from {base}")
    return PastisBenchmark(
        name="pastis",
        label_kind="segmentation",
        patches=tuple(patches),
        tile_size=PASTIS_TILE_SIZE,
        ignore_index=IGNORE_INDEX,
        data_quality={"skipped_inputs": missing_records} if missing_records else {},
    )


def make_targets(bench) -> tuple[np.ndarray, np.ndarray]:
    """Dense targets are loaded tile-wise by the segmentation runner."""
    return np.empty(0, dtype=np.int64), bench.groups


def _validate_source_budgets(budgets: list[float | int]) -> None:
    bad = [budget for budget in budgets if float(budget) <= 0.0 or float(budget) > 1.0]
    if bad:
        raise ValueError(f"source budgets must be fractions in (0, 1]; invalid: {bad}")


def run_probes_segmentation(
    rows: list[dict[str, Any]],
    x_train: np.ndarray,
    x_val: np.ndarray,
    y_train: np.ndarray,
    y_val: np.ndarray,
    seed: int,
    *,
    eval_streams: dict[str, Any],
    budgets: list[float],
    meta: dict[str, Any] | None = None,
    groups_train: np.ndarray | None = None,
    family: str = "logistic",
) -> None:
    _validate_source_budgets(budgets)
    meta = dict(meta or {})
    eval_classes = np.arange(19, dtype=np.int64)
    for budget in budgets:
        sub_seed = perf._budget_seed(seed, budget)
        sub = perf.subset_indices(y_train, budget, sub_seed, stratify=True)
        # ERM: the dense probe fits the cached features as they are.
        x_train_t, x_val_t = x_train, x_val
        clf, probe_meta = fit_probe_multiclass(
            x_train_t[sub],
            y_train[sub],
            sub_seed,
            x_val=x_val_t,
            y_val=y_val,
            family=family,
        )
        for split_name, tiles in eval_streams.items():
            rows.append({
                **meta,
                "evaluation_split": split_name,
                "budget_type": "source",
                "label_budget": budget,
                "seed": seed,
                "n_train_sub": int(len(sub)),
                **probe_meta,
                **score_segmentation_streamed(clf, tiles(), eval_classes),
            })


def run_probes_segmentation_target(
    rows: list[dict[str, Any]],
    x_source: np.ndarray,
    y_source: np.ndarray,
    seed: int,
    *,
    target_patches: Any,
    sample_target: Any,
    stream_target: Any,
    x_val: np.ndarray,
    y_val: np.ndarray,
    budgets: list[int | float],
    meta: dict[str, Any] | None = None,
    family: str = "logistic",
    groups_source: np.ndarray | None = None,
    target_id_budget: float | int = -1,
    pool_patches: Any = None,
    target_test_patches: Any = None,
) -> None:
    meta = dict(meta or {})
    eval_classes = np.arange(19, dtype=np.int64)
    split_seed = perf._budget_seed(seed, 0.5)
    if pool_patches is not None and target_test_patches is not None:
        # schema v2: the FROZEN target_label_pool / target_test patch sets from the artifact -- few-shot
        # patches are drawn ONLY from the pool and every budget is scored on the fixed target_test.
        test_patches = {int(p) for p in target_test_patches}
        pool_sorted = sorted(int(p) for p in pool_patches)
        pool_order = [int(p) for p in np.random.default_rng(split_seed).permutation(pool_sorted).tolist()]
        all_patches = test_patches | set(pool_sorted)
        degenerate = len(pool_order) == 0 or len(test_patches) == 0
    else:
        patches = np.array(sorted({int(p) for p in target_patches}))
        all_patches = set(patches.tolist())
        degenerate = len(patches) < 2
        perm = np.random.default_rng(split_seed).permutation(patches)
        n_test_patches = max(1, int(round(0.2 * len(patches)))) if not degenerate else len(patches)
        test_patches = set(perm[:n_test_patches].tolist())
        pool_order = [int(p) for p in perm.tolist() if p not in test_patches]

    for budget in budgets:
        if budget != 0 and (degenerate or not pool_order):
            continue
        sub_seed = perf._budget_seed(seed, budget)
        cal_x_raw, cal_y, tune_internal = x_val, y_val, False
        if budget == 0:
            x_tr_raw, y_tr = x_source, y_source
            n_patch_train = 0
        elif budget == target_id_budget:
            sampled = sample_target(set(pool_order), sub_seed)
            x_tr_raw, y_tr = sampled[:2]
            cal_x_raw, cal_y, tune_internal = None, None, True
            n_patch_train = len(pool_order)
        else:
            k = min(len(pool_order), perf._target_budget_count(budget, len(pool_order)))
            sampled = sample_target(set(pool_order[:k]), sub_seed)
            xf, yf = sampled[:2]
            x_tr_raw = np.concatenate([x_source, xf])
            y_tr = np.concatenate([y_source, yf])
            n_patch_train = k

        x_tr = x_tr_raw
        cal_x = cal_x_raw if cal_x_raw is not None and len(cal_x_raw) else None
        clf, probe_meta = fit_probe_multiclass(
            x_tr,
            y_tr,
            sub_seed,
            x_val=cal_x,
            y_val=cal_y,
            family=family,
            tune_internal=tune_internal,
        )
        rows.append({
            **meta,
            "evaluation_split": "held_out",
            "budget_type": "target",
            "label_budget": budget,
            "label_budget_unit": "target_patches",
            "n_target_patches_train": n_patch_train,
            "n_target_patches_test": len(test_patches),
            "seed": seed,
            "n_train_sub": int(len(y_tr)),
            **probe_meta,
            **score_segmentation_streamed(clf, stream_target(test_patches), eval_classes),
        })
        if budget == 0:
            rows.append({
                **meta,
                "evaluation_split": "full",
                "budget_type": "target",
                "label_budget": budget,
                "label_budget_unit": "target_patches",
                "n_target_patches_train": n_patch_train,
                "n_target_patches_test": len(all_patches),
                "seed": seed,
                "n_train_sub": int(len(y_tr)),
                **probe_meta,
                **score_segmentation_streamed(clf, stream_target(all_patches), eval_classes),
            })
