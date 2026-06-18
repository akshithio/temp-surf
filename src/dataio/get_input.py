"""Benchmark input loading for the frozen-embedding robustness pipeline.

Loads raw agricultural EO benchmarks into a single dense in-memory container
(:class:`Benchmark`) that every encoder in ``src/models/*`` can consume: each
encoder applies the stress conditions (sensor-off / temporal-drop) to these
arrays and produces embeddings. Nothing here touches a model or a degradation op.

Expected layout under ``data/input/benchmarks/`` (datasets staged here by you)::

    data/input/benchmarks/cropharvest/
        labels.geojson
        features/arrays/<index>_<dataset>.h5      # one (T, C) array per sample
    data/input/benchmarks/eurocropsml/
        preprocess/*.npz                          # one (T, 13) array per parcel
        split/latvia_portugal_vs_estonia/...      # official transnational split

CropHarvest is rebuilt from the raw per-sample h5 arrays (band layout below);
EuroCropsML is read from the preprocessed npz parcels.

Band layouts follow the nasaharvest/cropharvest convention and the EuroCropsML
13-band ordering so embeddings stay comparable across pipelines.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from datetime import datetime, timedelta
from pathlib import Path

import h5py
import numpy as np
from sklearn.preprocessing import LabelEncoder

DEFAULT_ROOT = Path("data/input/benchmarks")

# --- CropHarvest band layout (raw array column indices) ---------------------
# Raw 18-col array: [S1 VV,VH] + [S2: B2..B8A, B9, B11, B12] + [ERA5 temp,precip]
#   + [SRTM elevation(15), slope(16)] + [NDVI(17)].
CH_S1_IDXS = [0, 1]
CH_S2_IDXS = [2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 17]  # B2..B12 (no B1/B10) + NDVI (col 17)
CH_CLIMATE_IDXS = [13, 14, 15, 16]  # temperature, precipitation, elevation, slope (slope = Presto's SRTM band)
CH_MIN_CHANNELS = max(CH_S1_IDXS + CH_S2_IDXS + CH_CLIMATE_IDXS) + 1
CH_S1_BANDS = ["VV", "VH"]
CH_S2_BANDS = ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12", "NDVI"]
CH_CLIMATE_BANDS = ["temperature", "precipitation", "elevation", "slope"]
CH_TIMESTEPS = 12

# --- EuroCropsML band layout (raw npz column indices) -----------------------
# VERIFIED (2026-06-14) against the staged npz: the 13 columns are the native Sentinel-2
# bands B1,B2,B3,B4,B5,B6,B7,B8,B8A,B9,B10,B11,B12 -- there is NO SCL column (the last
# column is B12: range 0-~3800 with thousands of unique values, not 0-11 class labels),
# and col 10 is B10/cirrus (mean ~34, median 15, >90% < 100 -- the near-zero giveaway).
# So from B10 on, the old assumed order (.,B9,B11,B12,SCL) was shifted by one band. Keep
# B2..B8A (idx 1-8) + B11,B12 (idx 11,12) and append a computed NDVI. B1/B9/B10 are dropped
# (no encoder uses them). No native S1 / climate.
EC_S2_IDXS = [1, 2, 3, 4, 5, 6, 7, 8, 11, 12]
EC_B4_IDX = 3  # B4 (red), native order
EC_B8_IDX = 7  # B8 (NIR), native order
EC_S2_BANDS = ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12", "NDVI"]
EC_COUNTRY_PREFIX = {"EE": "Estonia", "LV": "Latvia", "PT": "Portugal"}


@dataclass
class Benchmark:
    """Dense, in-memory multimodal pixel/parcel time series for one benchmark.

    Modality arrays are ``(N, T, C)`` with the spatial 1x1 dimension squeezed out.
    ``*_mask`` are ``(N, T)`` per-timestep availability (1 = observed). ``doy`` is
    ``(N, T)`` day-of-year. ``labels`` are the task target (binary is_crop or
    encoded class id); ``groups`` are the strict-holdout group (dataset for
    CropHarvest, country for EuroCropsML).
    """

    name: str
    task: str  # "binary" | "multiclass" | "regression"
    s2: np.ndarray
    s1: np.ndarray
    climate: np.ndarray
    s2_mask: np.ndarray
    s1_mask: np.ndarray
    climate_mask: np.ndarray
    doy: np.ndarray
    labels: np.ndarray
    groups: np.ndarray
    latlon: np.ndarray  # (N, 2) as [lat, lon]; used by location-aware encoders (Presto, ...)
    s2_bands: list[str]
    s1_bands: list[str]
    climate_bands: list[str]
    label_names: list[str] | None = None  # class id -> name, for multiclass
    years: np.ndarray | None = None  # (N,) calendar year of each sample's observation window

    @property
    def n_samples(self) -> int:
        return int(self.s2.shape[0])

    @property
    def timesteps(self) -> int:
        return int(self.s2.shape[1])


@dataclass
class SpatialBenchmark:
    """Dense, in-memory multimodal *patch* time series for a segmentation benchmark.

    Unlike :class:`Benchmark` (1x1 pixel/parcel series, ``(N, T, C)``), modality arrays
    here keep the spatial dimensions: ``(N, T, C, H, W)`` patches. ``labels`` is a
    per-pixel class mask ``(N, H, W)``; ``ignore_index`` is the class id excluded from the
    probe and from mIoU (background / void). ``groups`` is the per-patch fold id used for
    the geographic-style holdout (PASTIS ships 5 spatially-disjoint folds). ``*_mask`` are
    ``(N, T)`` per-timestep availability. Consumed by encoders' ``encode_dense``.
    """

    name: str
    task: str  # "segmentation"
    s2: np.ndarray  # (N, T, C2, H, W)
    s1: np.ndarray  # (N, T, C1, H, W)  (empty C1=0 if absent)
    s2_mask: np.ndarray  # (N, T)
    s1_mask: np.ndarray  # (N, T)
    doy: np.ndarray  # (N, T)
    labels: np.ndarray  # (N, H, W) per-pixel class id
    groups: np.ndarray  # (N,) fold id per patch
    s2_bands: list[str]
    s1_bands: list[str]
    label_names: list[str] | None = None  # class id -> name
    ignore_index: int = 0  # class excluded from probe + mIoU (PASTIS: 0=background, 19=void)
    void_index: int | None = None  # second class to drop (PASTIS void)
    years: np.ndarray | None = None  # (N,) calendar year per patch

    @property
    def n_samples(self) -> int:
        return int(self.s2.shape[0])

    @property
    def timesteps(self) -> int:
        return int(self.s2.shape[1])

    @property
    def hw(self) -> tuple[int, int]:
        return (int(self.s2.shape[-2]), int(self.s2.shape[-1]))


@dataclass(frozen=True)
class PastisPatch:
    """Paths and temporal metadata for one lazily loaded PASTIS-R patch."""

    patch_id: int
    fold: int
    s2_path: Path
    s1_path: Path
    target_path: Path
    s2_months: np.ndarray
    s1_months: np.ndarray


@dataclass(frozen=True)
class PastisTile:
    """One monthly 64x64 PASTIS-R tile passed to dense encoders."""

    s2: np.ndarray
    s1: np.ndarray
    s2_mask: np.ndarray
    s1_mask: np.ndarray
    labels: np.ndarray
    valid: np.ndarray
    fold: int

    @property
    def height(self) -> int:
        return int(self.labels.shape[0])

    @property
    def width(self) -> int:
        return int(self.labels.shape[1])

    def pixel_benchmark(self) -> Benchmark:
        return _pastis_pixel_benchmark(
            self.s2,
            self.s1,
            self.s2_mask,
            self.s1_mask,
            self.labels.reshape(-1),
            self.valid.reshape(-1),
            self.fold,
        )


@dataclass(frozen=True)
class PastisBenchmark:
    """Lazy PASTIS-R release descriptor.

    The release is roughly 69 GB and its monthly S2 tensor alone would occupy
    about 19 GB in memory.  This object therefore stores only file records.
    ``iter_tiles`` materializes one 64x64 tile at a time.
    """

    name: str
    task: str
    patches: tuple[PastisPatch, ...]
    tile_size: int = 64
    sensor_off: str = "none"
    temporal_drop: float = 0.0
    degradation_seed: int = 0
    ignore_index: int = 19

    @property
    def n_samples(self) -> int:
        tiles_per_axis = 128 // self.tile_size
        return len(self.patches) * tiles_per_axis * tiles_per_axis

    @property
    def groups(self) -> np.ndarray:
        tiles_per_patch = (128 // self.tile_size) ** 2
        return np.repeat([patch.fold for patch in self.patches], tiles_per_patch).astype(np.int64)

    def iter_tiles(self, folds: set[int] | None = None):
        """Yield ``(tile_id, fold, pixel_benchmark, labels)`` lazily.

        Pixels with the PASTIS void label (19) are removed. Background (0) is
        retained because it is an evaluated class in the published protocol.
        """
        for patch in self.patches:
            if folds is not None and patch.fold not in folds:
                continue
            s2 = np.load(patch.s2_path, mmap_mode="r")
            s1 = np.load(patch.s1_path, mmap_mode="r")
            target = np.load(patch.target_path, mmap_mode="r")[0]
            s2_monthly, s2_mask = _monthly_patch(s2, patch.s2_months)
            s1_monthly, s1_mask = _monthly_patch(s1, patch.s1_months)

            for row in range(0, target.shape[0], self.tile_size):
                for col in range(0, target.shape[1], self.tile_size):
                    ys = slice(row, row + self.tile_size)
                    xs = slice(col, col + self.tile_size)
                    labels = np.asarray(target[ys, xs], dtype=np.int64)
                    valid = labels != self.ignore_index
                    tile_id = f"{patch.patch_id}_{row // self.tile_size}_{col // self.tile_size}"
                    tile = PastisTile(
                        s2=s2_monthly[:, :, ys, xs],
                        s1=s1_monthly[:, :, ys, xs],
                        s2_mask=s2_mask,
                        s1_mask=s1_mask,
                        labels=labels,
                        valid=valid,
                        fold=patch.fold,
                    )
                    if self.sensor_off != "none" or self.temporal_drop:
                        tile_seed = self.degradation_seed + patch.patch_id + row * 128 + col
                        tile = _degrade_pastis_tile(tile, self.sensor_off, self.temporal_drop, tile_seed)
                    yield tile_id, patch.fold, tile, labels[valid]


# --------------------------------------------------------------------------- #
# Stress degradation
# --------------------------------------------------------------------------- #


def degrade(
    bench: Benchmark | PastisBenchmark,
    sensor_off: str = "none",
    temporal_drop: float = 0.0,
    seed: int = 0,
) -> Benchmark | PastisBenchmark:
    """Apply a stress condition to a benchmark, returning a degraded copy.

    This realizes the protocol conditions defined in ``src/evals/evals.py`` but
    takes the primitives directly (``sensor_off`` in {none, s2, s1, climate};
    ``temporal_drop`` in [0, 1)) so this module stays independent of the eval
    layer -- the caller maps a named condition to these args.

    * sensor-off zeros the modality and its availability mask.
    * temporal-drop randomly zeros a fraction of timesteps across *all* modalities
      (timestep 0 is always kept, and at least two timesteps survive), mirroring
      the extraction-time degradation so embeddings stay comparable.
    """
    if isinstance(bench, PastisBenchmark):
        if sensor_off not in {"none", "s2", "s1"}:
            raise ValueError(f"PASTIS-R does not provide modality {sensor_off!r}")
        return replace(
            bench,
            sensor_off=sensor_off,
            temporal_drop=temporal_drop,
            degradation_seed=seed,
        )

    s2, s1, climate = bench.s2.copy(), bench.s1.copy(), bench.climate.copy()
    s2_mask, s1_mask, climate_mask = bench.s2_mask.copy(), bench.s1_mask.copy(), bench.climate_mask.copy()

    if sensor_off == "s2":
        s2[:] = 0.0
        s2_mask[:] = 0.0
    elif sensor_off == "s1":
        s1[:] = 0.0
        s1_mask[:] = 0.0
    elif sensor_off == "climate":
        climate[:] = 0.0
        climate_mask[:] = 0.0
    elif sensor_off != "none":
        raise ValueError(f"Unknown sensor_off={sensor_off!r}")

    if temporal_drop > 0.0:
        n, t = bench.s2.shape[:2]
        keep_prob = max(0.0, min(1.0, 1.0 - temporal_drop))
        rng = np.random.default_rng(seed)
        keep = rng.binomial(1, keep_prob, size=(n, t)).astype(np.float32)
        keep[:, 0] = 1.0  # always keep the first observation
        low = keep.sum(axis=1) < 2
        keep[low, 1] = 1.0  # guarantee >= 2 surviving timesteps
        ks = keep[:, :, None]
        s2 *= ks
        s1 *= ks
        climate *= ks
        s2_mask *= keep
        s1_mask *= keep
        climate_mask *= keep

    return replace(
        bench,
        s2=s2,
        s1=s1,
        climate=climate,
        s2_mask=s2_mask,
        s1_mask=s1_mask,
        climate_mask=climate_mask,
    )


# --------------------------------------------------------------------------- #
# CropHarvest
# --------------------------------------------------------------------------- #


def _synthetic_month_doy(timesteps: int) -> np.ndarray:
    """CropHarvest arrays are monthly-regularized; use a synthetic year."""
    days = [datetime(2000, m, 15).timetuple().tm_yday for m in range(1, timesteps + 1)]
    return np.asarray(days, dtype=np.float32)


def _ch_window_year(export_end_date: str | None) -> int:
    """Calendar year at the MIDDLE of CropHarvest's 12-month window ending at
    ``export_end_date``.

    CropHarvest windows end ~Feb 1, so the 12 months (prev-Feb .. Jan) sit mostly in the
    prior calendar year; the window midpoint (~183 days before the end) selects that
    majority year principledly (e.g. export_end 2021-02-01 -> 2020). 0 if unknown.
    """
    if not export_end_date:
        return 0
    try:
        end = datetime.fromisoformat(str(export_end_date).replace("Z", ""))
    except ValueError:
        return 0
    return int((end - timedelta(days=183)).year)


def _load_ch_labels(labels_geojson: Path) -> dict[tuple[int, str], tuple[int, float, float, int]]:
    """Map (index, dataset) -> (is_crop, lat, lon, window_year)."""
    geo = json.loads(labels_geojson.read_text())
    out: dict[tuple[int, str], tuple[int, float, float, int]] = {}
    for f in geo["features"]:
        p = f["properties"]
        key = (int(p["index"]), str(p["dataset"]))
        lat = float(p["lat"]) if p.get("lat") is not None else float("nan")
        lon = float(p["lon"]) if p.get("lon") is not None else float("nan")
        out[key] = (int(p["is_crop"]), lat, lon, _ch_window_year(p.get("export_end_date")))
    return out


def _select_files(files: list[Path], shuffle: bool, seed: int, max_samples: int | None) -> list[Path]:
    """Deterministically shuffle (so a max_samples subset spans groups/countries) then truncate.

    Shuffle uses a fixed seed so the row order is reproducible -- important because
    cached embeddings are aligned to this order.
    """
    files = sorted(files)
    if shuffle:
        order = np.random.default_rng(seed).permutation(len(files))
        files = [files[i] for i in order]
    if max_samples:
        files = files[:max_samples]
    return files


def load_cropharvest(
    root: Path = DEFAULT_ROOT, max_samples: int | None = None, shuffle: bool = True, seed: int = 0
) -> Benchmark:
    """Rebuild CropHarvest crop/non-crop from raw per-sample h5 arrays."""
    base = Path(root) / "cropharvest"
    arrays_dir = base / "features" / "arrays"
    labels_geojson = base / "labels.geojson"
    if not arrays_dir.exists():
        raise FileNotFoundError(f"CropHarvest arrays not found: {arrays_dir}")
    if not labels_geojson.exists():
        raise FileNotFoundError(f"CropHarvest labels not found: {labels_geojson}")

    label_map = _load_ch_labels(labels_geojson)
    files = _select_files([p for p in arrays_dir.glob("*.h5") if p.is_file()], shuffle, seed, max_samples)

    s2_list, s1_list, clim_list = [], [], []
    labels, groups, latlons, years = [], [], [], []
    for path in files:
        idx_str, dataset = path.stem.split("_", 1)
        key = (int(idx_str), dataset)
        if key not in label_map:
            continue
        try:
            with h5py.File(path, "r") as f:
                if "array" not in f:
                    continue
                arr = np.asarray(f["array"], dtype=np.float32)
        except OSError:
            continue
        if arr.ndim != 2 or arr.shape[0] != CH_TIMESTEPS or arr.shape[1] < CH_MIN_CHANNELS:
            continue
        arr = np.nan_to_num(arr, nan=0.0)
        is_crop, lat, lon, year = label_map[key]
        s2_list.append(arr[:, CH_S2_IDXS])
        s1_list.append(arr[:, CH_S1_IDXS])
        clim_list.append(arr[:, CH_CLIMATE_IDXS])
        labels.append(is_crop)
        groups.append(dataset)
        latlons.append((lat, lon))
        years.append(year)

    if not s2_list:
        raise ValueError(f"No valid CropHarvest arrays parsed from {arrays_dir}")

    n = len(s2_list)
    s2 = np.stack(s2_list).astype(np.float32)
    s1 = np.stack(s1_list).astype(np.float32)
    climate = np.stack(clim_list).astype(np.float32)
    doy = np.tile(_synthetic_month_doy(CH_TIMESTEPS), (n, 1))
    ones = np.ones((n, CH_TIMESTEPS), dtype=np.float32)
    return Benchmark(
        name="cropharvest",
        task="binary",
        s2=s2,
        s1=s1,
        climate=climate,
        s2_mask=ones.copy(),
        s1_mask=ones.copy(),
        climate_mask=ones.copy(),
        doy=doy,
        labels=np.asarray(labels, dtype=np.int64),
        groups=np.asarray(groups, dtype=object),
        latlon=np.asarray(latlons, dtype=np.float32),
        s2_bands=CH_S2_BANDS,
        s1_bands=CH_S1_BANDS,
        climate_bands=CH_CLIMATE_BANDS,
        years=np.asarray(years, dtype=np.int64),
    )


# --------------------------------------------------------------------------- #
# EuroCropsML
# --------------------------------------------------------------------------- #


def _ec_country(stem: str) -> str:
    return EC_COUNTRY_PREFIX.get(stem[:2], stem[:2])


def _resample_to(arr: np.ndarray, dates_ns: np.ndarray, timesteps: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Cap/pad a variable-length parcel series to a fixed length.

    Returns (values (T, C), mask (T,), doy (T,)). Longer series are span-preserving
    linspace-downsampled; shorter ones are zero-padded with mask=0.
    """
    n_t, c = arr.shape
    doy_all = (dates_ns.astype("datetime64[ns]").astype("datetime64[D]")
               - dates_ns.astype("datetime64[ns]").astype("datetime64[Y]").astype("datetime64[D]")
               ).astype(np.int64) + 1
    if n_t >= timesteps:
        take = np.linspace(0, n_t - 1, timesteps).round().astype(np.int64)
        return arr[take].astype(np.float32), np.ones(timesteps, np.float32), doy_all[take].astype(np.float32)
    values = np.zeros((timesteps, c), np.float32)
    values[:n_t] = arr
    mask = np.zeros(timesteps, np.float32)
    mask[:n_t] = 1.0
    doy = np.zeros(timesteps, np.float32)
    doy[:n_t] = doy_all
    return values, mask, doy


EC_N_MONTHS = 12  # EuroCropsML is harmonized to 12 monthly composites (see _monthly_composite)


def _monthly_composite(arr: np.ndarray, dates: np.ndarray, n_months: int = EC_N_MONTHS):
    """Bin a variable-length parcel series into calendar-month composites (mean per month).

    Returns (values (n_months, C), mask (n_months,), doy (n_months,)). This is the harmonized
    temporal protocol: it matches the **monthly cadence Presto/CropHarvest assume** -- Presto's
    positional encoding treats each timestep as one consecutive month, so a sub-monthly grid
    (e.g. 96 acquisitions in a year) would be mis-read as 96 *months*. Month-mean uses ALL
    observations (no lossy linspace subsample); empty months are masked; doy is mid-month.
    EuroCropsML parcels range 1-216 acquisitions/yr (median ~45), so monthly aggregation is
    the principled common grid; encoders that prefer raw observations get this harmonized
    input as a documented limitation (see README).
    """
    months = np.asarray(dates, dtype="datetime64[M]").astype(np.int64) % 12
    c = arr.shape[1]
    values = np.zeros((n_months, c), np.float32)
    mask = np.zeros(n_months, np.float32)
    for m in range(n_months):
        sel = months == m
        if sel.any():
            values[m] = arr[sel].mean(axis=0)
            mask[m] = 1.0
    return values, mask, _synthetic_month_doy(n_months)


def load_eurocropsml(
    root: Path = DEFAULT_ROOT,
    max_samples: int | None = None,
    shuffle: bool = True,
    seed: int = 0,
) -> Benchmark:
    """Load EuroCropsML crop-type parcels as 12 monthly composites (S2 only; S1/climate absent).

    Temporal protocol: each parcel's irregular Sentinel-2 series is aggregated to 12 calendar-
    month composites (``_monthly_composite``) -- the monthly cadence Presto/CropHarvest expect.

    Shuffling matters here: npz filenames sort by country prefix (EE/LV/PT), so a
    sorted max_samples subset would be all-Estonia. Shuffling spans countries,
    which the transnational (Latvia+Portugal -> Estonia) split requires.
    """
    base = Path(root) / "eurocropsml"
    preprocess_dir = base / "preprocess"
    if not preprocess_dir.exists():
        raise FileNotFoundError(f"EuroCropsML preprocess dir not found: {preprocess_dir}")
    files = _select_files([p for p in preprocess_dir.glob("*.npz") if p.is_file()], shuffle, seed, max_samples)
    if not files:
        raise ValueError(f"No EuroCropsML npz files in {preprocess_dir}")

    s2_list, mask_list, doy_list = [], [], []
    label_codes, groups, latlons, years = [], [], [], []
    for path in files:
        with np.load(str(path)) as data:
            raw = data["data"].astype(np.float32)
            dates = data["dates"]
            center = data["center"] if "center" in data else None  # [lon, lat]
        if raw.ndim != 2 or raw.shape[1] < 13 or len(dates) != raw.shape[0]:
            continue
        b4, b8 = raw[:, EC_B4_IDX], raw[:, EC_B8_IDX]
        ndvi = np.divide(b8 - b4, b8 + b4, out=np.zeros_like(b4), where=(b8 + b4) > 0)
        s2_full = np.concatenate([raw[:, EC_S2_IDXS], ndvi[:, None]], axis=1)
        dates_arr = np.asarray(dates)
        values, mask, doy = _monthly_composite(s2_full, dates_arr)
        s2_list.append(values)
        mask_list.append(mask)
        doy_list.append(doy)
        label_codes.append(path.stem.split("_")[-1])
        groups.append(_ec_country(path.stem))
        latlons.append((float(center[1]), float(center[0])) if center is not None else (float("nan"), float("nan")))
        years.append(int(str(dates_arr.ravel()[0])[:4]) if dates_arr.size else 0)

    if not s2_list:
        raise ValueError(f"No valid EuroCropsML parcels parsed from {preprocess_dir}")

    n = len(s2_list)
    encoder = LabelEncoder()
    labels = encoder.fit_transform(label_codes).astype(np.int64)
    s2 = np.stack(s2_list).astype(np.float32)
    s2_mask = np.stack(mask_list).astype(np.float32)
    doy = np.stack(doy_list).astype(np.float32)
    empty_s1 = np.zeros((n, EC_N_MONTHS, 2), np.float32)
    empty_clim = np.zeros((n, EC_N_MONTHS, 0), np.float32)
    zeros_mask = np.zeros((n, EC_N_MONTHS), np.float32)
    return Benchmark(
        name="eurocropsml",
        task="multiclass",
        s2=s2,
        s1=empty_s1,
        climate=empty_clim,
        s2_mask=s2_mask,
        s1_mask=zeros_mask.copy(),
        climate_mask=zeros_mask.copy(),
        doy=doy,
        labels=labels,
        groups=np.asarray(groups, dtype=object),
        latlon=np.asarray(latlons, dtype=np.float32),
        s2_bands=EC_S2_BANDS,
        s1_bands=["VV", "VH"],
        climate_bands=[],
        label_names=list(encoder.classes_),
        years=np.asarray(years, dtype=np.int64),
    )


# --------------------------------------------------------------------------- #
# BreizhCrops  (Brittany crop-type classification, Sentinel-2 L1C, 9 classes)
# --------------------------------------------------------------------------- #

# The breizhcrops package returns L1C ``X`` as (T, 13) with bands in THIS order
# (alphabetical), from which we keep the 10 S2 spectral bands + a computed NDVI.
BZ_X_BANDS = ["B1", "B10", "B11", "B12", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B9"]
BZ_S2_KEEP = ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"]
BZ_S2_BANDS = BZ_S2_KEEP + ["NDVI"]
BZ_REGIONS = ["frh01", "frh02", "frh03", "frh04", "belle-ile"]
BZ_TIMESTEPS = 12


def _resample_fixed(arr: np.ndarray, t: int) -> np.ndarray:
    """Linspace-resample a (T, C) series to exactly ``t`` timesteps (subsample or repeat)."""
    n = arr.shape[0]
    if n == t:
        return arr.astype(np.float32, copy=False)
    idx = np.linspace(0, max(n - 1, 0), t).round().astype(np.int64)
    return arr[idx].astype(np.float32, copy=False)


def _bz_class_names(base: Path) -> list[str] | None:
    """classmapping.csv (cols: ,id,classname,code) -> class names ordered by class id."""
    p = base / "classmapping.csv"
    if not p.exists():
        return None
    try:
        id_to_name: dict[int, str] = {}
        for line in p.read_text().splitlines()[1:]:
            r = line.split(",")
            if len(r) >= 3 and r[1].strip().isdigit():
                id_to_name.setdefault(int(r[1]), r[2])
        return [id_to_name[i] for i in sorted(id_to_name)] if id_to_name else None
    except (OSError, ValueError):
        return None


def load_breizhcrops(
    root: Path = DEFAULT_ROOT,
    max_samples: int | None = None,
    shuffle: bool = True,
    seed: int = 0,
    regions: list[str] | None = None,
) -> Benchmark:
    """Load BreizhCrops L1C S2 time series as 12-step parcels (crop-type classification).

    Uses the ``breizhcrops`` package to read each NUTS-3 region's variable-length L1C
    parcels, keeps the 10 S2 bands + a computed NDVI, rescales reflectance to DN (x1e4,
    matching the other benchmarks), and linspace-resamples each parcel to 12 timesteps with
    a synthetic monthly DOY. ``groups`` = region (geographic holdout; frh04 is the paper's
    test region). S2-only (no S1 / climate).
    """
    try:
        import breizhcrops as bzh
    except ImportError as exc:
        raise ImportError("BreizhCrops needs the `breizhcrops` package (uv pip install breizhcrops).") from exc

    base = Path(root) / "breizhcrops"
    regions = regions or BZ_REGIONS
    col = {b: i for i, b in enumerate(BZ_X_BANDS)}
    keep_idx = [col[b] for b in BZ_S2_KEEP]
    b4i, b8i = col["B4"], col["B8"]

    s2_list: list[np.ndarray] = []
    labels: list[int] = []
    groups: list[str] = []
    for region in regions:
        ds = bzh.BreizhCrops(region, root=str(base), level="L1C", load_timeseries=True, verbose=False)
        for i in range(len(ds)):
            x, y, _ = ds[i]
            x = np.asarray(x, dtype=np.float32)
            if x.ndim != 2 or x.shape[1] < len(BZ_X_BANDS) or x.shape[0] < 1:
                continue
            x = x[:, : len(BZ_X_BANDS)] * 1e4  # reflectance (0-1) -> DN, like the other benchmarks
            b4, b8 = x[:, b4i], x[:, b8i]
            denom = b8 + b4
            ndvi = np.divide(b8 - b4, denom, out=np.zeros_like(b4), where=denom != 0)
            s2 = np.concatenate([x[:, keep_idx], ndvi[:, None]], axis=1)  # (T, 11)
            s2_list.append(_resample_fixed(s2, BZ_TIMESTEPS))
            labels.append(int(y))
            groups.append(region)

    if not s2_list:
        raise ValueError(f"No BreizhCrops parcels parsed from {base}")

    order = np.arange(len(s2_list))
    if shuffle:
        order = np.random.default_rng(seed).permutation(order)
    if max_samples:
        order = order[:max_samples]
    s2 = np.stack([s2_list[i] for i in order]).astype(np.float32)
    labels_arr = np.asarray([labels[i] for i in order], dtype=np.int64)
    groups_arr = np.asarray([groups[i] for i in order], dtype=object)

    n = len(s2)
    doy = np.tile(_synthetic_month_doy(BZ_TIMESTEPS), (n, 1))
    ones = np.ones((n, BZ_TIMESTEPS), dtype=np.float32)
    zeros = np.zeros((n, BZ_TIMESTEPS), np.float32)
    return Benchmark(
        name="breizhcrops",
        task="multiclass",
        s2=s2,
        s1=np.zeros((n, BZ_TIMESTEPS, 2), np.float32),
        climate=np.zeros((n, BZ_TIMESTEPS, 0), np.float32),
        s2_mask=ones.copy(),
        s1_mask=zeros.copy(),
        climate_mask=zeros.copy(),
        doy=doy,
        labels=labels_arr,
        groups=groups_arr,
        latlon=np.full((n, 2), np.nan, np.float32),
        s2_bands=BZ_S2_BANDS,
        s1_bands=["VV", "VH"],
        climate_bands=[],
        label_names=_bz_class_names(base),
        years=np.full(n, 2017, np.int64),
    )


# --------------------------------------------------------------------------- #
# PASTIS-R  (crop-type mapping / semantic segmentation, 128x128 patch time series)
# --------------------------------------------------------------------------- #

# PASTIS DATA_S2 is (T, 10, 128, 128) in this 10-band order. DATA_S1A is
# (T, 3, 128, 128) as VV, VH, VV/VH. The semantic target is channel 0 of
# ANNOTATIONS/TARGET_<id>.npy (20 classes: 0=background, 1-18 crops, 19=void).
PASTIS_S2_BANDS = ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"]
PASTIS_S1_BANDS = ["VV", "VH", "VV/VH"]
PASTIS_N_CLASSES = 20
PASTIS_IGNORE = 19
PASTIS_TIMESTEPS = 12
PASTIS_TILE_SIZE = 64


def _pastis_months(dates_field) -> np.ndarray:
    """dates-S2 {idx: YYYYMMDD} -> 0-indexed calendar month per timestep."""
    d = dates_field if isinstance(dates_field, dict) else json.loads(dates_field)
    items = sorted(d.items(), key=lambda kv: int(kv[0]))
    return np.array([((int(v) // 100) % 100) - 1 for _, v in items], dtype=np.int64)


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


def _degrade_pastis_tile(tile: PastisTile, sensor_off: str, temporal_drop: float, seed: int) -> PastisTile:
    s2, s1 = tile.s2.copy(), tile.s1.copy()
    s2_mask, s1_mask = tile.s2_mask.copy(), tile.s1_mask.copy()
    if sensor_off == "s2":
        s2.fill(0)
        s2_mask.fill(0)
    elif sensor_off == "s1":
        s1.fill(0)
        s1_mask.fill(0)
    elif sensor_off != "none":
        raise ValueError(f"PASTIS-R does not provide modality {sensor_off!r}")
    if temporal_drop:
        rng = np.random.default_rng(seed)
        keep = rng.binomial(1, 1.0 - temporal_drop, size=PASTIS_TIMESTEPS).astype(np.float32)
        keep[0] = 1.0
        if keep.sum() < 2:
            keep[1] = 1.0
        s2 *= keep[:, None, None, None]
        s1 *= keep[:, None, None, None]
        s2_mask *= keep
        s1_mask *= keep
    return replace(tile, s2=s2, s1=s1, s2_mask=s2_mask, s1_mask=s1_mask)


def _pastis_pixel_benchmark(
    s2: np.ndarray,
    s1: np.ndarray,
    s2_mask: np.ndarray,
    s1_mask: np.ndarray,
    labels: np.ndarray,
    valid: np.ndarray,
    fold: int,
) -> Benchmark:
    """Convert one spatial tile to a bounded batch of valid pixel time series."""
    _, _, height, width = s2.shape
    s2_pixels = s2.transpose(2, 3, 0, 1).reshape(height * width, PASTIS_TIMESTEPS, -1)[valid]
    s1_pixels = s1.transpose(2, 3, 0, 1).reshape(height * width, PASTIS_TIMESTEPS, -1)[valid]
    red = s2_pixels[:, :, PASTIS_S2_BANDS.index("B4")]
    nir = s2_pixels[:, :, PASTIS_S2_BANDS.index("B8")]
    ndvi = np.divide(nir - red, nir + red, out=np.zeros_like(red), where=(nir + red) != 0)
    s2_pixels = np.concatenate([s2_pixels, ndvi[:, :, None]], axis=2).astype(np.float32)
    n = int(valid.sum())
    return Benchmark(
        name="pastis",
        task="segmentation",
        s2=s2_pixels,
        s1=s1_pixels.astype(np.float32),
        climate=np.zeros((n, PASTIS_TIMESTEPS, 0), dtype=np.float32),
        s2_mask=np.broadcast_to(s2_mask, (n, PASTIS_TIMESTEPS)).copy(),
        s1_mask=np.broadcast_to(s1_mask, (n, PASTIS_TIMESTEPS)).copy(),
        climate_mask=np.zeros((n, PASTIS_TIMESTEPS), dtype=np.float32),
        doy=np.broadcast_to(_synthetic_month_doy(PASTIS_TIMESTEPS), (n, PASTIS_TIMESTEPS)).copy(),
        labels=labels[valid].astype(np.int64),
        groups=np.full(n, fold, dtype=np.int64),
        latlon=np.zeros((n, 2), dtype=np.float32),
        s2_bands=PASTIS_S2_BANDS + ["NDVI"],
        s1_bands=PASTIS_S1_BANDS,
        climate_bands=[],
        years=np.full(n, 2019, dtype=np.int64),
    )


def load_pastis(
    root: Path = DEFAULT_ROOT,
    max_samples: int | None = None,
    shuffle: bool = True,
    seed: int = 0,
    folds: list[int] | None = None,
) -> PastisBenchmark:
    """Build a lazy PASTIS-R descriptor using the official five folds.

    S2 and ascending-orbit S1 are monthly aggregated only when a 64x64 tile is
    requested. ``max_samples`` limits source patches, not pixels or tiles.
    """
    base = Path(root) / "pastis"
    if not (base / "metadata.geojson").exists():
        raise FileNotFoundError(f"PASTIS metadata not found: {base / 'metadata.geojson'}")
    geo = json.loads((base / "metadata.geojson").read_text())
    rows = [feature["properties"] for feature in geo["features"]]
    if folds:
        rows = [row for row in rows if int(row["Fold"]) in folds]
    order = np.arange(len(rows))
    if shuffle:
        order = np.random.default_rng(seed).permutation(order)
    if max_samples:
        order = order[:max_samples]

    patches: list[PastisPatch] = []
    for i in order:
        r = rows[int(i)]
        pid = int(r["ID_PATCH"])
        s2_path = base / "DATA_S2" / f"S2_{pid}.npy"
        s1_path = base / "DATA_S1A" / f"S1A_{pid}.npy"
        target_path = base / "ANNOTATIONS" / f"TARGET_{pid}.npy"
        if not (s2_path.exists() and s1_path.exists() and target_path.exists()):
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
            )
        )

    if not patches:
        raise ValueError(f"No PASTIS patches parsed from {base}")
    return PastisBenchmark(
        name="pastis",
        task="segmentation",
        patches=tuple(patches),
        tile_size=PASTIS_TILE_SIZE,
        ignore_index=PASTIS_IGNORE,
    )


# --------------------------------------------------------------------------- #
# Dispatch
# --------------------------------------------------------------------------- #

LOADERS = {
    "cropharvest": load_cropharvest,
    "eurocropsml": load_eurocropsml,
    "breizhcrops": load_breizhcrops,
    "pastis": load_pastis,
}


def get_input(name: str, root: Path = DEFAULT_ROOT, **kwargs) -> Benchmark | PastisBenchmark:
    """Load a benchmark by name."""
    if name not in LOADERS:
        raise KeyError(f"Unknown benchmark {name!r}. Known: {sorted(LOADERS)}")
    return LOADERS[name](root=root, **kwargs)
