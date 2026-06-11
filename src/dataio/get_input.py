"""Benchmark input loading for the frozen-embedding robustness pipeline.

Loads raw agricultural EO benchmarks into a single dense in-memory container
(:class:`Benchmark`) that every encoder in ``src/models/*`` can consume: each
encoder applies the stress conditions (sensor-off / temporal-drop) to these
arrays and produces embeddings. Nothing here touches a model or a corruption op.

Expected layout under ``data/input/`` (datasets staged here by you):

    data/input/cropharvest/
        labels.geojson
        features/arrays/<index>_<dataset>.h5      # one (T, C) array per sample
    data/input/eurocropsml/
        preprocess/*.npz                          # one (T, 13) array per parcel
        split/latvia_portugal_vs_estonia/...      # official transnational split
    data/input/sickle/
        sickle_dataset_tabular.csv                # phenology / yield annotations
        images/{S2,S1}/npy/<uid>/*.npz            # per-acquisition band chips
        masks/10m/<uid>.tif                       # plot / phenology / yield rasters
    data/input/yieldsat/
        preprocessed-24-ts/<Country>/merge_s2-soil-dem-weather-coords.nc

CropHarvest is rebuilt from the raw per-sample h5 arrays (band layout below);
EuroCropsML is read from the preprocessed npz parcels. SICKLE is read from the
per-plot acquisition chips, with phenology-date targets from its tabular CSV.
YieldSAT is read from the ML-ready NetCDFs and aggregated from pixels to fields.

Band layouts follow the nasaharvest/cropharvest convention and the EuroCropsML
13-band ordering so embeddings stay comparable across pipelines.
"""

from __future__ import annotations

import csv
import glob
import json
import os
import re
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np
import rasterio
from joblib import Parallel, delayed
from sklearn.preprocessing import LabelEncoder

DEFAULT_ROOT = Path("data/input")

# --- CropHarvest band layout (raw array column indices) ---------------------
# Raw array columns: [S1 VV,VH] + [S2 11 bands] + [ERA5 2] + [SRTM 2] + [NDVI].
CH_S1_IDXS = [0, 1]
CH_S2_IDXS = [2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 17]  # B2..B12 (no B1/B10) + NDVI (col 17)
CH_CLIMATE_IDXS = [13, 14, 15]
CH_MIN_CHANNELS = max(CH_S1_IDXS + CH_S2_IDXS + CH_CLIMATE_IDXS) + 1
CH_S1_BANDS = ["VV", "VH"]
CH_S2_BANDS = ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12", "NDVI"]
CH_CLIMATE_BANDS = ["temperature", "precipitation", "elevation"]
CH_TIMESTEPS = 12

# --- EuroCropsML band layout (raw npz column indices) -----------------------
# 13-band ordering: B1,B2,B3,B4,B5,B6,B7,B8,B8A,B9,B11,B12,SCL. Keep B2..B8A,B11,B12
# and append a computed NDVI. No native S1 / climate.
EC_S2_IDXS = [1, 2, 3, 4, 5, 6, 7, 8, 10, 11]
EC_B4_IDX = 3
EC_B8_IDX = 7
EC_S2_BANDS = ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12", "NDVI"]
EC_COUNTRY_PREFIX = {"EE": "Estonia", "LV": "Latvia", "PT": "Portugal"}

# --- YieldSAT band layout (ML-ready NetCDF band coordinate names) -----------
YS_COUNTRIES = ["Argentina", "Brazil", "Germany", "Uruguay"]
YS_NETCDF_NAME = "merge_s2-soil-dem-weather-coords.nc"
YS_S2_SOURCE_BANDS = ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"]
YS_S2_BANDS = CH_S2_BANDS
YS_CLIMATE_SOURCE_BANDS = ["temp_mean", "total_prec", "dem"]
YS_CLIMATE_BANDS = CH_CLIMATE_BANDS
YS_COORD_BANDS = ["coord_x", "coord_y", "coord_z"]
YS_COUNTRY_CENTROID = {
    "Argentina": (-34.0, -64.0),
    "Brazil": (-14.2, -51.9),
    "Germany": (51.2, 10.4),
    "Uruguay": (-32.5, -55.8),
}


@dataclass
class Benchmark:
    """Dense, in-memory multimodal pixel/parcel time series for one benchmark.

    Modality arrays are ``(N, T, C)`` with the spatial 1x1 dimension squeezed out.
    ``*_mask`` are ``(N, T)`` per-timestep availability (1 = observed). ``doy`` is
    ``(N, T)`` day-of-year. ``labels`` are the task target (binary is_crop,
    encoded class id, or regression value); ``groups`` are the strict-holdout
    group (dataset for CropHarvest, country for EuroCropsML/YieldSAT).
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

    @property
    def n_samples(self) -> int:
        return int(self.s2.shape[0])

    @property
    def timesteps(self) -> int:
        return int(self.s2.shape[1])


# --------------------------------------------------------------------------- #
# Stress corruption
# --------------------------------------------------------------------------- #


def corrupt(
    bench: Benchmark,
    sensor_off: str = "none",
    temporal_drop: float = 0.0,
    seed: int = 0,
) -> Benchmark:
    """Apply a stress condition to a benchmark, returning a corrupted copy.

    This realizes the protocol conditions defined in ``src/evals/evals.py`` but
    takes the primitives directly (``sensor_off`` in {none, s2, s1, climate};
    ``temporal_drop`` in [0, 1)) so this module stays independent of the eval
    layer -- the caller maps a named condition to these args.

    * sensor-off zeros the modality and its availability mask.
    * temporal-drop randomly zeros a fraction of timesteps across *all* modalities
      (timestep 0 is always kept, and at least two timesteps survive), mirroring
      the extraction-time corruption so embeddings stay comparable.
    """
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


def _load_ch_labels(labels_geojson: Path) -> dict[tuple[int, str], tuple[int, float, float]]:
    """Map (index, dataset) -> (is_crop, lat, lon)."""
    geo = json.loads(labels_geojson.read_text())
    out: dict[tuple[int, str], tuple[int, float, float]] = {}
    for f in geo["features"]:
        p = f["properties"]
        key = (int(p["index"]), str(p["dataset"]))
        lat = float(p["lat"]) if p.get("lat") is not None else float("nan")
        lon = float(p["lon"]) if p.get("lon") is not None else float("nan")
        out[key] = (int(p["is_crop"]), lat, lon)
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
    labels, groups, latlons = [], [], []
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
        is_crop, lat, lon = label_map[key]
        s2_list.append(arr[:, CH_S2_IDXS])
        s1_list.append(arr[:, CH_S1_IDXS])
        clim_list.append(arr[:, CH_CLIMATE_IDXS])
        labels.append(is_crop)
        groups.append(dataset)
        latlons.append((lat, lon))

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


def load_eurocropsml(
    root: Path = DEFAULT_ROOT,
    timesteps: int = 96,
    max_samples: int | None = None,
    shuffle: bool = True,
    seed: int = 0,
) -> Benchmark:
    """Load EuroCropsML crop-type parcels (S2 only; S1/climate absent).

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
    label_codes, groups, latlons = [], [], []
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
        values, mask, doy = _resample_to(s2_full, np.asarray(dates), timesteps)
        s2_list.append(values)
        mask_list.append(mask)
        doy_list.append(doy)
        label_codes.append(path.stem.split("_")[-1])
        groups.append(_ec_country(path.stem))
        latlons.append((float(center[1]), float(center[0])) if center is not None else (float("nan"), float("nan")))

    if not s2_list:
        raise ValueError(f"No valid EuroCropsML parcels parsed from {preprocess_dir}")

    n = len(s2_list)
    encoder = LabelEncoder()
    labels = encoder.fit_transform(label_codes).astype(np.int64)
    s2 = np.stack(s2_list).astype(np.float32)
    s2_mask = np.stack(mask_list).astype(np.float32)
    doy = np.stack(doy_list).astype(np.float32)
    empty_s1 = np.zeros((n, timesteps, 2), np.float32)
    empty_clim = np.zeros((n, timesteps, 0), np.float32)
    zeros_mask = np.zeros((n, timesteps), np.float32)
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
    )


# --------------------------------------------------------------------------- #
# SICKLE (paddy phenology: sowing / transplanting / harvesting day-of-season)
# --------------------------------------------------------------------------- #

SICKLE_S2_SRC = ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"]  # B4=red, B8=nir
SICKLE_S2_BANDS = SICKLE_S2_SRC + ["NDVI"]
SICKLE_S1_SRC = ["VV", "VH"]
SICKLE_CENTROID = (10.95, 79.4)  # Cauvery Delta, Tamil Nadu (lat, lon); SICKLE tifs aren't geo-referenced
SICKLE_TARGET_COL = {"sowing": "SOWING_DAY", "transplanting": "TRANSPLANTING_DAY", "harvesting": "HARVESTING_DAY"}
_SICKLE_DATE_RE = re.compile(r"(\d{8})T\d{6}")


def _sickle_date(filename: str) -> datetime:
    match = _SICKLE_DATE_RE.search(filename)
    stamp = match.group(1) if match else filename.split("_")[0][:8]
    return datetime.strptime(stamp, "%Y%m%d")


def _resize_bool(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Nearest-neighbour resize of a boolean mask (SICKLE chips are clipped at image borders)."""
    if mask.shape == shape:
        return mask
    yi = np.linspace(0, mask.shape[0] - 1, shape[0]).round().astype(int)
    xi = np.linspace(0, mask.shape[1] - 1, shape[1]).round().astype(int)
    return mask[np.ix_(yi, xi)]


def _sickle_plot_mean(arr: np.ndarray, sel: np.ndarray) -> float:
    """Mean of one band over plot pixels, resizing the mask to the band's native resolution."""
    band_sel = _resize_bool(sel, arr.shape)
    if not band_sel.any():
        band_sel = np.ones_like(band_sel)
    return float(np.nanmean(arr[band_sel]))


def _sickle_series(uid_dir: Path, src_bands: list[str], sel: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Plot-mean band vectors over time -> (ordinals (K,), values (K, C)), sorted by date."""
    ords, vecs = [], []
    for path in sorted(glob.glob(str(uid_dir / "*.npz"))):
        try:
            data = np.load(path)
        except (OSError, ValueError):
            continue
        if not all(b in data for b in src_bands):
            continue
        # Bands are at native S2 resolution (10m bands 33x33, 20m bands ~17x17),
        # so the plot mask is resized to each band's own shape.
        vecs.append([_sickle_plot_mean(data[b], sel) for b in src_bands])
        ords.append(_sickle_date(os.path.basename(path)).toordinal())
    if not ords:
        return np.empty(0), np.empty((0, len(src_bands)), np.float32)
    order = np.argsort(ords)
    return np.asarray(ords)[order], np.asarray(vecs, np.float32)[order]


def _sickle_resample(ords: np.ndarray, vals: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Nearest-acquisition resample of (K, C) onto the (T,) ordinal grid -> (T, C)."""
    idx = np.abs(ords[None, :] - grid[:, None]).argmin(axis=1)
    return vals[idx]


def _sickle_one_sample(row: dict, base: Path, timesteps: int, col: str):
    """Parse one SICKLE plot -> (s2, s1, doy, s1_mask, target, group), or None to skip."""
    uid = int(float(row["UNIQUE_ID"]))
    try:
        plot_id = int(float(row["PLOT_ID"]))
    except (ValueError, KeyError, TypeError):
        plot_id = -1
    mask_path = base / "masks" / "10m" / f"{uid}.tif"
    s2_dir = base / "images" / "S2" / "npy" / str(uid)
    if not mask_path.exists() or not s2_dir.exists():
        return None
    with rasterio.open(mask_path) as src:
        plot = src.read(1)
    sel = plot == plot_id
    if sel.sum() == 0:
        sel = plot > 0
    if sel.sum() == 0:
        sel = np.ones_like(plot, bool)

    s2_ords, s2_vals = _sickle_series(s2_dir, SICKLE_S2_SRC, sel)
    if len(s2_ords) < 2:
        return None
    grid = np.linspace(s2_ords.min(), s2_ords.max(), timesteps)
    s2g = _sickle_resample(s2_ords, s2_vals, grid)
    b8, b4 = s2g[:, 6], s2g[:, 2]
    denom = b8 + b4
    ndvi = np.where(denom > 0, (b8 - b4) / np.where(denom > 0, denom, 1.0), 0.0)
    s2g = np.concatenate([s2g, ndvi[:, None]], axis=1).astype(np.float32)
    doy = np.asarray([datetime.fromordinal(int(round(g))).timetuple().tm_yday for g in grid], np.float32)

    s1_ords, s1_vals = _sickle_series(base / "images" / "S1" / "npy" / str(uid), SICKLE_S1_SRC, sel)
    if len(s1_ords) >= 1:
        s1g = _sickle_resample(s1_ords, s1_vals, grid).astype(np.float32)
        s1_mask = np.ones(timesteps, np.float32)
    else:
        s1g = np.zeros((timesteps, 2), np.float32)
        s1_mask = np.zeros(timesteps, np.float32)
    group = (row.get("RIVER_PART") or "").strip() or "unknown"
    return s2g, s1g, doy, s1_mask, float(row[col]), group


def load_sickle(
    root: Path = DEFAULT_ROOT,
    target: str = "harvesting",
    timesteps: int = 12,
    max_samples: int | None = None,
    shuffle: bool = True,
    seed: int = 0,
) -> Benchmark:
    """Load SICKLE paddy phenology as a regression benchmark.

    Target (``target`` in {sowing, transplanting, harvesting}) is the day-of-season
    of that event, read from ``sickle_dataset_tabular.csv`` (an external field
    annotation -- not derived from the imagery). Each pixel time series is the
    plot-pixel mean (mask plot == PLOT_ID) of the S2/S1 chips, nearest-resampled
    onto a fixed ``timesteps`` grid spanning that plot's acquisition season.
    SICKLE tifs are not geo-referenced, so a fixed Cauvery-Delta centroid is used
    for ``latlon``; there is no climate modality.
    """
    base = Path(root) / "sickle"
    csv_path = base / "sickle_dataset_tabular.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"SICKLE tabular CSV not found: {csv_path}")
    if target not in SICKLE_TARGET_COL:
        raise ValueError(f"target must be one of {sorted(SICKLE_TARGET_COL)}; got {target!r}")
    col = SICKLE_TARGET_COL[target]

    def _valid(row: dict) -> bool:
        try:
            return int(float(row["PADDY_BIN"])) == 1 and float(row[col]) > 0
        except (ValueError, KeyError):
            return False

    rows = [r for r in csv.DictReader(csv_path.open()) if _valid(r)]
    if shuffle:
        order = np.random.default_rng(seed).permutation(len(rows))
        rows = [rows[i] for i in order]
    if max_samples:
        rows = rows[:max_samples]

    # Each plot reads ~50 npz + a mask; the zip/npz parsing is GIL-bound, so parse
    # plots across processes (loky) -- ~3x on 12 cores, more on the 32-core box.
    # (The assembled bench is then pickle-cached upstream, so this is a cold-path cost.)
    parsed = Parallel(n_jobs=-1)(delayed(_sickle_one_sample)(row, base, timesteps, col) for row in rows)
    parsed = [p for p in parsed if p is not None]
    if not parsed:
        raise ValueError(f"No usable SICKLE samples parsed under {base}")

    s2_list = [p[0] for p in parsed]
    s1_list = [p[1] for p in parsed]
    doy_list = [p[2] for p in parsed]
    s1m_list = [p[3] for p in parsed]
    labels = [p[4] for p in parsed]
    groups = [p[5] for p in parsed]
    s2m_list = [np.ones(timesteps, np.float32) for _ in parsed]
    n = len(s2_list)
    return Benchmark(
        name="sickle",
        task="regression",
        s2=np.stack(s2_list).astype(np.float32),
        s1=np.stack(s1_list).astype(np.float32),
        climate=np.zeros((n, timesteps, 0), np.float32),
        s2_mask=np.stack(s2m_list).astype(np.float32),
        s1_mask=np.stack(s1m_list).astype(np.float32),
        climate_mask=np.zeros((n, timesteps), np.float32),
        doy=np.stack(doy_list).astype(np.float32),
        labels=np.asarray(labels, np.float32),
        groups=np.asarray(groups, dtype=object),
        latlon=np.tile(np.asarray(SICKLE_CENTROID, np.float32), (n, 1)),
        s2_bands=SICKLE_S2_BANDS,
        s1_bands=SICKLE_S1_SRC,
        climate_bands=[],
    )


# --------------------------------------------------------------------------- #
# YieldSAT (field-level crop yield regression from ML-ready NetCDFs)
# --------------------------------------------------------------------------- #

def _decode_attrs(attrs: dict) -> dict[int, str]:
    """Decode integer-code attrs used by YieldSAT categorical variables."""
    out: dict[int, str] = {}
    for key, value in attrs.items():
        try:
            out[int(key)] = str(value)
        except (TypeError, ValueError):
            continue
    return out


def _decode_values(values: np.ndarray, attrs: dict) -> np.ndarray:
    mapping = _decode_attrs(attrs)
    if not mapping:
        return values.astype(str)
    return np.asarray([mapping.get(int(v), str(int(v))) for v in values], dtype=object)


def _ys_doy(times: np.ndarray) -> np.ndarray:
    dates = times.astype("datetime64[ns]").astype("datetime64[D]")
    years = dates.astype("datetime64[Y]").astype("datetime64[D]")
    return (dates - years).astype(np.int64).astype(np.float32) + 1.0


def _ys_latlon_from_unit_xyz(x: float, y: float, z: float, fallback: tuple[float, float]) -> tuple[float, float]:
    vec = np.asarray([x, y, z], dtype=np.float64)
    norm = np.linalg.norm(vec)
    if not np.isfinite(norm) or norm == 0:
        return fallback
    vec = vec / norm
    lat = np.degrees(np.arcsin(np.clip(vec[2], -1.0, 1.0)))
    lon = np.degrees(np.arctan2(vec[1], vec[0]))
    if not np.isfinite(lat) or not np.isfinite(lon):
        return fallback
    return float(lat), float(lon)


def _ys_existing_band_names(ds, names: list[str]) -> list[str]:
    bands = {str(b) for b in ds["band"].values}
    missing = [name for name in names if name not in bands]
    if missing:
        raise ValueError(f"YieldSAT file is missing expected bands: {missing}")
    return names


def _ys_field_records(path: Path, country_name: str, field_limit: set[str] | None):
    import xarray as xr

    ds = xr.open_dataset(path)
    try:
        fields = _decode_values(ds["field_shared_name"].values, ds["field_shared_name"].attrs)
        unique_fields = np.unique(fields)
        if field_limit is not None:
            unique_fields = np.asarray([f for f in unique_fields if str(f) in field_limit], dtype=object)
        if len(unique_fields) == 0:
            return []

        s2_names = _ys_existing_band_names(ds, YS_S2_SOURCE_BANDS)
        climate_names = _ys_existing_band_names(ds, YS_CLIMATE_SOURCE_BANDS)
        coord_names = [b for b in YS_COORD_BANDS if b in {str(v) for v in ds["band"].values}]
        fallback_latlon = YS_COUNTRY_CENTROID.get(country_name, (0.0, 0.0))
        records = []
        for field_name in unique_fields:
            idx = np.flatnonzero(fields == field_name)
            if len(idx) == 0:
                continue
            s2_raw = ds["sample"].isel(index=idx).sel(band=s2_names).values.astype(np.float32)
            target = ds["target"].isel(index=idx).values.astype(np.float32)
            if s2_raw.ndim != 3 or len(target) == 0 or not np.isfinite(target).any():
                continue
            s2_mean = np.nanmean(s2_raw, axis=0)
            b4, b8 = s2_mean[:, s2_names.index("B04")], s2_mean[:, s2_names.index("B08")]
            ndvi = np.divide(b8 - b4, b8 + b4, out=np.zeros_like(b4), where=(b8 + b4) > 0)
            s2 = np.concatenate([s2_mean, ndvi[:, None]], axis=1).astype(np.float32)

            climate_raw = ds["sample"].isel(index=idx).sel(band=climate_names).values.astype(np.float32)
            climate = np.nanmean(climate_raw, axis=0).astype(np.float32)

            times = ds["times"].isel(index=int(idx[0])).values
            doy = _ys_doy(np.asarray(times))
            if len(doy) != s2.shape[0]:
                doy = np.linspace(1, 365, s2.shape[0]).astype(np.float32)

            latlon = fallback_latlon
            if len(coord_names) == 3:
                coords = ds["sample"].isel(index=idx).sel(band=coord_names).values.astype(np.float32)
                xyz = np.nanmean(coords, axis=(0, 1))
                latlon = _ys_latlon_from_unit_xyz(float(xyz[0]), float(xyz[1]), float(xyz[2]), fallback_latlon)

            records.append((s2, climate, doy, float(np.nanmean(target)), country_name, latlon, str(field_name)))
        return records
    finally:
        ds.close()


def load_yieldsat(
    root: Path = DEFAULT_ROOT,
    timesteps: int = 24,
    max_samples: int | None = None,
    shuffle: bool = True,
    seed: int = 0,
) -> Benchmark:
    """Load YieldSAT as field-level yield regression.

    YieldSAT's ML-ready release is pixel-level: one row per 10 m pixel, with
    ``sample(index,time_step,band)`` and ``target(index)`` yield. This loader
    groups all pixels belonging to the same ``field_shared_name`` and averages
    both inputs and yield targets, producing one field-level sample per field.
    That keeps this frozen-embedding pipeline tractable while preserving an
    externally measured yield regression target.
    """
    base = Path(root) / "yieldsat"
    preprocessed = base / "preprocessed-24-ts"
    if not preprocessed.exists():
        raise FileNotFoundError(f"YieldSAT preprocessed directory not found: {preprocessed}")

    country_paths = [(country, preprocessed / country / YS_NETCDF_NAME) for country in YS_COUNTRIES]
    missing = [str(path) for _, path in country_paths if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing YieldSAT NetCDF files:\n  " + "\n  ".join(missing))

    field_names_by_country: dict[str, list[str]] = {}
    for country, path in country_paths:
        import xarray as xr

        ds = xr.open_dataset(path)
        try:
            fields = _decode_values(ds["field_shared_name"].values, ds["field_shared_name"].attrs)
            field_names_by_country[country] = sorted(np.unique(fields).astype(str).tolist())
        finally:
            ds.close()

    selected = [(country, field) for country in YS_COUNTRIES for field in field_names_by_country[country]]
    if shuffle:
        order = np.random.default_rng(seed).permutation(len(selected))
        selected = [selected[i] for i in order]
    if max_samples:
        selected = selected[:max_samples]
    wanted: dict[str, set[str]] = {}
    for country, field in selected:
        wanted.setdefault(country, set()).add(field)

    parsed = []
    for country, path in country_paths:
        parsed.extend(_ys_field_records(path, country, wanted.get(country, set())))
    if not parsed:
        raise ValueError(f"No usable YieldSAT fields parsed under {preprocessed}")

    if timesteps != parsed[0][0].shape[0]:
        raise ValueError(f"YieldSAT preprocessed files are 24-step; got timesteps={timesteps}")

    s2_list = [p[0] for p in parsed]
    clim_list = [p[1] for p in parsed]
    doy_list = [p[2] for p in parsed]
    labels = [p[3] for p in parsed]
    groups = [p[4] for p in parsed]
    latlons = [p[5] for p in parsed]
    n = len(parsed)
    ones = np.ones((n, timesteps), dtype=np.float32)
    return Benchmark(
        name="yieldsat",
        task="regression",
        s2=np.stack(s2_list).astype(np.float32),
        s1=np.zeros((n, timesteps, 2), np.float32),
        climate=np.stack(clim_list).astype(np.float32),
        s2_mask=ones.copy(),
        s1_mask=np.zeros((n, timesteps), np.float32),
        climate_mask=ones.copy(),
        doy=np.stack(doy_list).astype(np.float32),
        labels=np.asarray(labels, np.float32),
        groups=np.asarray(groups, dtype=object),
        latlon=np.asarray(latlons, dtype=np.float32),
        s2_bands=YS_S2_BANDS,
        s1_bands=CH_S1_BANDS,
        climate_bands=YS_CLIMATE_BANDS,
    )


# --------------------------------------------------------------------------- #
# Dispatch
# --------------------------------------------------------------------------- #

LOADERS = {
    "cropharvest": load_cropharvest,
    "eurocropsml": load_eurocropsml,
    "sickle": load_sickle,
    "yieldsat": load_yieldsat,
}


def get_input(name: str, root: Path = DEFAULT_ROOT, **kwargs) -> Benchmark:
    """Load a benchmark by name."""
    if name not in LOADERS:
        raise KeyError(f"Unknown benchmark {name!r}. Known: {sorted(LOADERS)}")
    return LOADERS[name](root=root, **kwargs)
