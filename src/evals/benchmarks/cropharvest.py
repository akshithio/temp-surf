"""Benchmark: binary crop / non-crop classification (CropHarvest).

The primary benchmark: real per-sample ``is_crop`` labels, evaluated under
strict geographic holdout (one source region held out at a time).

CropHarvest is distributed as 12 monthly steps per sample (a Feb-start agricultural year), so its
"native" cadence is already monthly. The loader carries every native band (S1 VV/VH, all S2
spectral incl B9, ERA5, SRTM, NDVI) with each step's true calendar month -- no band pre-selection.
Each model selects what it needs via the Benchmark view accessors.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timedelta
from pathlib import Path

import h5py
import numpy as np

from dataio.get_input import (
    Benchmark,
    ModalitySeries,
    NativeSeries,
    _select_files,
    _synthetic_month_doy,
)

BENCHMARK = "cropharvest"
LABEL_KIND = "binary"
HOLDOUTS = ["togo", "ethiopia", "lem-brazil", "rwanda", "togo-eval"]
# Binary task with global coordinates and multi-year windows, so it supports the
# committed set: region, climate (Köppen via lat/lon), and forward-time. (Phenology
# was dropped: NDVI-phenology domains are confounded with the crop/non-crop label.)
SPLIT_REGIMES = ["random_id", "geographic_ood", "climate_ood", "temporal_ood"]

# --- Raw array band layout --------------------------------------------------
# Raw 18-col array: [S1 VV,VH] + [S2: B2..B8A, B9, B11, B12] + [ERA5 temp,precip]
#   + [SRTM elevation(15), slope(16)] + [NDVI(17)].
CH_S1_IDXS = [0, 1]
CH_S2_IDXS_ALL = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 17]  # all S2 spectral (incl B9 at col 10) + NDVI (col 17)
CH_CLIMATE_IDXS = [13, 14, 15, 16]  # temperature, precipitation, elevation, slope
CH_MIN_CHANNELS = max(CH_S1_IDXS + CH_S2_IDXS_ALL + CH_CLIMATE_IDXS) + 1
CH_S1_BANDS = ["VV", "VH"]
CH_S2_BANDS = ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B9", "B11", "B12", "NDVI"]
CH_CLIMATE_BANDS = ["temperature", "precipitation", "elevation", "slope"]
CH_TIMESTEPS = 12
# CropHarvest's 12 steps are a Feb-start agricultural year (prev-Feb .. Jan), so step k's true
# calendar month is CH_MONTHS[k]. The monthly view re-bins these to a Jan-Dec grid.
CH_MONTHS = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 0], dtype=np.int64)


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


def load_benchmark(
    root: Path = Path("data/input/benchmarks"),
    max_samples: int | None = None,
    shuffle: bool = True,
    seed: int = 0,
) -> Benchmark:
    """Rebuild CropHarvest crop/non-crop from raw per-sample h5 arrays (native monthly cadence)."""
    base = root / "cropharvest"
    arrays_dir = base / "features" / "arrays"
    labels_geojson = base / "labels.geojson"
    if not arrays_dir.exists():
        raise FileNotFoundError(f"CropHarvest arrays not found: {arrays_dir}")
    if not labels_geojson.exists():
        raise FileNotFoundError(f"CropHarvest labels not found: {labels_geojson}")

    label_map = _load_ch_labels(labels_geojson)
    files = _select_files([p for p in arrays_dir.glob("*.h5") if p.is_file()], shuffle, seed, max_samples)

    s2_series, s1_series, clim_series = [], [], []
    labels, groups, latlons, years = [], [], [], []
    skipped = 0
    for path in files:
        idx_str, dataset = path.stem.split("_", 1)
        key = (int(idx_str), dataset)
        if key not in label_map:
            continue  # no label for this array (expected for the unlabeled pool); not a corruption
        try:
            with h5py.File(path, "r") as f:
                if "array" not in f:
                    skipped += 1
                    continue
                arr = np.asarray(f["array"], dtype=np.float32)
        except OSError:
            skipped += 1  # unreadable / corrupt HDF5
            continue
        if arr.ndim != 2 or arr.shape[0] != CH_TIMESTEPS or arr.shape[1] < CH_MIN_CHANNELS:
            skipped += 1  # malformed shape
            continue
        arr = np.nan_to_num(arr, nan=0.0)
        is_crop, lat, lon, year = label_map[key]
        s2_series.append(arr[:, CH_S2_IDXS_ALL].astype(np.float32))
        s1_series.append(arr[:, CH_S1_IDXS].astype(np.float32))
        clim_series.append(arr[:, CH_CLIMATE_IDXS].astype(np.float32))
        labels.append(is_crop)
        groups.append(dataset)
        latlons.append((lat, lon))
        years.append(year)

    if skipped:
        # Corrupt/malformed arrays must not silently shrink the dataset.
        msg = f"CropHarvest: {skipped} labeled arrays were unreadable/malformed in {arrays_dir}"
        if os.environ.get("OVERWRITE_MODE", "").strip().lower() not in ("", "0", "false", "no"):
            raise ValueError(msg + " (OVERWRITE_MODE is set)")
        print(f"   !! {msg} -- those samples are skipped (set OVERWRITE_MODE=1 to fail instead)", flush=True)
    if not s2_series:
        raise ValueError(f"No valid CropHarvest arrays parsed from {arrays_dir}")

    n = len(s2_series)
    # Shared per-step calendar months / day-of-year (identical for every sample); per-sample years.
    doy_vals = _synthetic_month_doy(12)[CH_MONTHS].astype(np.float32)
    months_l = [CH_MONTHS] * n
    doy_l = [doy_vals] * n
    years_l = [np.full(CH_TIMESTEPS, y, dtype=np.int64) for y in years]
    native = NativeSeries(
        s2=ModalitySeries(s2_series, months_l, doy_l, years_l, CH_S2_BANDS),
        s1=ModalitySeries(s1_series, months_l, doy_l, years_l, CH_S1_BANDS),
        climate=ModalitySeries(clim_series, months_l, doy_l, years_l, CH_CLIMATE_BANDS),
    )
    return Benchmark(
        name="cropharvest",
        label_kind="binary",
        native=native,
        labels=np.asarray(labels, dtype=np.int64),
        groups=np.asarray(groups, dtype=object),
        latlon=np.asarray(latlons, dtype=np.float32),
        years=np.asarray(years, dtype=np.int64),
    )


def make_targets(bench) -> tuple[np.ndarray, np.ndarray]:
    """y = is_crop (real label); groups = source dataset (geographic holdout)."""
    return bench.labels.astype(np.int64), bench.groups
