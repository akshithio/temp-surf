"""Benchmark: multi-class crop-type classification (EuroCropsML).

Real per-parcel crop-type labels. EuroCropsML labels are 10-digit HCAT codes
(Hierarchical Crop and Agriculture Taxonomy) -- the full code identifies a
specific crop/variety, which yields ~100+ long-tail classes and a near-floor
macro-F1. HCAT is hierarchical, so truncating the code to its leading digits is a
principled coarsening to a broader crop-type level. We truncate to
``HCAT_PREFIX`` digits (6 -> ~36 crop-type classes, vs ~100 at full granularity)
and re-encode to contiguous ids.

Evaluated as transnational transfer: train on Latvia + Portugal, test on Estonia
(``groups`` is the country, so holding out "Estonia" reproduces that split).

LABEL_KIND = "multiclass" -> probes use run_probes_multiclass + multiclass
metrics (macro/weighted F1, balanced acc, accuracy, macro AUC). Unseen-class drops
at low label budgets are expected and are part of the transfer story.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from sklearn.preprocessing import LabelEncoder

from dataio.get_input import (
    Benchmark,
    _select_files,
    _synthetic_month_doy,
)

BENCHMARK = "eurocropsml"
LABEL_KIND = "multiclass"
HOLDOUTS = ["Estonia"]  # train Latvia+Portugal -> test Estonia (official transnational split)
HCAT_PREFIX = 6  # truncate the 10-digit HCAT code to this many leading digits (crop-type level)

# --- Raw npz band layout ----------------------------------------------------
# 13 columns are the native Sentinel-2 bands B1..B12
EC_S2_IDXS = [1, 2, 3, 4, 5, 6, 7, 8, 11, 12]
EC_B4_IDX = 3  # B4 (red), native order
EC_B8_IDX = 7  # B8 (NIR), native order
EC_S2_BANDS = ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12", "NDVI"]
EC_COUNTRY_PREFIX = {"EE": "Estonia", "LV": "Latvia", "PT": "Portugal"}
EC_N_MONTHS = 12


def _ec_country(stem: str) -> str:
    return EC_COUNTRY_PREFIX.get(stem[:2], stem[:2])


def _resample_to(arr: np.ndarray, dates_ns: np.ndarray, timesteps: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Cap/pad a variable-length parcel series to a fixed length.

    Returns (values (T, C), mask (T,), doy (T,)). Longer series are span-preserving
    linspace-downsampled; shorter ones are zero-padded with mask=0.
    """
    n_t, c = arr.shape
    d_start = dates_ns.astype("datetime64[ns]").astype("datetime64[D]")
    d_end = dates_ns.astype("datetime64[ns]").astype("datetime64[Y]").astype("datetime64[D]")
    doy_all = (d_start - d_end).astype(np.int64) + 1
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


def _monthly_composite(arr: np.ndarray, dates: np.ndarray, n_months: int = EC_N_MONTHS):
    """Bin a variable-length parcel series into calendar-month composites (mean per month).

    Returns (values (n_months, C), mask (n_months,), doy (n_months,)). This is the harmonized
    temporal protocol: it matches the **monthly cadence Presto/CropHarvest assume** -- Presto's
    positional encoding treats each timestep as one consecutive month, so a sub-monthly grid
    (e.g. 96 acquisitions in a year) would be mis-read as 96 *months*. Month-mean uses ALL
    observations (no lossy linspace subsample); empty months are masked; doy is mid-month.
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


def load_benchmark(
    root: Path = Path("data/input/benchmarks"),
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
    base = root / "eurocropsml"
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
    model = LabelEncoder()
    labels = model.fit_transform(label_codes).astype(np.int64)
    s2 = np.stack(s2_list).astype(np.float32)
    s2_mask = np.stack(mask_list).astype(np.float32)
    doy = np.stack(doy_list).astype(np.float32)
    empty_s1 = np.zeros((n, EC_N_MONTHS, 2), np.float32)
    empty_clim = np.zeros((n, EC_N_MONTHS, 0), np.float32)
    zeros_mask = np.zeros((n, EC_N_MONTHS), np.float32)
    return Benchmark(
        name="eurocropsml",
        label_kind="multiclass",
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
        label_names=list(model.classes_),
        years=np.asarray(years, dtype=np.int64),
    )


def make_targets(bench) -> tuple[np.ndarray, np.ndarray]:
    """y = HCAT-coarsened crop-type id; groups = country (transnational holdout)."""
    if bench.label_names is None:
        raise ValueError("eurocropsml needs bench.label_names (HCAT codes) for coarsening.")
    codes = np.asarray([str(bench.label_names[i]) for i in bench.labels])
    coarse = np.array([c[:HCAT_PREFIX] for c in codes])
    _, y = np.unique(coarse, return_inverse=True)  # contiguous class ids
    return y.astype(np.int64), bench.groups
