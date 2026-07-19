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
from evals import split_spec

BENCHMARK = "cropharvest"
LABEL_KIND = "binary"
HOLDOUTS = ["kenya", "togo", "ethiopia", "lem-brazil", "rwanda"]
#: The official CropHarvest split is Togo ONLY: the released is_crop task does not reproduce the
#: Kenya maize / Brazil coffee release evaluations, so those are NOT official holdouts.
OFFICIAL_HOLDOUTS = ["togo"]
#: Official Togo split provenance (the un-merged source-collection names the canonical region
#: collapses): the ``togo`` source pool (1,272) subdivided per seed vs the fixed ``togo-eval`` target
#: evaluation (306). Consumed by the official regime; distinct from the canonical geographic region.
OFFICIAL_PROVENANCE = {"source": "togo", "target": "togo-eval"}
#: Basis for the retired ``spatial_blocks`` strategy only -- see spatial_block_domains().
GEOGRAPHIC_BLOCK_DEGREES = 2.0
GEOGRAPHIC_SPLIT = {
    # Leave-one-domain-out over the FULL canonical domain census (17 canonical regions after the
    # Mali merge), not a curated subset: `HOLDOUTS` above covers only ~15% of samples and omits the
    # five largest domains entirely, so it cannot support a claim about geographic robustness.
    "strategy": "leave_one_domain_out",
    "label": "lodo_canonical_purge50km",
    "purge_km": 50.0,
    # Validity is declared HERE, before any model runs (see geographic_ood.domain_census):
    #   target -- any domain with >= min_target_n samples. One-class domains are REAL deployment
    #     regions (central-asia, tanzania, uganda, zimbabwe) and are kept;
    #     they are excluded only from metrics that mathematically require both classes, which
    #     score_binary already handles by returning auc=nan while accuracy / balanced_accuracy /
    #     calibration and the test_pos_rate class-conditional diagnostic stay well defined.
    #   validation -- must carry BOTH classes, because the binary probe's decision threshold is
    #     calibrated on it; a one-class validation region cannot select a threshold.
    "min_target_n": 10,
    "allow_one_class_target": True,
}
# spatial_cluster_ood: coordinate-only spherical-K-means cells (5 cells, purge_km from split_spec);
# no benchmark-specific override -- see evals.regimes.spatial_cluster_ood / evals.split_spec.
SPLIT_REGIMES = ["random_id", "official", "geographic_ood", "spatial_cluster_ood"]

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


#: THE canonical-region mapper (provenance -> region). The merge rules live once in
#: ``split_spec.CROPHARVEST_REGION_MERGES``; this module holds no rules of its own. ``bench.groups``
#: is built from this.
canonical_region = split_spec.cropharvest_canonical_region
_ch_geo_group = canonical_region  # internal alias (loader + geographic_domains)


def provenance_dataset(sample_id: str) -> str:
    """Original source-collection provenance (22 datasets) recovered from a stable sample id.

    Stable ids are ``<index>_<dataset>.h5``; the official Togo split needs the un-merged provenance
    (Togo source vs Togo-eval), which the canonical region collapses. Recovering it from the id
    keeps a single source of truth and needs no extra benchmark field.
    """
    stem = str(sample_id)
    if stem.endswith(".h5"):
        stem = stem[:-3]
    return stem.split("_", 1)[1] if "_" in stem else stem


def provenance_groups(bench) -> np.ndarray:
    """Per-sample provenance dataset (22) -- the basis for the official Togo split, not geography."""
    return np.asarray([provenance_dataset(s) for s in bench.sample_ids], dtype=object)


def spatial_block_domains(bench) -> np.ndarray:
    """The PRE-LODO domain basis: 2-degree lat/lon blocks.

    Retained solely so the frozen ``spatial_block_2deg_purge50km`` result artifacts stay
    reproducible: 20 result files across five runs were produced on this basis, four of them in the
    canonical ``output-erm-full-20260711`` tree the paper currently cites. It is NOT consumed by any
    current split regime -- schema-v2 ``geographic_ood`` is true region/tile LODO and
    ``spatial_cluster_ood`` is coordinate-only spherical-K-means cells -- and only the no-coordinate
    edge case is pinned (test_tasks.test_cropharvest_retains_the_block_basis_for_historical_artifacts).
    """
    latlon = np.asarray(bench.latlon, dtype=float)
    out = np.full(len(latlon), "unknown", dtype=object)
    if latlon.ndim != 2 or latlon.shape[1] != 2:
        return out
    valid = np.isfinite(latlon).all(axis=1)
    block = GEOGRAPHIC_BLOCK_DEGREES
    lat_bin = np.floor((latlon[valid, 0] + 90.0) / block).astype(int)
    lon_bin = np.floor((latlon[valid, 1] + 180.0) / block).astype(int)
    out[valid] = [f"block_{lat}_{lon}" for lat, lon in zip(lat_bin, lon_bin, strict=True)]
    return out


def geographic_domains(bench) -> np.ndarray:
    """Canonical CropHarvest domains: the source collection each sample was gathered in.

    This is the universe ``geographic_ood`` leaves out one at a time. It is a PROVENANCE label,
    not a polygon, and two of the 18 domains are globally distributed collections rather than
    regions: ``geowiki-landcover-2017`` spans ~35,000 km (24,761 samples, 37% of the benchmark)
    and ``croplands`` ~30,000 km. Their points are interleaved with every other domain, so on the
    other folds they contribute samples that sit inside the held-out target region. The 50 km
    purge in GEOGRAPHIC_SPLIT is what removes those, and is therefore load-bearing for the
    holdout's meaning rather than cosmetic -- it bounds, but does not eliminate, that leakage.
    """
    return np.asarray(bench.groups, dtype=object)


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
    excluded = set(split_spec.CROPHARVEST_EXCLUDED_FILES)
    all_h5 = [p for p in arrays_dir.glob("*.h5") if p.is_file()]
    # A frozen, verified-malformed exclusion is NOT an unexpected corruption: record it in
    # data_quality separately and never let it trip the STRICT_MODE skipped-inputs failure below.
    frozen_excluded = sorted(p.name for p in all_h5 if p.name in excluded)
    files = _select_files([p for p in all_h5 if p.name not in excluded], shuffle, seed, None)

    s2_series, s1_series, clim_series = [], [], []
    labels, groups, latlons, years, sample_ids = [], [], [], [], []
    skipped_records: list[dict[str, str]] = []
    valid_count = 0
    for path in files:
        if max_samples is not None and valid_count >= max_samples:
            break
        idx_str, dataset = path.stem.split("_", 1)
        key = (int(idx_str), dataset)
        if key not in label_map:
            continue  # no label for this array (expected for the unlabeled pool); not a corruption
        try:
            with h5py.File(path, "r") as f:
                if "array" not in f:
                    skipped_records.append({"path": str(path), "reason": "missing array dataset"})
                    continue
                arr = np.asarray(f["array"], dtype=np.float32)
        except OSError:
            skipped_records.append({"path": str(path), "reason": "unreadable hdf5"})
            continue
        if arr.ndim != 2 or arr.shape[0] != CH_TIMESTEPS or arr.shape[1] < CH_MIN_CHANNELS:
            skipped_records.append({"path": str(path), "reason": f"malformed shape {arr.shape}"})
            continue
        arr = np.nan_to_num(arr, nan=0.0)
        is_crop, lat, lon, year = label_map[key]
        s2_series.append(arr[:, CH_S2_IDXS_ALL].astype(np.float32))
        s1_series.append(arr[:, CH_S1_IDXS].astype(np.float32))
        clim_series.append(arr[:, CH_CLIMATE_IDXS].astype(np.float32))
        labels.append(is_crop)
        groups.append(_ch_geo_group(dataset))
        latlons.append((lat, lon))
        years.append(year)
        sample_ids.append(path.name)
        valid_count += 1

    if skipped_records:
        msg = f"CropHarvest: {len(skipped_records)} labeled arrays were unreadable/malformed in {arrays_dir}"
        if os.environ.get("STRICT_MODE", "").strip().lower() not in ("", "0", "false", "no"):
            raise ValueError(msg + " (STRICT_MODE is set)")
        print(f"   !! {msg} -- those samples are skipped (set STRICT_MODE=True to fail instead)", flush=True)
    if not s2_series:
        raise ValueError(f"No valid CropHarvest arrays parsed from {arrays_dir}")

    data_quality: dict = {}
    if skipped_records:
        data_quality["skipped_inputs"] = skipped_records
    if frozen_excluded:
        data_quality["frozen_exclusions"] = [
            {"file": name, "reason": "verified malformed; frozen exclusion applied before all regimes"}
            for name in frozen_excluded
        ]

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
        sample_ids=np.asarray(sample_ids, dtype=object),
        data_quality=data_quality,
        monthly_order=CH_MONTHS,
    )


def make_targets(bench) -> tuple[np.ndarray, np.ndarray]:
    """y = is_crop (real label); groups = source dataset (geographic holdout)."""
    return bench.labels.astype(np.int64), bench.groups
