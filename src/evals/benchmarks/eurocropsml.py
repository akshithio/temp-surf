"""Benchmark: multi-class crop-type classification (EuroCropsML).

Real per-parcel crop-type labels. EuroCropsML labels are 10-digit HCAT codes
(Hierarchical Crop and Agriculture Taxonomy) -- the full code identifies a
specific crop/variety, which yields ~100+ long-tail classes and a near-floor
macro-F1. HCAT is hierarchical, so truncating the code to its leading digits is a
principled coarsening to a broader crop-type level. We truncate to
``HCAT_PREFIX`` digits (6 -> ~36 crop-type classes, vs ~100 at full granularity)
and re-encode to contiguous ids.

The official regime reads the release split files for Latvia->Estonia and
Latvia+Portugal->Estonia. Geographic OOD uses country/domain holdouts.

The loader emits the NATIVE per-parcel acquisition series (every real S2 observation, true
dates, ALL 13 native bands + NDVI). It does NOT pre-aggregate or drop bands: each model does its
own temporal handling and band selection via the Benchmark view accessors. (The old loader
composited to 12 months and kept only 10 bands, which silently denied B1/B9/B10 to models that
want them, e.g. OlmoEarth.)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from sklearn.preprocessing import LabelEncoder

from dataio.get_input import (
    Benchmark,
    ModalitySeries,
    NativeSeries,
    _select_files,
)

BENCHMARK = "eurocropsml"
LABEL_KIND = "multiclass"
HOLDOUTS = ["Estonia"]
OFFICIAL_HOLDOUTS = ["latvia_vs_estonia", "latvia_portugal_vs_estonia"]
GEOGRAPHIC_HOLDOUTS = ["Estonia", "Latvia", "Portugal"]
GEOGRAPHIC_PURGE_KM = 25.0
SPATIAL_CLUSTER_SPLIT = {
    "label": "spatial_cluster_purge25km",
    "n_clusters": 9,
    "val_fraction": 0.10,
    "test_fraction": 0.20,
    "purge_km": 25.0,
}
SPLIT_REGIMES = ["random_id", "official", "geographic_ood", "spatial_cluster_ood"]
HCAT_PREFIX = 6  # truncate the 10-digit HCAT code to this many leading digits (crop-type level)

# --- Raw npz band layout ----------------------------------------------------
# The 13 columns are the native Sentinel-2 bands in standard order (B1..B12, with B8A between B8
# and B9). We keep ALL of them + a computed NDVI; each model selects what it needs by name.
EC_S2_BANDS_ALL = ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B9", "B10", "B11", "B12", "NDVI"]
EC_B4_IDX = 3  # B4 (red), native order
EC_B8_IDX = 7  # B8 (NIR), native order
EC_COUNTRY_PREFIX = {"EE": "Estonia", "LV": "Latvia", "PT": "Portugal"}


def _ec_country(stem: str) -> str:
    return EC_COUNTRY_PREFIX.get(stem[:2], stem[:2])


def _split_indices(split_ids: list[str], id_to_idx: dict[str, int]) -> np.ndarray:
    return np.asarray([id_to_idx[x] for x in split_ids if x in id_to_idx], dtype=np.int64)


def _official_splits(base: Path, sample_ids: list[str]) -> dict[str, dict[str, np.ndarray]]:
    import json

    id_to_idx = {sid: i for i, sid in enumerate(sample_ids)}
    out: dict[str, dict[str, np.ndarray]] = {}
    for name in OFFICIAL_HOLDOUTS:
        split_dir = base / "split" / name
        pretrain = split_dir / "pretrain" / "region_split.json"
        finetune = split_dir / "finetune" / "region_split_all.json"
        if not (pretrain.exists() and finetune.exists()):
            continue
        pre = json.loads(pretrain.read_text())
        fin = json.loads(finetune.read_text())
        out[name] = {
            "train": _split_indices(pre.get("train", []), id_to_idx),
            "val": _split_indices(pre.get("val", []), id_to_idx),
            "test": _split_indices(fin.get("test", []), id_to_idx),
            "target_train": _split_indices(fin.get("train", []) + fin.get("val", []), id_to_idx),
        }
    return out


def load_benchmark(
    root: Path = Path("data/input/benchmarks"),
    max_samples: int | None = None,
    shuffle: bool = True,
    seed: int = 0,
) -> Benchmark:
    """Load EuroCropsML crop-type parcels as native per-parcel S2 series (S1/climate absent).

    Each parcel's irregular Sentinel-2 series is carried at its native cadence with true acquisition
    dates and all 13 bands + NDVI -- no temporal aggregation, no band pre-selection. Models do their
    own temporal handling (Presto composites to monthly, TESSERA consumes the full series).

    Shuffling matters here: npz filenames sort by country prefix (EE/LV/PT), so a sorted
    max_samples subset would be all-Estonia. Shuffling spans countries, which the transnational
    (Latvia+Portugal -> Estonia) split requires.
    """
    base = root / "eurocropsml"
    preprocess_dir = base / "preprocess"
    if not preprocess_dir.exists():
        raise FileNotFoundError(f"EuroCropsML preprocess dir not found: {preprocess_dir}")
    files = _select_files([p for p in preprocess_dir.glob("*.npz") if p.is_file()], shuffle, seed, max_samples)
    if not files:
        raise ValueError(f"No EuroCropsML npz files in {preprocess_dir}")

    s2_series, s2_months, s2_doy, s2_years = [], [], [], []
    label_codes, groups, latlons, years, sample_ids = [], [], [], [], []
    for path in files:
        with np.load(str(path)) as data:
            raw = data["data"].astype(np.float32)
            dates = data["dates"]
            center = data["center"] if "center" in data else None  # [lon, lat]
        if raw.ndim != 2 or raw.shape[1] < 13 or len(dates) != raw.shape[0]:
            continue
        b4, b8 = raw[:, EC_B4_IDX], raw[:, EC_B8_IDX]
        ndvi = np.divide(b8 - b4, b8 + b4, out=np.zeros_like(b4), where=(b8 + b4) > 0)
        s2_full = np.concatenate([raw[:, :13], ndvi[:, None]], axis=1).astype(np.float32)  # all 13 bands + NDVI
        dates_arr = np.asarray(dates)
        d_day = dates_arr.astype("datetime64[D]")
        s2_series.append(s2_full)
        s2_months.append((dates_arr.astype("datetime64[M]").astype(np.int64) % 12).astype(np.int64))
        s2_doy.append(((d_day - dates_arr.astype("datetime64[Y]").astype("datetime64[D]")).astype(np.int64) + 1).astype(np.float32))
        s2_years.append((dates_arr.astype("datetime64[Y]").astype(np.int64) + 1970).astype(np.int64))
        label_codes.append(path.stem.split("_")[-1])
        groups.append(_ec_country(path.stem))
        latlons.append((float(center[1]), float(center[0])) if center is not None else (float("nan"), float("nan")))
        years.append(int(str(dates_arr.ravel()[0])[:4]) if dates_arr.size else 0)
        sample_ids.append(path.name)

    if not s2_series:
        raise ValueError(f"No valid EuroCropsML parcels parsed from {preprocess_dir}")

    n = len(s2_series)
    encoder = LabelEncoder()
    labels = encoder.fit_transform(label_codes).astype(np.int64)
    native = NativeSeries(
        s2=ModalitySeries(values=s2_series, months=s2_months, doy=s2_doy, years=s2_years, bands=EC_S2_BANDS_ALL),
        s1=ModalitySeries.absent(n),
        climate=ModalitySeries.absent(n),
    )
    return Benchmark(
        name="eurocropsml",
        label_kind="multiclass",
        native=native,
        labels=labels,
        groups=np.asarray(groups, dtype=object),
        latlon=np.asarray(latlons, dtype=np.float32),
        label_names=list(encoder.classes_),
        years=np.asarray(years, dtype=np.int64),
        sample_ids=np.asarray(sample_ids, dtype=object),
        official_splits=_official_splits(base, sample_ids),
    )


def make_targets(bench) -> tuple[np.ndarray, np.ndarray]:
    """y = HCAT-coarsened crop-type id; groups = country (transnational holdout)."""
    if bench.label_names is None:
        raise ValueError("eurocropsml needs bench.label_names (HCAT codes) for coarsening.")
    codes = np.asarray([str(bench.label_names[i]) for i in bench.labels])
    coarse = np.array([c[:HCAT_PREFIX] for c in codes])
    _, y = np.unique(coarse, return_inverse=True)  # contiguous class ids
    return y.astype(np.int64), bench.groups
