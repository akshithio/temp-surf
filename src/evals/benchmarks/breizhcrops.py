"""Benchmark: BreizhCrops crop-type classification (Brittany, Sentinel-2 L1C, 9 classes).

Multiclass crop-type classification over NUTS-3 regions of Brittany, France. Geographic
holdout reproduces the BreizhCrops paper's published fold: ``frh04`` is the test region,
``frh03`` (``VAL_HOLDOUT``) is the validation region, and the remaining regions train.
``groups`` is the region.

Parcel coordinates are recovered from BreizhCrops' own parcel shapefile (see
``_bz_parcel_latlon``) so location-aware encoders receive a real lat/lon rather than a
NaN placeholder; the dataset is still S2-only and single-year, so coordinates do not
unlock a climate or temporal regime (all of Brittany is one Köppen zone).

LABEL_KIND = "multiclass" -> probes use run_probes_multiclass + multiclass metrics.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from dataio.get_input import (
    Benchmark,
    _synthetic_month_doy,
)

BENCHMARK = "breizhcrops"
LABEL_KIND = "multiclass"
HOLDOUTS = ["frh04"]  # paper's TEST region
# Paper fold: frh01+frh02 train, frh03 validation, frh04 test. VAL_HOLDOUT makes frh03 the
# validation region (not a random carve), so geographic_ood reproduces the published split
# instead of scattering frh03/belle-ile into training.
VAL_HOLDOUT = "frh03"
# Region holdout is the only OOD axis here: crop-type task (phenology is label-confounded),
# a single year 2017 (no forward-time split), and although parcels now carry real coordinates,
# all of Brittany sits in one Köppen zone (no climate contrast to hold out).
SPLIT_REGIMES = ["random_id", "geographic_ood"]

# The breizhcrops package returns L1C ``X`` as (T, 13) with bands in THIS order
# (alphabetical), from which we keep the 10 S2 spectral bands + a computed NDVI.
BZ_X_BANDS = ["B1", "B10", "B11", "B12", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B9"]
BZ_S2_KEEP = ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"]
BZ_S2_BANDS = BZ_S2_KEEP + ["NDVI"]
# Only the four FRH NUTS-3 regions, matching the published BreizhCrops fold (frh01+frh02 train,
# frh03 val, frh04 test). belle-ile is deliberately EXCLUDED: including it would put it in the
# training pool and break the published-anchor comparison.
BZ_REGIONS = ["frh01", "frh02", "frh03", "frh04"]
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


def _bz_parcel_latlon(ds) -> dict[int, tuple[float, float]]:
    """Parcel ``id`` -> ``(lat, lon)`` from BreizhCrops' own parcel shapefile.

    ``geodataframe()`` downloads and caches the per-region RPG parcel polygons on first
    call. We reproject from RGF93 / Lambert-93 to WGS84 and take a representative interior
    point per parcel. Location-aware encoders (Presto, Galileo, OlmoEarth) consume lat/lon
    as a model input, so feeding the true coordinate instead of a NaN/zero placeholder
    honours their input contract. Returns an empty map (callers fall back to NaN) when the
    geometry is unavailable, so a missing shapefile never breaks the load.
    """
    try:
        gdf = ds.geodataframe().to_crs(4326)
    except Exception as exc:  # geopandas/pyproj missing, or shapefile download failed
        print(f"[breizhcrops] parcel geometry unavailable ({exc}); latlon stays NaN", flush=True)
        return {}
    pts = gdf.geometry.representative_point()
    ids = gdf["id"].to_numpy()
    return {
        int(i): (float(lat), float(lon))
        for i, lon, lat in zip(ids, pts.x.to_numpy(), pts.y.to_numpy(), strict=True)
    }


def load_benchmark(
    root: Path = Path("data/input/benchmarks"),
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
    test region). S2-only (no S1 / climate). ``latlon`` is joined per parcel id from the
    parcel shapefile (``_bz_parcel_latlon``); parcels with no geometry stay NaN.
    """
    try:
        import breizhcrops as bzh
    except ImportError as exc:
        raise ImportError("BreizhCrops needs the `breizhcrops` package (uv pip install breizhcrops).") from exc

    base = root / "breizhcrops"
    regions = regions or BZ_REGIONS
    col = {b: i for i, b in enumerate(BZ_X_BANDS)}
    keep_idx = [col[b] for b in BZ_S2_KEEP]
    b4i, b8i = col["B4"], col["B8"]

    s2_list: list[np.ndarray] = []
    labels: list[int] = []
    groups: list[str] = []
    latlons: list[tuple[float, float]] = []
    for region in regions:
        ds = bzh.BreizhCrops(region, root=str(base), level="L1C", load_timeseries=True, verbose=False)
        latlon_map = _bz_parcel_latlon(ds)
        for i in range(len(ds)):
            x, y, fid = ds[i]
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
            latlons.append(latlon_map.get(int(fid), (np.nan, np.nan)))

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
    latlon_arr = np.asarray([latlons[i] for i in order], dtype=np.float32)

    n = len(s2)
    doy = np.tile(_synthetic_month_doy(BZ_TIMESTEPS), (n, 1))
    ones = np.ones((n, BZ_TIMESTEPS), dtype=np.float32)
    zeros = np.zeros((n, BZ_TIMESTEPS), np.float32)
    return Benchmark(
        name="breizhcrops",
        label_kind="multiclass",
        s2=s2,
        s1=np.zeros((n, BZ_TIMESTEPS, 2), np.float32),
        climate=np.zeros((n, BZ_TIMESTEPS, 0), np.float32),
        s2_mask=ones.copy(),
        s1_mask=zeros.copy(),
        climate_mask=zeros.copy(),
        doy=doy,
        labels=labels_arr,
        groups=groups_arr,
        latlon=latlon_arr,
        s2_bands=BZ_S2_BANDS,
        s1_bands=["VV", "VH"],
        climate_bands=[],
        label_names=_bz_class_names(base),
        years=np.full(n, 2017, np.int64),
    )


def make_targets(bench) -> tuple[np.ndarray, np.ndarray]:
    """y = crop-class id (9 classes); groups = NUTS-3 region (geographic holdout)."""
    return bench.labels.astype(np.int64), bench.groups
