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
    ModalitySeries,
    NativeSeries,
    _synthetic_month_doy,
)

BENCHMARK = "breizhcrops"
LABEL_KIND = "multiclass"
HOLDOUTS = ["frh04"]  # paper's TEST region
VAL_HOLDOUT = "frh03"
OFFICIAL_HOLDOUTS = ["frh04"]
OFFICIAL_VAL_HOLDOUT = "frh03"
GEOGRAPHIC_HOLDOUTS = ["frh01", "frh02", "frh03", "frh04"]
GEOGRAPHIC_PURGE_KM = 5.0
SPATIAL_CLUSTER_SPLIT = {
    "label": "spatial_cluster_purge5km",
    "n_clusters": 8,
    "val_fraction": 0.10,
    "test_fraction": 0.20,
    "purge_km": 5.0,
}
SPLIT_REGIMES = ["random_id", "official", "geographic_ood", "spatial_cluster_ood"]

# The breizhcrops package returns L1C ``X`` as (T, 13) with bands in THIS order
# (alphabetical), from which we keep the 10 S2 spectral bands + a computed NDVI.
BZ_X_BANDS = ["B1", "B10", "B11", "B12", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B9"]
BZ_S2_KEEP = ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"]
BZ_S2_BANDS = BZ_S2_KEEP + ["NDVI"]
BZ_S2_BANDS_ALL = BZ_X_BANDS + ["NDVI"]  # ALL native L1C bands + computed NDVI (no pre-selection)
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
    """Load BreizhCrops L1C parcels at their native cadence (crop-type classification).

    Uses the ``breizhcrops`` package to read each NUTS-3 region's variable-length L1C parcels,
    keeps ALL native bands + a computed NDVI, rescales reflectance to DN (x1e4, matching the other
    benchmarks), and carries the FULL acquisition series (no resampling) so models do their own
    temporal handling. ``groups`` = region (geographic holdout; frh04 is the paper's test region).
    S2-only (no S1 / climate). ``latlon`` is joined per parcel id from the parcel shapefile
    (``_bz_parcel_latlon``); parcels with no geometry stay NaN. NOTE: real per-acquisition dates are
    not exposed by this loader, so each parcel's acquisitions are assigned an even synthetic spread
    over the calendar year -- the full cadence is preserved, only the month labels are approximate.
    """
    try:
        import breizhcrops as bzh
    except ImportError as exc:
        raise ImportError("BreizhCrops needs the `breizhcrops` package (uv pip install breizhcrops).") from exc

    base = root / "breizhcrops"
    regions = regions or BZ_REGIONS
    b4i, b8i = BZ_X_BANDS.index("B4"), BZ_X_BANDS.index("B8")

    series: list[np.ndarray] = []
    labels: list[int] = []
    groups: list[str] = []
    latlons: list[tuple[float, float]] = []
    # Stable per-parcel id (additive metadata only): the breizhcrops field id, qualified by region
    # because field ids are not unique across NUTS-3 regions. Carried through the exact same order
    # as every other column, so it never affects sample membership, ordering, labels, or embeddings.
    sample_ids: list[str] = []
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
            series.append(np.concatenate([x, ndvi[:, None]], axis=1).astype(np.float32))  # all 13 bands + NDVI
            labels.append(int(y))
            groups.append(region)
            latlons.append(latlon_map.get(int(fid), (np.nan, np.nan)))
            sample_ids.append(f"{region}:{int(fid)}")

    if not series:
        raise ValueError(f"No BreizhCrops parcels parsed from {base}")

    order = np.arange(len(series))
    if shuffle:
        order = np.random.default_rng(seed).permutation(order)
    if max_samples:
        order = order[:max_samples]
    series = [series[i] for i in order]
    labels_arr = np.asarray([labels[i] for i in order], dtype=np.int64)
    groups_arr = np.asarray([groups[i] for i in order], dtype=object)
    latlon_arr = np.asarray([latlons[i] for i in order], dtype=np.float32)
    sample_ids_arr = np.asarray([sample_ids[i] for i in order], dtype=object)

    n = len(series)
    months = [np.clip((np.arange(len(s)) / max(len(s) - 1, 1) * 11).round().astype(np.int64), 0, 11) for s in series]
    doy = [_synthetic_month_doy(12)[m].astype(np.float32) for m in months]
    years_l = [np.full(len(s), 2017, dtype=np.int64) for s in series]
    native = NativeSeries(
        s2=ModalitySeries(series, months, doy, years_l, BZ_S2_BANDS_ALL),
        s1=ModalitySeries.absent(n),
        climate=ModalitySeries.absent(n),
    )
    return Benchmark(
        name="breizhcrops",
        label_kind="multiclass",
        native=native,
        labels=labels_arr,
        groups=groups_arr,
        latlon=latlon_arr,
        label_names=_bz_class_names(base),
        years=np.full(n, 2017, np.int64),
        sample_ids=sample_ids_arr,
    )


def make_targets(bench) -> tuple[np.ndarray, np.ndarray]:
    """y = crop-class id (9 classes); groups = NUTS-3 region (geographic holdout)."""
    return bench.labels.astype(np.int64), bench.groups
