"""Benchmark: BreizhCrops crop-type classification (Brittany, Sentinel-2 L1C, 9 classes).

Multiclass crop-type classification over NUTS-3 regions of Brittany, France. Geographic
holdout reproduces the BreizhCrops paper's split direction: the held-out region is the
test set (``frh04`` in the paper) and the remaining regions train. ``groups`` is the
region, so holding out ``frh04`` matches the published evaluation.

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
HOLDOUTS = ["frh04"]  # paper's test region (train on frh01/frh02/frh03/belle-ile)
# Only ~4-5 NUTS-3 domains -> random group folds ≈ leave-one-region-out, so grouped_ood
# is omitted here (redundant with geographic_ood).
SPLIT_REGIMES = ["random_id", "geographic_ood", "phenology_ood"]

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
    test region). S2-only (no S1 / climate).
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
        latlon=np.full((n, 2), np.nan, np.float32),
        s2_bands=BZ_S2_BANDS,
        s1_bands=["VV", "VH"],
        climate_bands=[],
        label_names=_bz_class_names(base),
        years=np.full(n, 2017, np.int64),
    )


def make_targets(bench) -> tuple[np.ndarray, np.ndarray]:
    """y = crop-class id (9 classes); groups = NUTS-3 region (geographic holdout)."""
    return bench.labels.astype(np.int64), bench.groups
