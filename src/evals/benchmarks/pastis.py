"""PASTIS-R crop-type semantic segmentation."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from dataio.get_input import (
    Benchmark,
    ModalitySeries,
    NativeSeries,
    _synthetic_month_doy,
)
from evals.confounds import score_segmentation_streamed
from evals.probes import FeatureTransform, _apply, fit_probe_multiclass
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
SPATIAL_CLUSTER_SPLIT = {
    "label": "spatial_cluster_purge2km",
    "n_clusters": 12,
    "val_fraction": 0.10,
    "test_fraction": 0.20,
    "purge_km": 2.0,
}
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
            self.s2_months,
            self.s1_months,
            self.labels.reshape(-1),
            self.valid.reshape(-1),
            self.fold,
            self.latlon,
        )


@dataclass(frozen=True)
class PastisBenchmark:
    """Lazy PASTIS-R release descriptor."""

    name: str
    label_kind: str
    patches: tuple[PastisPatch, ...]
    tile_size: int = 64
    ignore_index: int = IGNORE_INDEX

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
    def latlon(self) -> np.ndarray:
        tiles_per_patch = (128 // self.tile_size) ** 2
        return np.repeat([patch.latlon for patch in self.patches], tiles_per_patch, axis=0).astype(np.float32)

    def iter_tiles(self, folds: set[int] | None = None):
        """Yield native-cadence tiles."""
        for patch in self.patches:
            if folds is not None and patch.fold not in folds:
                continue
            s2 = np.load(patch.s2_path, mmap_mode="r")  # (T_s2, 10, 128, 128) native cadence
            s1 = np.load(patch.s1_path, mmap_mode="r")  # (T_s1, 3, 128, 128)
            target = np.load(patch.target_path, mmap_mode="r")[0]
            s2_ones = np.ones(s2.shape[0], dtype=np.float32)
            s1_ones = np.ones(s1.shape[0], dtype=np.float32)

            for row in range(0, target.shape[0], self.tile_size):
                for col in range(0, target.shape[1], self.tile_size):
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


def _geometry_latlon(geometry: dict[str, Any] | None) -> tuple[float, float]:
    def valid(lat: float, lon: float) -> bool:
        return -90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0

    if geometry:
        try:
            from shapely.geometry import shape

            point = shape(geometry).centroid
            lat, lon = float(point.y), float(point.x)
            if valid(lat, lon):
                return (lat, lon)
        except Exception:
            pass

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
        return (np.nan, np.nan)
    lon = float(np.mean([c[0] for c in coords]))
    lat = float(np.mean([c[1] for c in coords]))
    if valid(lat, lon):
        return (lat, lon)
    return (np.nan, np.nan)


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
    missing = 0
    for i in order:
        r = rows[int(i)]
        pid = int(r["ID_PATCH"])
        s2_path = base / "DATA_S2" / f"S2_{pid}.npy"
        s1_path = base / "DATA_S1A" / f"S1A_{pid}.npy"
        target_path = base / "ANNOTATIONS" / f"TARGET_{pid}.npy"
        if not (s2_path.exists() and s1_path.exists() and target_path.exists()):
            missing += 1
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
            )
        )

    if missing:
        msg = f"PASTIS: {missing}/{len(order)} metadata patches have missing .npy files in {base}"
        if os.environ.get("OVERWRITE_MODE", "").strip().lower() not in ("", "0", "false", "no"):
            raise ValueError(msg + " (OVERWRITE_MODE is set)")
        print(f"   !! {msg} -- those patches are skipped (set OVERWRITE_MODE=1 to fail instead)", flush=True)
    if not patches:
        raise ValueError(f"No PASTIS patches parsed from {base}")
    return PastisBenchmark(
        name="pastis",
        label_kind="segmentation",
        patches=tuple(patches),
        tile_size=PASTIS_TILE_SIZE,
        ignore_index=IGNORE_INDEX,
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
    transform: FeatureTransform | None = None,
    budgets: list[float],
    meta: dict[str, Any] | None = None,
    family: str = "logistic",
) -> None:
    """PASTIS source-fraction sweep, scored on every valid pixel in each evaluation fold."""
    _validate_source_budgets(budgets)
    meta = dict(meta or {})
    x_train = _apply(transform, x_train)
    x_val = _apply(transform, x_val)
    eval_classes = np.arange(19, dtype=np.int64)
    for budget in budgets:
        sub_seed = perf._budget_seed(seed, budget)
        sub = perf.subset_indices(y_train, float(budget), sub_seed, stratify=True)
        clf, probe_meta = fit_probe_multiclass(
            x_train[sub], y_train[sub], sub_seed, x_val=x_val, y_val=y_val, family=family
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
                **score_segmentation_streamed(clf, tiles(), eval_classes, transform=transform),
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
    transform: FeatureTransform | None = None,
    budgets: list[int | float],
    meta: dict[str, Any] | None = None,
    family: str = "logistic",
    target_id_budget: float | int = -1,
) -> None:
    """Patch-level dense few-shot / oracle curve with a full-target zero-shot anchor."""
    meta = dict(meta or {})
    x_source = _apply(transform, x_source)
    eval_classes = np.arange(19, dtype=np.int64)
    patches = np.array(sorted({int(p) for p in target_patches}))
    all_patches = set(patches.tolist())
    degenerate = len(patches) < 2
    split_rng = np.random.default_rng(perf._budget_seed(seed, 0.5))
    perm = split_rng.permutation(patches)
    n_test_patches = max(1, int(round(0.2 * len(patches)))) if not degenerate else len(patches)
    test_patches = set(perm[:n_test_patches].tolist())
    pool_order = [int(p) for p in perm.tolist() if p not in test_patches]

    for budget in budgets:
        if budget != 0 and (degenerate or not pool_order):
            continue
        sub_seed = perf._budget_seed(seed, budget)
        cal_x, cal_y, tune_internal = x_val, y_val, False
        if budget == 0:
            x_tr, y_tr = x_source, y_source
        elif budget == target_id_budget:
            xo, yo = sample_target(set(pool_order), sub_seed)[:2]
            x_tr, y_tr = _apply(transform, xo), yo
            cal_x, cal_y, tune_internal = None, None, True
        else:
            k = min(len(pool_order), perf._target_budget_count(budget, len(pool_order)))
            xf, yf = sample_target(set(pool_order[:k]), sub_seed)[:2]
            x_tr = np.concatenate([x_source, _apply(transform, xf)])
            y_tr = np.concatenate([y_source, yf])
        clf, probe_meta = fit_probe_multiclass(
            x_tr, y_tr, sub_seed, x_val=cal_x, y_val=cal_y, family=family, tune_internal=tune_internal
        )
        rows.append({
            **meta,
            "evaluation_split": "held_out",
            "budget_type": "target",
            "label_budget": budget,
            "seed": seed,
            "n_train_sub": int(len(y_tr)),
            **probe_meta,
            **score_segmentation_streamed(clf, stream_target(test_patches), eval_classes, transform=transform),
        })
        if budget == 0:
            rows.append({
                **meta,
                "evaluation_split": "full",
                "budget_type": "target",
                "label_budget": budget,
                "seed": seed,
                "n_train_sub": int(len(y_tr)),
                **probe_meta,
                **score_segmentation_streamed(clf, stream_target(all_patches), eval_classes, transform=transform),
            })
