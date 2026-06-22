"""Benchmark: PASTIS-R crop-type semantic segmentation.

The published split is fixed: folds 1-3 train, fold 4 validates, and fold 5
tests. Class 19 is void and is removed; background class 0 remains evaluated.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from dataio.get_input import (
    Benchmark,
    _synthetic_month_doy,
)

BENCHMARK = "pastis_r"
LABEL_KIND = "segmentation"
TRAIN_FOLDS = {1, 2, 3}
VAL_FOLDS = {4}
TEST_FOLDS = {5}
HOLDOUTS = [5]
# Fold-based regimes (run via the dense path, not the classification sweep). Each regime
# owns its fold logic in evals/regimes/<regime>.py (iter_fold_splits):
#   random_id      = published 1-3/4/5 fold assignment -> PASTIS's in-distribution baseline.
#   geographic_ood = leave-one-spatial-fold-out (the deployment regime, supports worst-region).
SPLIT_REGIMES = ["random_id", "geographic_ood"]
IGNORE_INDEX = 19

# PASTIS DATA_S2 is (T, 10, 128, 128) in this 10-band order. DATA_S1A is
# (T, 3, 128, 128) as VV, VH, VV/VH. The semantic target is channel 0 of
# ANNOTATIONS/TARGET_<id>.npy (20 classes: 0=background, 1-18 crops, 19=void).
PASTIS_S2_BANDS = ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"]
PASTIS_S1_BANDS = ["VV", "VH", "VV/VH"]
PASTIS_N_CLASSES = 20
PASTIS_TIMESTEPS = 12
PASTIS_TILE_SIZE = 64


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
    """One monthly 64x64 PASTIS-R tile passed to dense models."""

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
                    yield tile_id, patch.fold, tile, labels[valid]


# --------------------------------------------------------------------------- #
# Internal helpers
# --------------------------------------------------------------------------- #


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
        name="pastis_r",
        label_kind="segmentation",
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
    base = root / "pastis_r"
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
            )
        )

    if missing:
        # A partial release must be visible: don't silently evaluate over the patches that happen
        # to be present. Loud warning; raise under STRICT_DATA so a partial release fails outright.
        msg = f"PASTIS: {missing}/{len(order)} metadata patches have missing .npy files in {base}"
        if os.environ.get("STRICT_DATA", "").strip().lower() not in ("", "0", "false", "no"):
            raise ValueError(msg + " (STRICT_DATA is set)")
        print(f"   !! {msg} -- those patches are skipped (set STRICT_DATA=1 to fail instead)", flush=True)
    if not patches:
        raise ValueError(f"No PASTIS patches parsed from {base}")
    return PastisBenchmark(
        name="pastis_r",
        label_kind="segmentation",
        patches=tuple(patches),
        tile_size=PASTIS_TILE_SIZE,
        ignore_index=IGNORE_INDEX,
    )


def make_targets(bench) -> tuple[np.ndarray, np.ndarray]:
    """Dense targets are loaded tile-wise by the segmentation runner."""
    return np.empty(0, dtype=np.int64), bench.groups
