"""Shared data infrastructure: Benchmark dataclass and dispatch.

Each benchmark loader lives in its own spec module under ``evals/benchmarks/``.
This module provides the shared dataclasses and a thin ``get_input()`` dispatcher.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

DEFAULT_ROOT = Path("data/input/benchmarks")

# --------------------------------------------------------------------------- #
# Shared benchmark dataclasses
# --------------------------------------------------------------------------- #


@dataclass
class Benchmark:
    """Dense, in-memory multimodal pixel/parcel time series for one benchmark.

    Modality arrays are ``(N, T, C)`` with the spatial 1x1 dimension squeezed out.
    ``*_mask`` are ``(N, T)`` per-timestep availability (1 = observed). ``doy`` is
    ``(N, T)`` day-of-year. ``labels`` are the label target (binary is_crop or
    encoded class id); ``groups`` are the strict-holdout group (dataset for
    CropHarvest, country for EuroCropsML).
    """

    name: str
    label_kind: str  # "binary" | "multiclass" | "regression"
    s2: np.ndarray
    s1: np.ndarray
    climate: np.ndarray
    s2_mask: np.ndarray
    s1_mask: np.ndarray
    climate_mask: np.ndarray
    doy: np.ndarray
    labels: np.ndarray
    groups: np.ndarray
    latlon: np.ndarray  # (N, 2) as [lat, lon]; used by location-aware models (Presto, ...)
    s2_bands: list[str]
    s1_bands: list[str]
    climate_bands: list[str]
    label_names: list[str] | None = None  # class id -> name, for multiclass
    years: np.ndarray | None = None  # (N,) calendar year of each sample's observation window

    @property
    def n_samples(self) -> int:
        return int(self.s2.shape[0])

    @property
    def timesteps(self) -> int:
        return int(self.s2.shape[1])



# --------------------------------------------------------------------------- #
# Shared utilities
# --------------------------------------------------------------------------- #


def _synthetic_month_doy(timesteps: int) -> np.ndarray:
    """Synthetic day-of-year for monthly-regularized benchmarks."""
    days = [datetime(2000, m, 15).timetuple().tm_yday for m in range(1, timesteps + 1)]
    return np.asarray(days, dtype=np.float32)


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


# --------------------------------------------------------------------------- #
# Dispatch
# --------------------------------------------------------------------------- #

LOADERS: dict[str, str] = {
    "cropharvest": "evals.benchmarks.cropharvest",
    "eurocropsml": "evals.benchmarks.eurocropsml",
    "breizhcrops": "evals.benchmarks.breizhcrops",
    "pastis_r": "evals.benchmarks.pastis_r",
}


def get_input(name: str, root: Path = DEFAULT_ROOT, **kwargs) -> Any:
    """Load a benchmark by name, delegating to the benchmark's own loader."""
    if name not in LOADERS:
        raise KeyError(f"Unknown benchmark {name!r}. Known: {sorted(LOADERS)}")
    mod = importlib.import_module(LOADERS[name])
    return mod.load_benchmark(root=root, **kwargs)
