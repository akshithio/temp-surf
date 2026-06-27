"""Shared data infrastructure: Benchmark dataclass and dispatch.

Each benchmark loader lives in its own spec module under ``evals/benchmarks/``.
This module provides the shared dataclasses and a thin ``get_input()`` dispatcher.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass, field, replace
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

DEFAULT_ROOT = Path("data/input/benchmarks")

# --------------------------------------------------------------------------- #
# Shared benchmark dataclasses
# --------------------------------------------------------------------------- #


@dataclass
class ModalitySeries:
    """One modality's NATIVE per-sample acquisition series — ALL source bands, true dates.

    ``values[i]`` is sample ``i``'s ``(T_i, C)`` series in ``bands`` order; ``months`` / ``doy`` /
    ``years`` are the matching per-acquisition ``(T_i,)`` calendar month (0-11), day-of-year
    (1-365), and calendar year. An absent modality has ``bands == []`` and zero-width per-sample
    arrays. There is intentionally no fixed ``T``: the cadence is whatever the source provides.
    """

    values: list[np.ndarray]
    months: list[np.ndarray]
    doy: list[np.ndarray]
    years: list[np.ndarray]
    bands: list[str]

    @classmethod
    def absent(cls, n: int) -> ModalitySeries:
        """An empty modality (no bands) for ``n`` samples — for a benchmark that lacks S1/climate."""
        return cls(
            values=[np.zeros((0, 0), dtype=np.float32) for _ in range(n)],
            months=[np.zeros(0, dtype=np.int64) for _ in range(n)],
            doy=[np.zeros(0, dtype=np.float32) for _ in range(n)],
            years=[np.zeros(0, dtype=np.int64) for _ in range(n)],
            bands=[],
        )


@dataclass
class NativeSeries:
    """A benchmark's native observations per modality — the Benchmark's single source of truth."""

    s2: ModalitySeries
    s1: ModalitySeries
    climate: ModalitySeries


@dataclass
class Benchmark:
    """Model-agnostic task descriptor: native per-sample observations + label/split metadata.

    Deliberately NO shared, pre-aggregated, band-subset tensor: each model builds its own input via
    the view accessors (:meth:`monthly` / :meth:`native_series`), selecting the bands it needs from
    the FULL native band set — so a model is never handicapped by a temporal or band choice made
    upstream to suit a different model. The cross-model invariants (labels, groups, latlon, years)
    stay fixed so the OOD gap is comparable across models.
    """

    name: str
    label_kind: str  # "binary" | "multiclass" | "regression"
    native: NativeSeries
    labels: np.ndarray
    groups: np.ndarray
    latlon: np.ndarray  # (N, 2) [lat, lon]; used by location-aware models (Presto, ...)
    label_names: list[str] | None = None  # class id -> name, for multiclass
    years: np.ndarray | None = None  # (N,) representative calendar year per sample
    sample_ids: np.ndarray | None = None
    official_splits: dict[str, dict[str, np.ndarray]] = field(default_factory=dict)
    data_quality: dict[str, Any] = field(default_factory=dict)
    monthly_order: np.ndarray | None = None

    @property
    def n_samples(self) -> int:
        return len(self.native.s2.values)

    @property
    def s2_bands(self) -> list[str]:
        return self.native.s2.bands

    @property
    def s1_bands(self) -> list[str]:
        return self.native.s1.bands

    @property
    def climate_bands(self) -> list[str]:
        return self.native.climate.bands

    def monthly(self, modality: str, n_months: int = 12):
        """Monthly-composite view of a modality's FULL native band set.

        Returns ``(values (N, n_months, C), mask (N, n_months), doy (N, n_months), bands)``. A model
        that wants a fixed monthly grid (Presto, Galileo, OlmoEarth, AgriFM) calls this and maps the
        bands it needs by name, ignoring the rest. Empty months are zeros with mask 0; an absent
        modality yields ``C == 0`` with an all-zero mask.
        """
        ms: ModalitySeries = getattr(self.native, modality)
        n, c = self.n_samples, len(ms.bands)
        values = np.zeros((n, n_months, c), dtype=np.float32)
        mask = np.zeros((n, n_months), dtype=np.float32)
        for i in range(n):
            if c and len(ms.values[i]):
                values[i], mask[i] = monthly_composite(ms.values[i], ms.months[i], n_months)
        order = np.arange(n_months) if self.monthly_order is None else np.asarray(self.monthly_order, dtype=np.int64)
        if len(order) != n_months or set(order.tolist()) != set(range(n_months)):
            raise ValueError(f"monthly_order must be a permutation of 0..{n_months - 1}")
        doy = np.broadcast_to(_synthetic_month_doy(n_months)[order], (n, n_months)).astype(np.float32)
        return values[:, order], mask[:, order], doy, list(ms.bands)

    def native_series(self, modality: str):
        """Native (ragged) view: ``(values_list, doy_list, months_list, bands)`` — the full
        per-sample acquisition series, for a model that does its own temporal handling (TESSERA)."""
        ms: ModalitySeries = getattr(self.native, modality)
        return ms.values, ms.doy, ms.months, list(ms.bands)

    def available_modalities(self) -> set[str]:
        """Modalities this benchmark actually provides (for the #10 input-footprint report): always
        ``s2`` + ``time``; ``s1`` / ``climate`` when their band list is non-empty; ``latlon`` when any
        coordinate is finite and non-zero (so a NaN/zero placeholder doesn't count as location info).
        """
        mods = {"s2", "time"}
        if self.native.s1.bands:
            mods.add("s1")
        if self.native.climate.bands:
            mods.add("climate")
        ll = np.asarray(self.latlon, dtype=float)
        if ll.size and np.isfinite(ll).any() and bool(np.any(ll != 0.0)):
            mods.add("latlon")
        return mods

    def s2_only(self) -> Benchmark:
        """A copy restricted to Sentinel-2 (+ its temporal axis): S1/climate emptied, coordinates
        zeroed. The common-input view for the #10 fairness table — every model then sees only the
        modality they ALL share, so a robustness difference can't be attributed to extra inputs
        (coordinates / climate / S1) rather than the representation."""
        n = self.n_samples
        return replace(
            self,
            native=NativeSeries(s2=self.native.s2, s1=ModalitySeries.absent(n), climate=ModalitySeries.absent(n)),
            latlon=np.zeros((n, 2), dtype=np.float32),
        )



# --------------------------------------------------------------------------- #
# Shared utilities
# --------------------------------------------------------------------------- #


def _synthetic_month_doy(timesteps: int) -> np.ndarray:
    """Synthetic day-of-year for monthly-regularized benchmarks."""
    if timesteps < 1 or timesteps > 12:
        raise ValueError(f"monthly day-of-year table requires 1..12 timesteps, got {timesteps}")
    days = [datetime(2000, m, 15).timetuple().tm_yday for m in range(1, timesteps + 1)]
    return np.asarray(days, dtype=np.float32)


def monthly_composite(values: np.ndarray, months: np.ndarray, n_months: int = 12) -> tuple[np.ndarray, np.ndarray]:
    """Mean-composite a native-cadence series into ``n_months`` calendar-month bins.

    ``values`` is ``(T, C)`` and ``months`` is ``(T,)`` 0-based calendar month. Returns the
    ``(n_months, C)`` month-mean values and a ``(n_months,)`` availability mask (1 where a month
    had >=1 acquisition; empty months are zero with mask 0). This is the canonical monthly-cadence
    view a month-indexed model (e.g. Presto) requests in its own ``to_native`` — defined once here
    so the aggregation lives with the model that wants it, not baked into every loader.
    """
    values = np.asarray(values, dtype=np.float32)
    months = np.asarray(months).astype(np.int64)
    c = values.shape[1] if values.ndim == 2 else 0
    out = np.zeros((n_months, c), dtype=np.float32)
    mask = np.zeros(n_months, dtype=np.float32)
    for m in range(n_months):
        sel = months == m
        if sel.any():
            out[m] = values[sel].mean(axis=0)
            mask[m] = 1.0
    return out, mask


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
    "pastis": "evals.benchmarks.pastis",
}


def get_input(name: str, root: Path = DEFAULT_ROOT, **kwargs) -> Any:
    """Load a benchmark by name, delegating to the benchmark's own loader."""
    if name not in LOADERS:
        raise KeyError(f"Unknown benchmark {name!r}. Known: {sorted(LOADERS)}")
    mod = importlib.import_module(LOADERS[name])
    return mod.load_benchmark(root=root, **kwargs)
