"""Shared types for split regimes.

A *regime* owns two things (see the individual regime files): the **domain basis**
— how each sample is assigned a domain label (``assign_domains(bench)``, tagged by
``GROUP_KIND``) — and the **split strategy** — how those domains become
train/val/test (``iter_splits(...)``). ``iter_splits`` yields :class:`Split`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def _empty() -> np.ndarray:
    return np.empty(0, dtype=np.int64)


@dataclass(frozen=True)
class Split:
    """One train/val/test partition produced by a regime.

    ``label`` identifies the held-out domain (or fold). ``val`` may be empty when a
    regime trains on the full non-target pool and leaves threshold calibration to
    the probe's own internal split.
    """

    label: str
    train: np.ndarray
    test: np.ndarray
    val: np.ndarray = field(default_factory=_empty)


def geography_domains(bench) -> np.ndarray:
    """Default domain assignment: the benchmark's native region/source groups."""
    return np.asarray(bench.groups, dtype=object)


def phenology_domains(bench) -> np.ndarray:
    """Assign samples to coarse NDVI phenology domains.

    This is intentionally simple and benchmark-agnostic: use the loaded S2 NDVI
    channel, respect the S2 observation mask, mark low-amplitude samples, and
    otherwise group by peak-NDVI timing (early/mid/late season). All current
    classification benchmarks expose an NDVI channel in ``bench.s2``.
    """
    bands = list(getattr(bench, "s2_bands", []))
    if "NDVI" not in bands:
        raise ValueError(f"{getattr(bench, 'name', 'benchmark')} does not expose an NDVI band")
    ndvi = np.asarray(bench.s2[:, :, bands.index("NDVI")], dtype=np.float32)
    mask = np.asarray(getattr(bench, "s2_mask", np.ones(ndvi.shape, dtype=np.float32))) > 0
    valid_counts = mask.sum(axis=1)
    if ndvi.ndim != 2:
        raise ValueError(f"Expected NDVI as (N,T), got {ndvi.shape}")

    safe_max = np.where(mask, ndvi, -np.inf)
    safe_min = np.where(mask, ndvi, np.inf)
    peak = np.argmax(safe_max, axis=1)
    max_ndvi = np.max(safe_max, axis=1)
    min_ndvi = np.min(safe_min, axis=1)
    amplitude = max_ndvi - min_ndvi
    finite_amp = amplitude[np.isfinite(amplitude)]
    low_threshold = float(np.quantile(finite_amp, 0.25)) if finite_amp.size else 0.0

    domains = np.empty(ndvi.shape[0], dtype=object)
    domains[:] = "phenology_missing"
    usable = valid_counts > 0
    low_amp = usable & np.isfinite(amplitude) & (amplitude <= low_threshold)
    domains[low_amp] = "phenology_low_amplitude"

    seasonal = usable & ~low_amp
    t = max(1, ndvi.shape[1])
    early = seasonal & (peak < t / 3)
    mid = seasonal & (peak >= t / 3) & (peak < 2 * t / 3)
    late = seasonal & (peak >= 2 * t / 3)
    domains[early] = "phenology_early_peak"
    domains[mid] = "phenology_mid_peak"
    domains[late] = "phenology_late_peak"
    return domains
