"""Split regime: ``phenology_ood`` — leave-one-phenology-domain-out.

This regime changes the domain basis from geography to crop-calendar behavior:
samples are grouped by simple NDVI phenology labels, then each phenology domain
is held out in turn. The split strategy is strict leave-domain-out, so zero-shot
and few-shot target-budget sweeps have the same interpretation as
``geographic_ood`` but the target domain is phenological rather than regional.
"""

from __future__ import annotations

import numpy as np

from evals.regimes.base import Split
from evals.regimes.geographic_ood import make_strict_holdout_splits


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


NAME = "phenology_ood"
GROUP_KIND = "phenology"
HAS_TARGET = True
# Leave-one-domain-out: the runner enforces that every phenology domain yielded a split (a dropped
# degenerate domain is routed through _regime_problem so STRICT catches a partial matrix).
LEAVE_ONE_DOMAIN_OUT = True
assign_domains = phenology_domains


def iter_splits(y, groups, *, seed, holdouts=None, n_folds=None, **_):
    """Yield one :class:`Split` (train/val/test) per phenology domain held out."""
    for holdout in sorted(set(np.asarray(groups, dtype=object).astype(str).tolist())):
        try:
            train, val, test, _train_val = make_strict_holdout_splits(y, groups, holdout, seed)
        except ValueError as exc:
            print(f"   !! phenology_ood: domain {holdout!r} dropped ({exc})", flush=True)
            continue
        yield Split(holdout, train, test, val, domain=str(holdout))
