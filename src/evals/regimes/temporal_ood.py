"""Split regime: ``temporal_ood`` — forward-in-time holdout.

Domain basis: YEAR (intrinsic metadata, no external data needed). Split: FORWARD — train
on earlier years, test on the most recent year. This is the "deploy next season" test;
a leave-one-year-out would let the model interpolate across years (train 2018+2020, test
2019) and hide the extrapolation failure, so the order is respected deliberately.

HAS_TARGET = True — the latest year is the held-out target, with the usual few-shot /
oracle budget sweep. Degenerate on single-year benchmarks (yields nothing), which is the
correct behavior: there is no future to deploy into.
"""

from __future__ import annotations

import numpy as np

from evals.regimes.base import Split
from evals.regimes.geographic_ood import make_strict_holdout_splits

NAME = "temporal_ood"
GROUP_KIND = "time"
HAS_TARGET = True


def year_domains(bench) -> np.ndarray:
    """Per-sample acquisition year (the temporal domain)."""
    years = getattr(bench, "years", None)
    if years is None:
        raise ValueError(f"{getattr(bench, 'name', 'benchmark')} exposes no per-sample year")
    return np.asarray(years)


assign_domains = year_domains


def iter_splits(y, groups, *, seed, holdouts=None, n_folds=None, **_):
    """Yield a single forward split: hold out the latest year, train on earlier ones."""
    years = np.asarray(groups)
    usable = sorted({int(v) for v in years.tolist() if int(v) > 0})  # 0 = unknown year
    if len(usable) < 2:
        return  # single usable year -> no forward split is possible
    latest = usable[-1]
    try:
        train, val, test, _train_val = make_strict_holdout_splits(y, years, latest, seed)
    except ValueError:
        return
    yield Split(f"test_year_{latest}", train, test, val)
