"""Split regime: ``phenology_ood`` — leave-one-phenology-domain-out.

This regime changes the domain basis from geography to crop-calendar behavior:
samples are grouped by simple NDVI phenology labels, then each phenology domain
is held out in turn. The split strategy is strict leave-domain-out, so zero-shot
and few-shot target-budget sweeps have the same interpretation as
``geographic_ood`` but the target domain is phenological rather than regional.
"""

from __future__ import annotations

import numpy as np

from evals.regimes.base import phenology_domains
from evals.regimes.geographic_ood import make_strict_holdout_splits

NAME = "phenology_ood"
GROUP_KIND = "phenology"
HAS_TARGET = True
assign_domains = phenology_domains


def iter_splits(y, groups, *, seed, holdouts=None, n_folds=None, **_):
    """Yield one ``(phenology_domain, train_idx, test_idx)`` per phenology group."""
    for holdout in sorted(set(np.asarray(groups, dtype=object).astype(str).tolist())):
        try:
            train, _val, test, _train_val = make_strict_holdout_splits(y, groups, holdout, seed)
        except ValueError:
            continue
        yield holdout, train, test
