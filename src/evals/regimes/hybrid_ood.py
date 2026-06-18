"""Split regime: ``hybrid_ood`` — geographic holdout × source group-folds.

For each curated target region, train on different random subsets of the SOURCE
regions (grouped folds over the source pool) and always test on the full target
region. This separates source-composition effects from pure geographic transfer.
It composes the two underlying primitives — strict holdout (to carve out the
target region and the source pool) and grouped folds (over that source pool).
``HAS_TARGET = True``.

See ``random_id.py`` for the regime-module contract.
"""

from __future__ import annotations

import numpy as np

from evals.regimes.base import geography_domains
from evals.regimes.geographic_ood import make_strict_holdout_splits
from evals.regimes.grouped_ood import N_FOLDS, make_grouped_holdout_folds

NAME = "hybrid_ood"
GROUP_KIND = "geography"
HAS_TARGET = True
assign_domains = geography_domains


def iter_splits(y, groups, *, seed, holdouts, n_folds=N_FOLDS, **_):
    """Yield ``("<holdout>/<fold>", train_idx, target_test_idx)`` per target × source-fold."""
    for holdout in holdouts:
        try:
            _src_train, _src_val, target_test, src_pool = make_strict_holdout_splits(y, groups, holdout, seed)
        except ValueError:
            continue  # one-class holdout or empty region -> skip
        # Grouped folds over the source pool only; map fold indices back to the original space.
        src_y, src_g = y[src_pool], groups[src_pool]
        for fold_label, fold_train, _fold_test in make_grouped_holdout_folds(src_y, src_g, seed, n_folds=n_folds):
            train_idx = np.sort(src_pool[fold_train])
            yield f"{holdout}/{fold_label}", train_idx, target_test
