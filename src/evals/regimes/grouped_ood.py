"""Split regime: ``grouped_ood`` — random group-fold leave-out.

Partitions all regions into ``n_folds`` random blocks and holds each block out in
turn. Group-aware (no region is shared between train and test, so it is genuine
OOD), but unlike ``geographic_ood`` it does not hand-pick a single curated region
— it averages over many random region partitions. Degenerate one-class folds are
skipped. ``HAS_TARGET = True``.

See ``random_id.py`` for the regime-module contract.
"""

from __future__ import annotations

import numpy as np

from evals.regimes.base import geography_domains

NAME = "grouped_ood"
GROUP_KIND = "geography"
HAS_TARGET = True
N_FOLDS = 5  # number of random group folds
assign_domains = geography_domains


def make_grouped_holdout_folds(y: np.ndarray, groups: np.ndarray, seed: int, n_folds: int = N_FOLDS):
    """Yield ``(fold_label, train_idx, test_idx)``; each fold holds out a random group block."""
    gstr = np.array([str(g) for g in groups])
    uniq = np.array(sorted(set(gstr.tolist())))
    if len(uniq) < 2:
        return
    n_folds = int(min(n_folds, len(uniq)))
    rng = np.random.default_rng(seed)
    fold_of = {g: int(i % n_folds) for i, g in enumerate(rng.permutation(uniq))}
    idx = np.arange(len(y))
    for f in range(n_folds):
        in_test = np.array([fold_of[g] == f for g in gstr])
        test, train = idx[in_test], idx[~in_test]
        if len(test) == 0 or len(train) == 0:
            continue
        if len(np.unique(y[test])) < 2 or len(np.unique(y[train])) < 2:
            continue
        yield f"grouped_fold{f}", np.sort(train), np.sort(test)


def iter_splits(y, groups, *, seed, holdouts=None, n_folds=N_FOLDS, **_):
    """Yield ``(fold_label, train_idx, test_idx)`` for each random group fold."""
    yield from make_grouped_holdout_folds(y, groups, seed, n_folds=n_folds)
