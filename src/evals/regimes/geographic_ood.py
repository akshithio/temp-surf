"""Split regime: ``geographic_ood`` — strict leave-one-region-out holdout.

The entire held-out region is the test set; the probe never sees it (zero target
labels). One split per curated holdout region, where the region list is supplied
by the benchmark spec (``bench_mod.HOLDOUTS``). Degenerate one-class holdouts are
skipped. ``HAS_TARGET = True``: few-shot target-budget sweeps are meaningful here.

See ``random_id.py`` for the regime-module contract.
"""

from __future__ import annotations

import numpy as np
from sklearn.model_selection import train_test_split

from evals.regimes.base import Split, geography_domains

NAME = "geographic_ood"
GROUP_KIND = "geography"
HAS_TARGET = True
assign_domains = geography_domains


def make_strict_holdout_splits(
    y: np.ndarray,
    groups: np.ndarray,
    heldout_group: str,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Strict geographic holdout: the entire heldout group is the test set.

    Returns ``(train, val, test, train_val)``. The probe never sees the target
    region. Strict *SSL* exclusion (the model also never pretrained on this region)
    is enforced upstream at extraction time, not here.
    """
    idx = np.arange(len(y))
    test = idx[groups == heldout_group]
    train_val = idx[groups != heldout_group]
    if len(test) == 0:
        raise ValueError(f"No samples found for strict holdout group: {heldout_group}")
    if len(np.unique(y[test])) < 2:
        raise ValueError(f"Strict holdout group is one-class: {heldout_group}")
    if len(np.unique(y[train_val])) < 2:
        raise ValueError(f"Strict holdout training pool is one-class after excluding: {heldout_group}")
    try:
        train, val = train_test_split(train_val, test_size=0.10, random_state=seed, stratify=y[train_val])
    except ValueError:
        train, val = train_test_split(train_val, test_size=0.10, random_state=seed, stratify=None)
    return np.sort(train), np.sort(val), np.sort(test), np.sort(train_val)


def iter_splits(y, groups, *, seed, holdouts, n_folds=None, **_):
    """Yield one :class:`Split` (train/val/test) per curated holdout region."""
    for holdout in holdouts:
        try:
            train, val, test, _train_val = make_strict_holdout_splits(y, groups, holdout, seed)
        except ValueError:
            continue  # one-class holdout or empty region -> skip
        yield Split(str(holdout), train, test, val)


def iter_fold_splits(bench_mod):
    """Dense (segmentation) realization: leave-one-spatial-fold-out.

    The segmentation analogue of leave-one-region-out — each spatial fold is the held-out
    test region in turn (zero target labels), the next fold (cyclically) is val, the rest
    train. Exercises every region as a target and supports worst-region reporting. Yields
    ``(label, train_folds, val_folds, test_folds)``.
    """
    all_folds = sorted(set(bench_mod.TRAIN_FOLDS) | set(bench_mod.VAL_FOLDS) | set(bench_mod.TEST_FOLDS))
    for i, test_fold in enumerate(all_folds):
        val_fold = all_folds[(i + 1) % len(all_folds)]
        train_folds = {f for f in all_folds if f not in (test_fold, val_fold)}
        yield (f"fold_{test_fold}", train_folds, {val_fold}, {test_fold})
