"""Split regime: ``random_id`` — in-distribution stratified split.

Train and test are drawn from the SAME region pool (80/10/10 stratified), so this
is the easy in-distribution upper bound, not a transfer test. There is no target
region, so no few-shot target-budget sweep applies (``HAS_TARGET = False``).

Regime-module contract:
  * ``NAME``       -- regime id, identical to this module's filename.
  * ``HAS_TARGET`` -- whether OOD target-budget sweeps are meaningful.
  * ``iter_splits(y, groups, *, seed, holdouts, n_folds)`` -> yields ``(label, train_idx, test_idx)``.

Each regime owns *all* details of how its own splitting is done.
"""

from __future__ import annotations

import numpy as np
from sklearn.model_selection import train_test_split

from evals.regimes.base import Split, geography_domains

NAME = "random_id"
GROUP_KIND = "geography"
HAS_TARGET = False  # train and test share regions -> no target region to sweep
assign_domains = geography_domains


def make_splits(y: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """80/10/10 stratified train/val/test on the full pool (the easy upper bound)."""
    idx = np.arange(len(y))
    try:
        train_val, test = train_test_split(idx, test_size=0.10, random_state=seed, stratify=y)
    except ValueError:
        train_val, test = train_test_split(idx, test_size=0.10, random_state=seed, stratify=None)
    try:
        train, val = train_test_split(
            train_val,
            test_size=0.1111111111,
            random_state=seed + 1,
            stratify=y[train_val],
        )
    except ValueError:
        train, val = train_test_split(
            train_val,
            test_size=0.1111111111,
            random_state=seed + 1,
            stratify=None,
        )
    return np.sort(train), np.sort(val), np.sort(test)


def iter_splits(y, groups, *, seed, holdouts=None, n_folds=None, **_):
    """Yield the single in-distribution :class:`Split` (train/val/test)."""
    train, val, test = make_splits(y, seed)
    yield Split("random_id", train, test, val)


def iter_fold_splits(bench_mod):
    """Dense (segmentation) realization: the dataset's published fold assignment.

    PASTIS is per-pixel, so there is no flat IID sample to draw an 80/10/10 split from.
    Its published cross-validation folds (train 1-3 / val 4 / test 5) are spatially balanced
    rather than a held-out block, so they stand in as the *in-distribution* reference the OOD
    gap is measured against — the dense analogue of the classification ``random_id`` split.
    Yields ``(label, train_folds, val_folds, test_folds)``.
    """
    test_fold = sorted(bench_mod.TEST_FOLDS)[0]
    yield (
        f"fold_{test_fold}",
        set(bench_mod.TRAIN_FOLDS),
        set(bench_mod.VAL_FOLDS),
        set(bench_mod.TEST_FOLDS),
    )
