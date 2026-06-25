"""In-distribution random split regime."""

from __future__ import annotations

import numpy as np
from sklearn.model_selection import train_test_split

from evals.regimes.base import DenseSplit, Split, geography_domains
from utils import cacheutils

NAME = "random_id"
GROUP_KIND = "geography"
HAS_TARGET = False  # train and test share regions -> no target region to sweep
assign_domains = geography_domains


def make_splits(y: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """80/10/10 stratified train/val/test on the full pool."""
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
    """Yield one in-distribution split."""
    del groups, holdouts, n_folds
    train, val, test = make_splits(y, seed)
    yield Split("random_id", train, test, val)


def make_patch_splits(patches: np.ndarray, seed: int) -> tuple[set[int], set[int], set[int]]:
    train_val, test = train_test_split(patches, test_size=0.10, random_state=seed)
    train, val = train_test_split(train_val, test_size=0.1111111111, random_state=seed + 1)
    return set(map(int, train)), set(map(int, val)), set(map(int, test))


def iter_dense_splits(bench_mod, *, emb_dir, seed, bench=None):
    del bench
    all_folds = sorted(set(bench_mod.TRAIN_FOLDS) | set(bench_mod.VAL_FOLDS) | set(bench_mod.TEST_FOLDS))
    patches = np.asarray(cacheutils.dense_fold_patches(emb_dir, set(all_folds)), dtype=np.int64)
    train, val, test = make_patch_splits(patches, seed)
    yield DenseSplit(
        "random_patch",
        set(all_folds),
        set(all_folds),
        set(all_folds),
        train_patches=train,
        val_patches=val,
        test_patches=test,
        has_target=HAS_TARGET,
        group_kind=GROUP_KIND,
    )
