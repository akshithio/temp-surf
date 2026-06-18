"""Split construction for the evaluation protocol."""

from __future__ import annotations

import numpy as np
from sklearn.model_selection import train_test_split


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


def make_strict_holdout_splits(
    y: np.ndarray,
    groups: np.ndarray,
    heldout_group: str,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Strict geographic holdout: the entire heldout group is the test set.

    Returns (train, val, test, train_val). The probe never sees the target region.
    Strict *SSL* exclusion (the encoder also never pretrained on this region) is
    enforced upstream at extraction time, not here.
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


def make_grouped_holdout_folds(
    y: np.ndarray,
    groups: np.ndarray,
    seed: int,
    n_folds: int = 5,
):
    """Grouped OOD: each fold holds out a disjoint, *randomly chosen* subset of groups.

    Group-aware (no group is shared between train and test, so it is genuine OOD)
    but, unlike ``make_strict_holdout_splits``, it does not hand-pick a single
    curated region -- it partitions all groups into ``n_folds`` random blocks and
    holds each out in turn. This gives an OOD anchor that averages over many random
    region partitions rather than one curated leave-one-out. Yields
    ``(fold_label, train_idx, test_idx)``; degenerate one-class folds are skipped.
    """
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


def subset_indices(y: np.ndarray, budget: float, seed: int, stratify: bool = True) -> np.ndarray:
    """Sub-sample of indices for a sparse-label budget.

    Falls back to non-stratified if a class is too small to stratify at
    the requested budget (expected for multiclass at tiny budgets -- unseen-class
    drops are part of the EuroCropsML transfer story).
    """
    idx = np.arange(len(y))
    if budget >= 1.0:
        return idx
    k = min(len(idx) - 1, max(2, int(round(budget * len(idx)))))
    strat = y if stratify else None
    try:
        sub, _ = train_test_split(idx, train_size=k, random_state=seed, stratify=strat)
    except ValueError:
        sub, _ = train_test_split(idx, train_size=k, random_state=seed, stratify=None)
    return np.sort(sub)
