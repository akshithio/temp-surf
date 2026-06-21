"""Shared types for split regimes.

A *regime* owns two things:

    (1) the domain basis: how each sample is assigned a domain label
    (2) the split strategy — how those domains become train/val/test
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.model_selection import train_test_split


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


def holdout_val(train: np.ndarray, y: np.ndarray, seed: int, frac: float = 0.10) -> tuple[np.ndarray, np.ndarray]:
    """Carve a stratified validation set out of a training index array.

    Returns ``(train_minus_val, val)``. Used by regimes whose split strategy does
    not already produce a held-out val (grouped/hybrid) so every regime yields a
    real train/val/test triple. Falls back to non-stratified, then to an empty val,
    when the training pool is too small or single-class to split.
    """
    train = np.asarray(train)
    if len(train) < 10 or len(np.unique(y[train])) < 2:
        return np.sort(train), _empty()
    try:
        tr, val = train_test_split(train, test_size=frac, random_state=seed, stratify=y[train])
    except ValueError:
        tr, val = train_test_split(train, test_size=frac, random_state=seed, stratify=None)
    return np.sort(tr), np.sort(val)
