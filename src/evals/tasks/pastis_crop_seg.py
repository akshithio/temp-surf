"""Task: PASTIS-R crop-type semantic segmentation.

The published split is fixed: folds 1-3 train, fold 4 validates, and fold 5
tests. Class 19 is void and is removed; background class 0 remains evaluated.
"""

from __future__ import annotations

import numpy as np

BENCHMARK = "pastis"
TASK_KIND = "segmentation"
TRAIN_FOLDS = {1, 2, 3}
VAL_FOLDS = {4}
TEST_FOLDS = {5}
HOLDOUTS = [5]
IGNORE_INDEX = 19


def make_targets(bench) -> tuple[np.ndarray, np.ndarray]:
    """Dense targets are loaded tile-wise by the segmentation runner."""
    return np.empty(0, dtype=np.int64), bench.groups
