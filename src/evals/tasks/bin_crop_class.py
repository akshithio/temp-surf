"""Task: binary crop / non-crop classification (CropHarvest).

The primary task and most controlled setting: real per-sample ``is_crop`` labels,
evaluated under strict geographic holdout (one source region held out at a time).

Task spec contract (consumed by the runner in main.py):
  * BENCHMARK   -- which dataset get_input loads
  * TASK_KIND   -- "binary" -> calibrated logistic probe + binary metrics
  * HOLDOUTS    -- strict-holdout groups (None -> runner uses evals.STRICT_HOLDOUTS)
  * make_targets(bench) -> (y, groups)
"""

from __future__ import annotations

import numpy as np

BENCHMARK = "cropharvest"
TASK_KIND = "binary"
HOLDOUTS = None  # use evals.STRICT_HOLDOUTS (togo, ethiopia, lem-brazil, rwanda, togo-eval)


def make_targets(bench) -> tuple[np.ndarray, np.ndarray]:
    """y = is_crop (real label); groups = source dataset (geographic holdout)."""
    return bench.labels.astype(np.int64), bench.groups
