"""Task: crop phenology date regression (SICKLE) -- externally labeled regression.

Unlike a derived "peak NDVI" probe, the target here is an **external field
annotation**: the day-of-season of an observed phenological event (sowing /
transplanting / harvesting) for paddy plots in the Cauvery Delta, from the SICKLE
dataset's tabular annotations. The loader (`load_sickle`) puts the chosen event's
day into `bench.labels` (set `target` in get_input; default "harvesting", the
best-populated). Predicting it from frozen embeddings asks: do the embeddings
support prediction of *observed* crop phenology dates?

Geographic groups are SICKLE river parts (`RIVER_PART`); strict holdout leaves one
river part out at a time. The dataset is small (~89 annotated paddy samples in the
toy release), so treat absolute numbers as indicative.

TASK_KIND = "regression" -> run_probes_regression + regression metrics
(RMSE, MAE, R2, Pearson, Spearman).
"""

from __future__ import annotations

import numpy as np

BENCHMARK = "sickle"
TASK_KIND = "regression"
# SICKLE river parts (geographic holdout). The runner skips any that are too small.
HOLDOUTS = ["Coastal Cauvery", "Lower Cauvery", "Upper Cauvery", "Middle Vennar", "Coastal Vennar"]


def make_targets(bench) -> tuple[np.ndarray, np.ndarray]:
    """y = observed phenology day-of-season (regression); groups = river part."""
    return bench.labels.astype(np.float32), bench.groups
