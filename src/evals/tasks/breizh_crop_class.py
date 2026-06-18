"""Task: BreizhCrops crop-type classification (Brittany, Sentinel-2 L1C, 9 classes).

Multiclass crop-type classification over NUTS-3 regions of Brittany, France. Geographic
holdout reproduces the BreizhCrops paper's split direction: the held-out region is the
test set (``frh04`` in the paper) and the remaining regions train. ``groups`` is the
region, so holding out ``frh04`` matches the published evaluation.

TASK_KIND = "multiclass" -> the runner uses run_probes_multiclass + multiclass metrics.
"""

from __future__ import annotations

import numpy as np

BENCHMARK = "breizhcrops"
TASK_KIND = "multiclass"
HOLDOUTS = ["frh04"]  # paper's test region (train on frh01/frh02/frh03/belle-ile)


def make_targets(bench) -> tuple[np.ndarray, np.ndarray]:
    """y = crop-class id (9 classes); groups = NUTS-3 region (geographic holdout)."""
    return bench.labels.astype(np.int64), bench.groups
