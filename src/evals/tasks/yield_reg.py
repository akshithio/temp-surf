"""Task: field-level crop yield regression (YieldSAT).

YieldSAT is a CVPR 2026 high-resolution crop-yield benchmark built from combine
harvester yield maps, Sentinel-2 time series, weather, soil, and topography
across Argentina, Brazil, Germany, and Uruguay. The loader uses the ML-ready
NetCDF release and aggregates its pixel-level samples to one field-level sample
per ``field_shared_name``. The target remains externally measured yield, averaged
over the field pixels.

Groups are countries. Strict holdout therefore tests cross-country transfer,
which matches YieldSAT's distribution-shift motivation while keeping the task
compatible with the repository's frozen-embedding probe protocol.
"""

from __future__ import annotations

import numpy as np

BENCHMARK = "yieldsat"
TASK_KIND = "regression"
HOLDOUTS = ["Argentina", "Brazil", "Germany", "Uruguay"]


def make_targets(bench) -> tuple[np.ndarray, np.ndarray]:
    """y = field-mean yield in t/ha; groups = country."""
    return bench.labels.astype(np.float32), bench.groups
