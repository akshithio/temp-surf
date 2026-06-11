"""Task: multi-class crop-type classification (EuroCropsML).

Real per-parcel crop-type labels. EuroCropsML labels are 10-digit HCAT codes
(Hierarchical Crop and Agriculture Taxonomy) -- the full code identifies a
specific crop/variety, which yields ~100+ long-tail classes and a near-floor
macro-F1. HCAT is hierarchical, so truncating the code to its leading digits is a
principled coarsening to a broader crop-type level. We truncate to
``HCAT_PREFIX`` digits (6 -> ~36 crop-type classes, vs ~100 at full granularity)
and re-encode to contiguous ids.

Evaluated as transnational transfer: train on Latvia + Portugal, test on Estonia
(``groups`` is the country, so holding out "Estonia" reproduces that split).

TASK_KIND = "multiclass" -> the runner uses run_probes_multiclass + multiclass
metrics (macro/weighted F1, balanced acc, accuracy, macro AUC). Unseen-class drops
at low label budgets are expected and are part of the transfer story.
"""

from __future__ import annotations

import numpy as np

BENCHMARK = "eurocropsml"
TASK_KIND = "multiclass"
HOLDOUTS = ["Estonia"]  # train Latvia+Portugal -> test Estonia (official transnational split)
HCAT_PREFIX = 6  # truncate the 10-digit HCAT code to this many leading digits (crop-type level)


def make_targets(bench) -> tuple[np.ndarray, np.ndarray]:
    """y = HCAT-coarsened crop-type id; groups = country (transnational holdout)."""
    if bench.label_names is None:
        raise ValueError("crop-class needs bench.label_names (HCAT codes) for coarsening.")
    codes = np.asarray([str(bench.label_names[i]) for i in bench.labels])
    coarse = np.array([c[:HCAT_PREFIX] for c in codes])
    _, y = np.unique(coarse, return_inverse=True)  # contiguous class ids
    return y.astype(np.int64), bench.groups
