"""Split regime: ``climate_ood`` — leave-one-Köppen-climate-zone-out.

Domain basis: CLIMATE. Each sample's domain is its main Köppen–Geiger group (A–E),
looked up from its lat/lon against a static grid (``dataio/koppen.py``). The climate
label is *evaluation metadata*, never model input — the model still sees exactly the
benchmark's ``x``, so this is faithful to both input contracts.

Split: strict leave-one-domain-out, the SAME mechanism as ``geographic_ood`` — so the
climate gap is directly comparable to the region gap and you can ask "does the
geographic failure reduce to a climate failure?". HAS_TARGET = True.

Requires sample coordinates + the staged Köppen grid; where either is absent the runner
(``main._iter_splits``) catches the failure and skips this regime for that benchmark.
"""

from __future__ import annotations

import numpy as np

from dataio.koppen import koppen_main_group
from evals.regimes.base import Split
from evals.regimes.geographic_ood import make_strict_holdout_splits

NAME = "climate_ood"
GROUP_KIND = "climate"
HAS_TARGET = True


def climate_domains(bench) -> np.ndarray:
    """Assign each sample its main Köppen group from lat/lon (metadata, not model input)."""
    latlon = getattr(bench, "latlon", None)
    if latlon is None:
        raise ValueError(f"{getattr(bench, 'name', 'benchmark')} exposes no coordinates for climate domains")
    return koppen_main_group(np.asarray(latlon))


assign_domains = climate_domains


def iter_splits(y, groups, *, seed, holdouts=None, n_folds=None, **_):
    """Yield one :class:`Split` per climate zone held out (skipping the 'unknown' bin)."""
    for target in sorted(set(np.asarray(groups, dtype=object).astype(str).tolist())):
        if target in ("unknown", "nan"):
            continue  # off-grid / coordinate-less samples are not a deployment domain
        try:
            train, val, test, _train_val = make_strict_holdout_splits(y, groups, target, seed)
        except ValueError as exc:
            print(f"   !! climate_ood: Köppen zone {target!r} dropped ({exc})", flush=True)
            continue
        yield Split(f"koppen_{target}", train, test, val)
