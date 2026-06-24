"""Domain-confound diagnostics.

A measured OOD gap is only attributable to the axis it is named for if that axis is not entangled
with another. Two known cases:
  * EuroCropsML ``climate_ood`` -- Köppen zone is (here) a function of country, so a "climate" gap
    is largely a geography gap (#8).
  * CropHarvest ``temporal_ood`` -- if source datasets / regions appear in different years, a "time"
    gap may be a dataset/geography gap (#9).

This module cross-tabulates the per-sample domain bases (geography, year, climate, class) and reports
how strongly each pair is entangled, so those gaps are read as *decomposition metadata*, not as
independent robustness results. It is pure numpy (no model/probe), and the report is written next to
the probe results as ``domain_confounds.json``.
"""

from __future__ import annotations

import numpy as np


def _entropy(counts: np.ndarray) -> float:
    total = counts.sum()
    if total <= 0:
        return 0.0
    p = counts[counts > 0] / total
    return float(-(p * np.log(p)).sum())


def _contingency(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ua, ai = np.unique(np.asarray(a).astype(str), return_inverse=True)
    ub, bi = np.unique(np.asarray(b).astype(str), return_inverse=True)
    counts = np.zeros((len(ua), len(ub)), dtype=np.int64)
    np.add.at(counts, (ai, bi), 1)
    return ua, ub, counts


def confound_pair(a: np.ndarray, b: np.ndarray, max_cells: int = 30) -> dict:
    """Entanglement stats between two per-sample label axes.

    ``nmi`` is normalized mutual information in [0, 1] (1 = the two partitions carry the same
    information). ``determines_b_given_a`` = 1 - H(b|a)/H(b) is the fraction of axis ``b`` that
    knowing axis ``a`` pins down (1 = ``a`` fully determines ``b``); ``determines_a_given_b`` is the
    reverse. A small contingency table is included when both axes are low-cardinality. Either a high
    ``nmi`` or a near-1 ``determines_*`` means the gap on one axis is largely the other axis.
    """
    ua, ub, counts = _contingency(a, b)
    n = int(counts.sum())
    base = {"n_a": int(len(ua)), "n_b": int(len(ub))}
    if n == 0 or len(ua) < 2 or len(ub) < 2:
        return {**base, "nmi": 0.0, "determines_b_given_a": 0.0, "determines_a_given_b": 0.0}
    h_a = _entropy(counts.sum(1))
    h_b = _entropy(counts.sum(0))
    h_ab = _entropy(counts.reshape(-1))
    mi = h_a + h_b - h_ab
    nmi = mi / np.sqrt(h_a * h_b) if h_a > 0 and h_b > 0 else 0.0
    det_b = 1.0 - (h_ab - h_a) / h_b if h_b > 0 else 0.0  # how much a determines b
    det_a = 1.0 - (h_ab - h_b) / h_a if h_a > 0 else 0.0  # how much b determines a
    out = {
        **base,
        "nmi": round(float(np.clip(nmi, 0.0, 1.0)), 4),
        "determines_b_given_a": round(float(np.clip(det_b, 0.0, 1.0)), 4),
        "determines_a_given_b": round(float(np.clip(det_a, 0.0, 1.0)), 4),
    }
    if len(ua) <= max_cells and len(ub) <= max_cells:
        out["contingency"] = {
            str(av): {str(bv): int(counts[i, j]) for j, bv in enumerate(ub) if counts[i, j]}
            for i, av in enumerate(ua)
        }
    return out


def domain_confound_report(axes: dict[str, np.ndarray | None]) -> dict:
    """Pairwise confound stats between the supplied per-sample domain axes.

    ``axes`` maps an axis name (e.g. ``geography``, ``year``, ``climate``, ``class``) to a per-sample
    label array (``None`` / single-valued axes are skipped). Returns the cardinality of each usable
    axis and, for every pair, the :func:`confound_pair` stats -- so e.g. a near-1 ``climate__vs__
    geography`` determination flags climate_ood as confounded with the country split, and a high
    ``year__vs__geography`` entanglement flags a temporal gap that is really a source/region gap.
    """
    usable = [
        name
        for name, values in axes.items()
        if values is not None and len(np.unique(np.asarray(values).astype(str))) > 1
    ]
    report: dict = {
        "n_samples": int(len(axes[usable[0]])) if usable else 0,
        "axis_cardinality": {name: int(len(np.unique(np.asarray(axes[name]).astype(str)))) for name in usable},
        "pairs": {},
    }
    for i in range(len(usable)):
        for j in range(i + 1, len(usable)):
            report["pairs"][f"{usable[i]}__vs__{usable[j]}"] = confound_pair(axes[usable[i]], axes[usable[j]])
    return report
