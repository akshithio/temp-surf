"""Shared types for split regimes.

A *regime* owns two things:

    (1) the domain basis: how each sample is assigned a domain label
    (2) the split strategy — how those domains become train/val/test
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def _empty() -> np.ndarray:
    return np.empty(0, dtype=np.int64)


@dataclass(frozen=True)
class Split:
    """One train/val/test partition produced by a regime.

    ``label`` identifies the held-out domain (or fold). ``val`` may be empty when a
    regime trains on the full non-target pool and leaves threshold calibration to
    the probe's own internal split. ``domain`` is the raw domain value held out (e.g. the
    Köppen zone ``"C"`` behind label ``"koppen_C"``); the runner uses it to detect a
    leave-one-domain-out regime that silently dropped a domain. Defaults to ``label``.
    """

    label: str
    train: np.ndarray
    test: np.ndarray
    val: np.ndarray = field(default_factory=_empty)
    domain: str | None = None


def geography_domains(bench) -> np.ndarray:
    """Default domain assignment: the benchmark's native region/source groups."""
    return np.asarray(bench.groups, dtype=object)
