"""Raw-feature baseline 

This is a baseline and not a learned encoder. It turns each sample's multimodal time
series directly into a feature vector and feeds it to the same machinery as every FM,
for a fair comparison. 

Modes (``RAW_MODE`` / ``mode``):
  * ``flatten`` (default) — concat S2+S1+climate over time, flattened to ``(N, T*C)``.
    The purest "no representation learning" control.
  * ``stats``  — per-band temporal summary stats (mean/std/min/max/amplitude); compact and
    robust to padding.

Both modes are purely PER-SAMPLE (no fitting), so the cached "embeddings" carry no information
across samples and the OOD splits stay leak-free. A learned reduction (e.g. PCA) is deliberately
NOT offered here: embeddings are computed once over the whole benchmark before any split exists,
so fitting a transform at that point would leak target/test rows into the source representation.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from dataio.get_input import Benchmark

RAW_MODE = os.environ.get("RAW_MODE", "flatten")  # flatten | stats


def _stack_modalities(bench) -> np.ndarray:
    """Concatenate the monthly S2, S1, climate views into one ``(N, T, C)`` array.

    The raw control isn't a learned encoder, so it just takes each modality's monthly composite
    (all native bands) and concatenates -- a cheap, fixed-length spectral-temporal featurization.
    """
    parts = []
    for modality in ("s2", "s1", "climate"):
        values = bench.monthly(modality)[0]  # (N, 12, C)
        if values.shape[-1] > 0:
            parts.append(np.asarray(values, dtype=np.float32))
    return np.concatenate(parts, axis=2)


def _featurize(x: np.ndarray, mode: str) -> np.ndarray:
    """``(N, T, C)`` -> ``(N, D)`` raw features (no learning, fully per-sample)."""
    n = x.shape[0]
    if mode == "stats":
        feats = np.concatenate(
            [x.mean(1), x.std(1), x.min(1), x.max(1), x.max(1) - x.min(1)], axis=1
        )
    else:  # flatten
        feats = x.reshape(n, -1)
    return np.nan_to_num(feats.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)


@dataclass
class RawModel:
    """Raw-feature baseline: Benchmark / tile -> ``(N, D)`` features. Not a model."""

    name: str = "raw"
    mode: str = RAW_MODE
    device: str = "cpu"  # accepted and ignored (no compute)
    embedding_dim: int = 0  # set after the first encode call

    def encode(self, bench: Benchmark) -> np.ndarray:
        feats = _featurize(_stack_modalities(bench), self.mode)
        self.embedding_dim = int(feats.shape[1])
        return feats

    def encode_dense(self, tile) -> np.ndarray:
        """Per-pixel raw features for PASTIS (same per-sample featurization)."""
        pix = tile.pixel_benchmark()  # valid pixels as a (n, T, C) Benchmark
        mode = "stats" if self.mode == "stats" else "flatten"
        feats = _featurize(_stack_modalities(pix), mode)
        self.embedding_dim = int(feats.shape[1])
        return feats

    def compute_macs(self) -> int:
        return 0  # not a model — no multiply-accumulates
