"""Raw-feature baseline (the reality-check control, not a foundation model).

This is the required "do the FMs actually beat cheap spectral-temporal cues?" baseline.
It is NOT a learned encoder: it turns each sample's multimodal time series directly into
a feature vector and feeds it to the same probe / regime / budget machinery as every FM,
so a fair comparison falls out for free. Especially important on LEM-Brazil, where raw
red-edge / NIR / NDVI already separates the classes — if an FM cannot beat this, the
"representation" is not buying anything there.

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
    """Concatenate S2, S1, climate time series into one ``(N, T, C)`` array."""
    parts = [np.asarray(a, dtype=np.float32) for a in (bench.s2, bench.s1, bench.climate) if a.shape[-1] > 0]
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
