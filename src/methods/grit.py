"""GRIT (a.k.a. ECMP): spurious-subspace projection on frozen embeddings.

Adapted from inouye-lab/GRIT (``solver/ecmp.py``). The method is a closed-form
feature projection -- no model training. Given counterfactual pairs (same
label, differing only in a nuisance attribute), the top singular directions of
their differences span the nuisance subspace; we project embeddings onto its
orthogonal complement before the downstream probe::

    U = top-`rank` right singular vectors of the (K, D) pair-difference matrix
    P = I - U Uᵀ                 # projector onto the stable complement
    z' = z @ P

This is a drop-in ``FeatureTransform`` for ``evals.run_probes(transform=...)``:
fit on the training pool, then ``transform`` is applied identically to train and
test embeddings before the calibrated probe. The model is never touched.

Two flavors of "nuisance", set by ``matching``:

* geographic ("conditional" / "nearest"): pairs are same-label samples from
  *different groups* (CropHarvest dataset / EuroCropsML country). The removed
  subspace is geographic spurious structure. This mirrors the GRIT paper's
  conditional and nearest-neighbour matching, with "domain" = our ``groups``.

EO caveat: ``rank`` is the key knob -- removing too many directions deletes real
agronomic signal. Sweep it (e.g. {1,2,4,6,8}) and report transfer vs rank.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.neighbors import NearestNeighbors

from utils import perfutils as perf

MATCHINGS = ("conditional", "nearest")


def conditional_diffs(
    emb: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    max_pairs: int,
    seed: int,
) -> np.ndarray:
    """Random same-label / different-group pair differences (paper: conditional)."""
    rng = np.random.default_rng(seed)
    classes = np.unique(y)
    per_class = max(1, max_pairs // len(classes))
    diffs: list[np.ndarray] = []
    for cls in classes:
        idx = np.where(y == cls)[0]
        if len(idx) < 2 or len(np.unique(groups[idx])) < 2:
            continue
        g_idx = groups[idx]
        count, attempts = 0, 0
        while count < per_class and attempts < per_class * 20:
            attempts += 1
            i = rng.choice(idx)
            cand = idx[g_idx != groups[i]]
            if len(cand) == 0:
                continue
            j = rng.choice(cand)
            diffs.append(emb[i] - emb[j])
            count += 1
    if not diffs:
        raise ValueError("conditional matching produced no cross-group same-label pairs.")
    return np.stack(diffs).astype(np.float32)


def nearest_diffs(
    emb: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    max_pairs: int,
    seed: int,
    k: int = 10,
) -> np.ndarray:
    """Nearest cross-group same-label neighbour differences (paper: nearest).

    Uses a per-class kNN index and, for each anchor, takes the closest neighbour
    whose group differs -- a cleaner nuisance estimate than random matching, and
    scalable (no full pairwise distance matrix).
    """
    rng = np.random.default_rng(seed)
    diffs: list[np.ndarray] = []
    for cls in np.unique(y):
        idx = np.where(y == cls)[0]
        if len(idx) < 2 or len(np.unique(groups[idx])) < 2:
            continue
        x_c, g_c = emb[idx], groups[idx]
        n_neighbors = min(k + 1, len(idx))
        nbrs = NearestNeighbors(n_neighbors=n_neighbors).fit(x_c)
        _, nbr = nbrs.kneighbors(x_c)
        for li in range(len(idx)):
            for lj in nbr[li][1:]:
                if g_c[lj] != g_c[li]:
                    diffs.append(emb[idx[li]] - emb[idx[lj]])
                    break
    if not diffs:
        raise ValueError("nearest matching produced no cross-group same-label pairs.")
    diffs_arr = np.stack(diffs).astype(np.float32)
    if max_pairs and len(diffs_arr) > max_pairs:
        sel = rng.choice(len(diffs_arr), size=max_pairs, replace=False)
        diffs_arr = diffs_arr[sel]
    return diffs_arr



def projection_from_diffs(diffs: np.ndarray, rank: int) -> np.ndarray:
    """P = I - U Uᵀ where U spans the top-`rank` directions of the pair differences."""
    dim = diffs.shape[1]
    rank = max(0, min(rank, min(diffs.shape)))
    if rank == 0:
        return np.eye(dim, dtype=np.float32)
    _, _, vt = np.linalg.svd(diffs.astype(np.float64), full_matrices=False)
    q = vt[:rank].T
    return (np.eye(dim) - q @ q.T).astype(np.float32)


@dataclass
class Grit:
    """Fitted GRIT projection usable as a FeatureTransform (has ``.transform``)."""

    rank: int = 4
    matching: str = "conditional"
    standardize: bool = False
    max_pairs: int = 8192
    seed: int = 0
    name: str = field(init=False)
    _proj: np.ndarray | None = field(default=None, init=False, repr=False)
    _mean: np.ndarray | None = field(default=None, init=False, repr=False)
    _std: np.ndarray | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.matching not in MATCHINGS:
            raise ValueError(f"matching must be one of {MATCHINGS}, got {self.matching!r}")
        self.name = f"grit_{self.matching}_r{self.rank}"

    def _scale(self, x: np.ndarray) -> np.ndarray:
        if not self.standardize:
            return x
        return (x - self._mean) / self._std

    def fit(
        self,
        x: np.ndarray,
        y: np.ndarray | None = None,
        groups: np.ndarray | None = None,
    ) -> Grit:
        x = np.asarray(x, dtype=np.float32)
        if self.standardize:
            self._mean = x.mean(axis=0)
            self._std = x.std(axis=0)
            self._std[self._std == 0] = 1.0
        xb = self._scale(x)
        if y is None or groups is None:
            raise ValueError(f"matching={self.matching!r} requires y and groups.")
        builder = conditional_diffs if self.matching == "conditional" else nearest_diffs
        diffs = builder(xb, np.asarray(y), np.asarray(groups, dtype=object), self.max_pairs, self.seed)

        with perf.measure("method.fit/grit", rank=self.rank, matching=self.matching, n_pairs=len(diffs), dim=diffs.shape[1]):
            self._proj = projection_from_diffs(diffs, self.rank)
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        if self._proj is None:
            raise RuntimeError("Grit.transform called before fit().")
        return (self._scale(np.asarray(x, dtype=np.float32)) @ self._proj).astype(np.float32)


def variants(label_kind: str) -> dict[str, dict]:
    v: dict[str, dict] = {}
    if label_kind != "regression":
        for r in (1, 2, 4, 8):
            v[f"grit_conditional_r{r}"] = {"matching": "conditional", "rank": r}
        v["grit_nearest_r4"] = {"matching": "nearest", "rank": 4}
    return v
