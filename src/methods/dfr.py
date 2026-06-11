"""DFR: retrain the final probe on a balanced frozen-feature subset."""

from __future__ import annotations
import numpy as np


class Dfr:
    """Deep feature reweighting via balanced last-layer retraining."""

    def __init__(self, seed: int = 0):
        self.seed = seed

    def fit(self, x, y=None, groups=None, x_paired=None):
        return self

    def transform(self, x):
        return x

    def subset_indices(self, y, groups, budget, seed):
        y = np.asarray(y)
        groups = np.zeros(len(y), dtype=np.int64) if groups is None else np.asarray(groups)
        target = len(y) if budget >= 1 else max(1, int(round(float(budget) * len(y))))
        rng = np.random.default_rng(seed)
        strata = []
        for cls in np.unique(y):
            cls_mask = y == cls
            for group in np.unique(groups[cls_mask]):
                idx = np.flatnonzero(cls_mask & (groups == group))
                if len(idx):
                    rng.shuffle(idx)
                    strata.append(idx)
        if not strata:
            return np.arange(len(y), dtype=np.int64)
        per_stratum = max(1, target // len(strata))
        per_stratum = min(per_stratum, min(len(idx) for idx in strata))
        selected = np.concatenate([idx[:per_stratum] for idx in strata])
        if len(selected) < min(target, len(y)):
            keep = np.ones(len(y), dtype=bool)
            keep[selected] = False
            rest = np.flatnonzero(keep)
            rng.shuffle(rest)
            selected = np.concatenate([selected, rest[: min(target, len(y)) - len(selected)]])
        rng.shuffle(selected)
        return np.sort(selected.astype(np.int64))


def variants(task_kind: str) -> dict[str, dict]:
    return {} if task_kind == "regression" else {"dfr": {}}
