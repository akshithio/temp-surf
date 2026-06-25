"""Strict deployment-style geographic OOD regime."""

from __future__ import annotations

import hashlib
import importlib
from typing import Any

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import BallTree

from evals.regimes.base import Split, geography_domains

NAME = "geographic_ood"
GROUP_KIND = "geography"
HAS_TARGET = True
USES_CURATED_HOLDOUTS = True
EARTH_RADIUS_KM = 6371.0088


def _bench_mod(bench):
    try:
        return importlib.import_module(f"evals.benchmarks.{bench.name}")
    except (ImportError, AttributeError):
        return None


def assign_domains(bench) -> np.ndarray:
    mod = _bench_mod(bench)
    fn = getattr(mod, "geographic_domains", None) if mod is not None else None
    if fn is not None:
        return np.asarray(fn(bench), dtype=object)
    return geography_domains(bench)


def _valid_domains(groups: np.ndarray) -> list[str]:
    return sorted({str(g) for g in np.asarray(groups).astype(str) if str(g) not in ("unknown", "nan")})


def _idx_for(groups: np.ndarray, domains: set[str]) -> np.ndarray:
    groups_s = np.asarray(groups).astype(str)
    return np.flatnonzero(np.isin(groups_s, list(domains)))


def _check_split(y: np.ndarray, train: np.ndarray, val: np.ndarray, test: np.ndarray, label: str) -> None:
    if len(train) == 0 or len(val) == 0 or len(test) == 0:
        raise ValueError(f"{label}: empty train/val/test partition")
    if len(np.unique(y[train])) < 2:
        raise ValueError(f"{label}: training partition is one-class")
    if len(np.unique(y[val])) < 2:
        raise ValueError(f"{label}: validation partition is one-class")
    if len(np.unique(y[test])) < 2:
        raise ValueError(f"{label}: test partition is one-class")


def _domain_size(groups: np.ndarray, domains: set[str]) -> int:
    return int(np.isin(np.asarray(groups).astype(str), list(domains)).sum())


def _domain_classes(y: np.ndarray, groups: np.ndarray, domains: set[str]) -> int:
    idx = _idx_for(groups, domains)
    return int(len(np.unique(y[idx]))) if len(idx) else 0


def _spatial_partitions(y: np.ndarray, groups: np.ndarray, *, val_frac: float, test_frac: float):
    domains = _valid_domains(groups)
    if len(domains) < 3:
        raise ValueError("spatial-block split needs at least three valid blocks")
    ranked = sorted(
        domains,
        key=lambda d: hashlib.sha256(d.encode()).hexdigest(),
    )

    valid_n = int(np.isin(np.asarray(groups).astype(str), domains).sum())
    test_target = max(1, int(round(test_frac * valid_n)))
    val_target = max(1, int(round(val_frac * valid_n)))

    def pick(used: set[str], target_n: int) -> set[str]:
        picked: set[str] = set()
        for domain in ranked:
            if domain in used:
                continue
            picked.add(domain)
            if _domain_size(groups, picked) >= target_n and _domain_classes(y, groups, picked) >= 2:
                return picked
        raise ValueError("spatial-block split cannot build a two-class validation/test partition")

    test = pick(set(), test_target)
    val = pick(test, val_target)
    train = set(domains) - val - test
    return train, val, test


def _fixed_partitions(groups: np.ndarray, spec: dict[str, Any]):
    train = {str(v) for v in spec["train"]}
    val = {str(v) for v in spec["val"]}
    test = {str(v) for v in spec["test"]}
    missing = (train | val | test) - set(_valid_domains(groups))
    if missing:
        raise ValueError(f"fixed geographic split references absent domain(s): {sorted(missing)}")
    if train & val or train & test or val & test:
        raise ValueError("fixed geographic split train/val/test domains must be disjoint")
    return train, val, test


def _purge_train_near_ood(train: np.ndarray, val: np.ndarray, test: np.ndarray, bench, radius_km: float) -> np.ndarray:
    if radius_km <= 0 or bench is None or not hasattr(bench, "latlon"):
        return train
    latlon = np.asarray(bench.latlon, dtype=float)
    if latlon.ndim != 2 or latlon.shape[1] != 2:
        return train
    ref = np.concatenate([val, test])
    ref_valid = ref[np.isfinite(latlon[ref]).all(axis=1)]
    train_valid_mask = np.isfinite(latlon[train]).all(axis=1)
    if len(ref_valid) == 0 or not train_valid_mask.any():
        return train
    tree = BallTree(np.deg2rad(latlon[ref_valid]), metric="haversine")
    dist = tree.query(np.deg2rad(latlon[train[train_valid_mask]]), k=1, return_distance=True)[0].ravel()
    keep_valid = dist > (radius_km / EARTH_RADIUS_KM)
    keep = np.ones(len(train), dtype=bool)
    keep[np.flatnonzero(train_valid_mask)] = keep_valid
    return np.sort(train[keep])


def _split_from_domain_sets(y, groups, train_domains, val_domains, test_domains, *, label, bench, purge_km):
    train = _idx_for(groups, train_domains)
    val = _idx_for(groups, val_domains)
    test = _idx_for(groups, test_domains)
    train = _purge_train_near_ood(train, val, test, bench, purge_km)
    _check_split(y, train, val, test, label)
    return Split(label, np.sort(train), np.sort(test), np.sort(val))


def make_strict_holdout_splits(
    y: np.ndarray,
    groups: np.ndarray,
    heldout_group: str,
    seed: int,
    val_group: str | None = None,
    require_domain_val: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    idx = np.arange(len(y))
    groups_s = np.asarray(groups).astype(str)
    heldout_group = str(heldout_group)
    test = idx[groups_s == heldout_group]
    if len(test) == 0:
        raise ValueError(f"No samples found for strict holdout group: {heldout_group}")
    if len(np.unique(y[test])) < 2:
        raise ValueError(f"Strict holdout group is one-class: {heldout_group}")
    if val_group is not None and np.any(groups_s == str(val_group)) and str(val_group) != heldout_group:
        val_group = str(val_group)
        val = idx[groups_s == val_group]
        train = idx[(groups_s != heldout_group) & (groups_s != val_group)]
        train_val = idx[groups_s != heldout_group]
        _check_split(y, train, val, test, f"{val_group}->{heldout_group}")
        return np.sort(train), np.sort(val), np.sort(test), np.sort(train_val)
    if require_domain_val:
        domains = _valid_domains(groups_s)
        start = domains.index(heldout_group)
        candidates = domains[start + 1:] + domains[:start]
        for candidate in candidates:
            val = idx[groups_s == candidate]
            train = idx[(groups_s != heldout_group) & (groups_s != candidate)]
            try:
                _check_split(y, train, val, test, f"{candidate}->{heldout_group}")
            except ValueError:
                continue
            train_val = idx[groups_s != heldout_group]
            return np.sort(train), np.sort(val), np.sort(test), np.sort(train_val)
        raise ValueError(f"No valid whole-domain validation group found for holdout: {heldout_group}")
    train_val = idx[groups_s != heldout_group]
    if len(np.unique(y[train_val])) < 2:
        raise ValueError(f"Strict holdout training pool is one-class after excluding: {heldout_group}")
    try:
        train, val = train_test_split(train_val, test_size=0.10, random_state=seed, stratify=y[train_val])
    except ValueError:
        train, val = train_test_split(train_val, test_size=0.10, random_state=seed, stratify=None)
    return np.sort(train), np.sort(val), np.sort(test), np.sort(train_val)


def iter_splits(y, groups, *, seed, holdouts, n_folds=None, val_group=None, bench=None, **_):
    del n_folds, val_group
    mod = _bench_mod(bench) if bench is not None else None
    purge_km = float(getattr(mod, "GEOGRAPHIC_PURGE_KM", 0.0)) if mod is not None else 0.0
    try:
        if isinstance(holdouts, dict) and holdouts.get("strategy") == "spatial_blocks":
            purge_km = float(holdouts.get("purge_km", purge_km))
            train_d, val_d, test_d = _spatial_partitions(
                y,
                groups,
                val_frac=float(holdouts.get("val_fraction", 0.10)),
                test_frac=float(holdouts.get("test_fraction", 0.20)),
            )
            yield _split_from_domain_sets(
                y,
                groups,
                train_d,
                val_d,
                test_d,
                label=str(holdouts.get("label", "spatial_blocks")),
                bench=bench,
                purge_km=purge_km,
            )
            return
        if isinstance(holdouts, dict):
            purge_km = float(holdouts.get("purge_km", purge_km))
            train_d, val_d, test_d = _fixed_partitions(groups, holdouts)
            yield _split_from_domain_sets(
                y,
                groups,
                train_d,
                val_d,
                test_d,
                label=str(holdouts.get("label", "fixed_geography")),
                bench=bench,
                purge_km=purge_km,
            )
            return
    except ValueError as exc:
        print(f"   !! geographic_ood: split dropped ({exc})", flush=True)
        return
    for holdout in holdouts:
        try:
            train, val, test, _train_val = make_strict_holdout_splits(
                y, groups, holdout, seed, require_domain_val=True
            )
            train = _purge_train_near_ood(train, val, test, bench, purge_km)
            _check_split(y, train, val, test, str(holdout))
        except ValueError as exc:
            print(f"   !! geographic_ood: holdout {holdout!r} dropped ({exc})", flush=True)
            continue
        yield Split(str(holdout), train, test, val)


def iter_fold_splits(bench_mod):
    split = getattr(bench_mod, "GEOGRAPHIC_FOLD_SPLIT", None)
    if split is not None:
        yield (
            str(split.get("label", "fixed_folds")),
            set(split["train"]),
            set(split["val"]),
            set(split["test"]),
        )
        return
    all_folds = sorted(set(bench_mod.TRAIN_FOLDS) | set(bench_mod.VAL_FOLDS) | set(bench_mod.TEST_FOLDS))
    for i, test_fold in enumerate(all_folds):
        val_fold = all_folds[(i + 1) % len(all_folds)]
        train_folds = {f for f in all_folds if f not in (test_fold, val_fold)}
        yield (f"fold_{test_fold}", train_folds, {val_fold}, {test_fold})
