"""Coordinate-cluster geographic OOD regime."""

from __future__ import annotations

import importlib
from types import SimpleNamespace
from typing import Any

import numpy as np
from sklearn.cluster import KMeans

from evals.regimes.base import DenseSplit, Split
from evals.regimes.geographic_ood import (
    _check_split,
    _idx_for,
    _purge_train_near_ood,
    _source_diag_indices,
    _source_diag_patches,
)
from utils import cacheutils

NAME = "spatial_cluster_ood"
GROUP_KIND = "spatial_cluster"
HAS_TARGET = True

DEFAULT_SPLIT = {
    "label": "spatial_clusters",
    "n_clusters": 12,
    "val_fraction": 0.10,
    "test_fraction": 0.20,
    "purge_km": 25.0,
}


def _bench_mod(bench):
    try:
        return importlib.import_module(f"evals.benchmarks.{bench.name}")
    except (ImportError, AttributeError):
        return None


def _spec(bench) -> dict[str, Any]:
    mod = _bench_mod(bench)
    spec = dict(DEFAULT_SPLIT)
    if mod is not None:
        spec.update(getattr(mod, "SPATIAL_CLUSTER_SPLIT", {}))
    return spec


def _valid_latlon(bench) -> tuple[np.ndarray, np.ndarray]:
    if not hasattr(bench, "latlon"):
        raise ValueError("spatial_cluster_ood needs bench.latlon")
    latlon = np.asarray(bench.latlon, dtype=float)
    if latlon.ndim != 2 or latlon.shape[1] != 2:
        raise ValueError("spatial_cluster_ood needs latlon shaped (n, 2)")
    valid = np.isfinite(latlon).all(axis=1)
    if valid.sum() < 3:
        raise ValueError("spatial_cluster_ood needs at least three located samples")
    distinct = np.unique(np.round(latlon[valid], 6), axis=0)
    if len(distinct) < 3:
        raise ValueError("spatial_cluster_ood needs at least three distinct coordinates")
    return latlon, valid


def _sphere_features(latlon: np.ndarray) -> np.ndarray:
    lat = np.deg2rad(latlon[:, 0])
    lon = np.deg2rad(latlon[:, 1])
    return np.column_stack([
        np.cos(lat) * np.cos(lon),
        np.cos(lat) * np.sin(lon),
        np.sin(lat),
    ])


def assign_domains(bench, holdouts: Any = None) -> np.ndarray:
    """Coordinate clusters. This regime has a single basis, so the strategy is not consulted."""
    del holdouts
    latlon, valid = _valid_latlon(bench)
    spec = _spec(bench)
    n_clusters = min(int(spec.get("n_clusters", DEFAULT_SPLIT["n_clusters"])), int(valid.sum()))
    n_clusters = min(n_clusters, len(np.unique(np.round(latlon[valid], 6), axis=0)))
    if n_clusters < 3:
        raise ValueError("spatial_cluster_ood needs at least three coordinate clusters")

    labels = KMeans(n_clusters=n_clusters, random_state=0, n_init=10).fit_predict(_sphere_features(latlon[valid]))
    remap: dict[int, int] = {}
    for new_id, old_id in enumerate(sorted(set(labels), key=lambda c: tuple(np.nanmean(latlon[valid][labels == c], axis=0)))):
        remap[int(old_id)] = int(new_id)

    out = np.full(len(latlon), "unknown", dtype=object)
    out[np.flatnonzero(valid)] = [f"cluster_{remap[int(c)]:02d}" for c in labels]
    return out


def _domain_sizes(groups: np.ndarray) -> dict[str, int]:
    return {str(g): int((groups == g).sum()) for g in sorted(set(groups.astype(str))) if str(g) != "unknown"}


def _domain_scores(bench, groups: np.ndarray) -> dict[str, float]:
    latlon, valid = _valid_latlon(bench)
    domains = sorted({str(g) for g in groups[valid].astype(str) if str(g) != "unknown"})
    centers = np.asarray([np.nanmean(latlon[(groups.astype(str) == d) & valid], axis=0) for d in domains])
    centered = centers - centers.mean(axis=0, keepdims=True)
    if len(domains) < 2 or np.allclose(centered, 0):
        return {d: float(i) for i, d in enumerate(domains)}
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    scores = centered @ vh[0]
    return {d: float(s) for d, s in zip(domains, scores, strict=True)}


def _pick_extreme(
    ordered: list[str],
    used: set[str],
    sizes: dict[str, int],
    y: np.ndarray,
    groups: np.ndarray,
    target_n: int,
) -> set[str]:
    picked: set[str] = set()
    for domain in ordered:
        if domain in used:
            continue
        picked.add(domain)
        idx = _idx_for(groups, picked)
        if sum(sizes[d] for d in picked) >= target_n and len(np.unique(y[idx])) >= 2:
            return picked
    raise ValueError("spatial_cluster_ood cannot build a two-class validation/test partition")


def _pick_extreme_patches(
    ordered: list[str],
    used: set[str],
    sizes: dict[str, int],
    classes: dict[str, set[int]],
    target_n: int,
) -> set[str]:
    picked: set[str] = set()
    for domain in ordered:
        if domain in used:
            continue
        picked.add(domain)
        cls = set().union(*(classes[d] for d in picked))
        if sum(sizes[d] for d in picked) >= target_n and len(cls) >= 2:
            return picked
    raise ValueError("spatial_cluster_ood cannot build a two-class validation/test partition")


def iter_splits(y, groups, *, seed, holdouts=None, n_folds=None, val_group=None, bench=None, **_):
    del holdouts, n_folds, val_group
    try:
        if bench is None:
            raise ValueError("spatial_cluster_ood needs the benchmark object")
        spec = _spec(bench)
        groups = np.asarray(groups, dtype=object)
        sizes = _domain_sizes(groups)
        if len(sizes) < 3:
            raise ValueError("spatial_cluster_ood needs at least three clusters")

        n_valid = sum(sizes.values())
        test_n = max(1, int(round(float(spec.get("test_fraction", 0.20)) * n_valid)))
        val_n = max(1, int(round(float(spec.get("val_fraction", 0.10)) * n_valid)))
        scores = _domain_scores(bench, groups)
        low_to_high = sorted(sizes, key=lambda d: (scores[d], d))
        high_to_low = list(reversed(low_to_high))

        test_domains = _pick_extreme(high_to_low, set(), sizes, y, groups, test_n)
        val_domains = _pick_extreme(low_to_high, test_domains, sizes, y, groups, val_n)
        train_domains = set(sizes) - test_domains - val_domains

        train = _idx_for(groups, train_domains)
        val = _idx_for(groups, val_domains)
        test = _idx_for(groups, test_domains)
        train = _purge_train_near_ood(train, val, test, bench, float(spec.get("purge_km", 25.0)))
        label = str(spec.get("label", "spatial_clusters"))
        _check_split(y, train, val, test, label)
        train, source_val, source_test = _source_diag_indices(y, train, seed)
    except ValueError as exc:
        print(f"   !! spatial_cluster_ood: split dropped ({exc})", flush=True)
        return
    yield Split(label, np.sort(train), np.sort(test), np.sort(val), source_val, source_test)


def _patch_class_sets(emb_dir, folds: set[int], patch_ids: np.ndarray) -> dict[int, set[int]]:
    out = {int(pid): set() for pid in patch_ids}
    for path in cacheutils._dense_label_paths(emb_dir, folds, set(out)):
        pid = int(path.name.split("_", 1)[0])
        out[pid].update(int(v) for v in np.unique(np.load(path, mmap_mode="r")))
    return out


def iter_dense_splits(bench_mod, *, emb_dir, seed, bench=None):
    try:
        all_folds = sorted(set(bench_mod.TRAIN_FOLDS) | set(bench_mod.VAL_FOLDS) | set(bench_mod.TEST_FOLDS))
        available = set(cacheutils.dense_fold_patches(emb_dir, set(all_folds)))
        if bench is None or not hasattr(bench, "patch_latlon"):
            raise ValueError("PASTIS spatial_cluster_ood needs patch coordinates")
        patch_latlon = {
            int(pid): ll
            for pid, ll in bench.patch_latlon.items()
            if int(pid) in available and np.isfinite(np.asarray(ll, dtype=float)).all()
        }
        if len(patch_latlon) < 3:
            raise ValueError("PASTIS spatial_cluster_ood needs at least three located cached patches")

        patch_ids = np.asarray(sorted(patch_latlon), dtype=np.int64)
        pseudo = SimpleNamespace(
            name=getattr(bench, "name", getattr(bench_mod, "BENCHMARK", "pastis")),
            latlon=np.asarray([patch_latlon[int(pid)] for pid in patch_ids], dtype=float),
        )
        groups = assign_domains(pseudo)
        sizes = _domain_sizes(groups)
        if len(sizes) < 3:
            raise ValueError("PASTIS spatial_cluster_ood needs at least three clusters")

        patch_classes = _patch_class_sets(emb_dir, set(all_folds), patch_ids)
        domain_classes = {
            d: set().union(*(patch_classes[int(pid)] for pid in patch_ids[groups.astype(str) == d]))
            for d in sizes
        }
        spec = _spec(pseudo)
        n_valid = sum(sizes.values())
        test_n = max(1, int(round(float(spec.get("test_fraction", 0.20)) * n_valid)))
        val_n = max(1, int(round(float(spec.get("val_fraction", 0.10)) * n_valid)))
        scores = _domain_scores(pseudo, groups)
        low_to_high = sorted(sizes, key=lambda d: (scores[d], d))
        high_to_low = list(reversed(low_to_high))

        test_domains = _pick_extreme_patches(high_to_low, set(), sizes, domain_classes, test_n)
        val_domains = _pick_extreme_patches(low_to_high, test_domains, sizes, domain_classes, val_n)
        train_domains = set(sizes) - test_domains - val_domains
        train = _purge_train_near_ood(
            _idx_for(groups, train_domains),
            _idx_for(groups, val_domains),
            _idx_for(groups, test_domains),
            pseudo,
            float(spec.get("purge_km", 25.0)),
        )
        train_patches = {int(pid) for pid in patch_ids[train]}
        train_patches, source_val_patches, source_test_patches = _source_diag_patches(train_patches, seed)
        val_patches = {int(pid) for pid in patch_ids[_idx_for(groups, val_domains)]}
        test_patches = {int(pid) for pid in patch_ids[_idx_for(groups, test_domains)]}
        for name, pids in {"train": train_patches, "val": val_patches, "test": test_patches}.items():
            if not pids or len(set().union(*(patch_classes[p] for p in pids))) < 2:
                raise ValueError(f"PASTIS spatial_cluster_ood has invalid {name} patch partition")
    except ValueError as exc:
        print(f"   !! spatial_cluster_ood: dense split dropped ({exc})", flush=True)
        return
    yield DenseSplit(
        str(spec.get("label", "spatial_clusters")),
        set(all_folds),
        set(all_folds),
        set(all_folds),
        train_patches=train_patches,
        val_patches=val_patches,
        test_patches=test_patches,
        source_val_patches=source_val_patches,
        source_test_patches=source_test_patches,
        has_target=HAS_TARGET,
        group_kind=GROUP_KIND,
    )
