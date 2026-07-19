"""Strict deployment-style geographic OOD regime."""

from __future__ import annotations

import hashlib
import importlib
from typing import Any

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import BallTree

from evals.regimes.base import DOMAIN_CENSUS, DenseSplit, Split, emit_split_audit_event, geography_domains
from utils import cacheutils

NAME = "geographic_ood"
GROUP_KIND = "geography"
HAS_TARGET = True
USES_CURATED_HOLDOUTS = True
LEAVE_ONE_DOMAIN_OUT = True
EARTH_RADIUS_KM = 6371.0088


def _bench_mod(bench):
    try:
        return importlib.import_module(f"evals.benchmarks.{bench.name}")
    except (ImportError, AttributeError):
        return None


def assign_domains(bench, holdouts: Any = None) -> np.ndarray:
    """Domain basis for this regime, SELECTED BY THE SPLIT STRATEGY.

    The basis and the strategy are not independent. ``spatial_blocks`` assembles val/test by
    hash-ranking domains until a sample-count target is met, which only reconstructs the
    historical ``spatial_block_2deg_purge50km`` partition when the domains are 2-degree blocks;
    run it over canonical provenance domains and it silently emits the historical label over a
    different split. Leave-one-domain-out needs the opposite -- the named canonical domains.
    Hence the strategy has to be known BEFORE domains are assigned, not after.
    """
    mod = _bench_mod(bench)
    if isinstance(holdouts, dict) and holdouts.get("strategy") == "spatial_blocks":
        fn = getattr(mod, "spatial_block_domains", None) if mod is not None else None
        if fn is None:
            raise ValueError(
                f"{getattr(bench, 'name', '?')}: the spatial_blocks strategy requires a "
                f"spatial_block_domains() basis; refusing to fall back to another basis, which "
                f"would emit a historical split label over a different partition"
            )
        return np.asarray(fn(bench), dtype=object)
    fn = getattr(mod, "geographic_domains", None) if mod is not None else None
    if fn is not None:
        return np.asarray(fn(bench), dtype=object)
    return geography_domains(bench)


def _valid_domains(groups: np.ndarray) -> list[str]:
    return sorted({str(g) for g in np.asarray(groups).astype(str) if str(g) not in ("unknown", "nan")})


def _idx_for(groups: np.ndarray, domains: set[str]) -> np.ndarray:
    groups_s = np.asarray(groups).astype(str)
    return np.flatnonzero(np.isin(groups_s, list(domains)))


def _check_split(
    y: np.ndarray,
    train: np.ndarray,
    val: np.ndarray,
    test: np.ndarray,
    label: str,
    *,
    allow_one_class_test: bool = False,
) -> None:
    """Reject partitions that cannot support probe training, threshold calibration, or scoring.

    ``allow_one_class_test`` keeps a one-class TARGET: several canonical domains genuinely
    contain a single class, and dropping them would silently shrink the evaluated universe. Only
    train and val are structurally required to be two-class -- train to fit the probe at all, val
    to calibrate a decision threshold. A one-class target still scores accuracy, balanced
    accuracy, calibration and test_pos_rate; the metrics that need both classes (auc) already
    come back as nan from score_binary.
    """
    if len(train) == 0 or len(val) == 0 or len(test) == 0:
        raise ValueError(f"{label}: empty train/val/test partition")
    if len(np.unique(y[train])) < 2:
        raise ValueError(f"{label}: training partition is one-class")
    if len(np.unique(y[val])) < 2:
        raise ValueError(f"{label}: validation partition is one-class")
    if not allow_one_class_test and len(np.unique(y[test])) < 2:
        raise ValueError(f"{label}: test partition is one-class")


def _domain_size(groups: np.ndarray, domains: set[str]) -> int:
    return int(np.isin(np.asarray(groups).astype(str), list(domains)).sum())


def _domain_classes(y: np.ndarray, groups: np.ndarray, domains: set[str]) -> int:
    idx = _idx_for(groups, domains)
    return int(len(np.unique(y[idx]))) if len(idx) else 0


def _is_lodo(holdouts: Any) -> bool:
    return isinstance(holdouts, dict) and holdouts.get("strategy") == "leave_one_domain_out"


def domain_census(y: np.ndarray, groups: np.ndarray, holdouts: Any) -> list[dict[str, Any]]:
    """Eligibility census over every canonical domain present in the data.

    Computed BEFORE any model runs, so validity is a declared property of the dataset rather than
    a side effect of whichever folds happened to survive. One row per domain, including the ones
    that are excluded and why -- an excluded domain must be visible in the artifact, never a
    silent gap in the results table.
    """
    if not _is_lodo(holdouts):
        return []
    min_n = int(holdouts.get("min_target_n", 10))
    allow_one_class = bool(holdouts.get("allow_one_class_target", True))
    rows: list[dict[str, Any]] = []
    for domain in _valid_domains(groups):
        n = _domain_size(groups, {domain})
        n_classes = _domain_classes(y, groups, {domain})
        excluded: list[str] = []
        if n < min_n:
            excluded.append(f"n<{min_n}")
        if n_classes < 2 and not allow_one_class:
            excluded.append("one_class")
        rows.append({
            "domain": domain,
            "n": n,
            "n_classes": n_classes,
            "one_class": n_classes < 2,
            "valid_target": not excluded,
            # a one-class domain cannot calibrate a decision threshold, so it cannot be the
            # validation region even when it is a perfectly valid target
            "valid_val": n_classes >= 2 and n >= min_n,
            "excluded_because": excluded,
            "metrics_excluded": ["auc"] if n_classes < 2 else [],
        })
    return rows


def expected_domains(y: np.ndarray, groups: np.ndarray, holdouts: Any) -> list[str] | None:
    """Domains this regime DECLARES it will produce a fold for (None => check not applicable).

    Compared against what actually got yielded, so a declared-valid domain that vanishes is a
    hard failure rather than a missing row someone notices later.
    """
    if not _is_lodo(holdouts):
        return None
    return [r["domain"] for r in domain_census(y, groups, holdouts) if r["valid_target"]]


def _source_diag_indices(y: np.ndarray, train: np.ndarray, seed: int):
    train = np.asarray(train, dtype=np.int64)
    if len(train) < 10 or len(np.unique(y[train])) < 2:
        return np.sort(train), np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    try:
        train_val, source_test = train_test_split(train, test_size=0.10, random_state=seed + 101, stratify=y[train])
        source_train, source_val = train_test_split(
            train_val, test_size=0.1111111111, random_state=seed + 102, stratify=y[train_val]
        )
    except ValueError:
        emit_split_audit_event("stratification_fallback", where="_source_diag_indices", stage="source_diag")
        train_val, source_test = train_test_split(train, test_size=0.10, random_state=seed + 101)
        source_train, source_val = train_test_split(train_val, test_size=0.1111111111, random_state=seed + 102)
    if len(source_train) == 0 or len(np.unique(y[source_train])) < 2:
        return np.sort(train), np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    return np.sort(source_train), np.sort(source_val), np.sort(source_test)


def _source_diag_patches(patches: list[int] | np.ndarray, seed: int, patch_classes: dict | None = None):
    patches = np.asarray(sorted(map(int, patches)), dtype=np.int64)
    if len(patches) < 10:
        return set(map(int, patches)), None, None
    train_val, source_test = train_test_split(patches, test_size=0.10, random_state=seed + 101)
    source_train, source_val = train_test_split(train_val, test_size=0.1111111111, random_state=seed + 102)
    if patch_classes is not None:
        if not len(source_train) or len(set().union(*(patch_classes.get(p, set()) for p in source_train))) < 2:
            return set(map(int, patches)), None, None
    return set(map(int, source_train)), set(map(int, source_val)), set(map(int, source_test))


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
    purged = np.asarray(train)[~keep]
    if len(purged):
        # Behavior-neutral: record exactly which training rows the (unchanged) purge removed and
        # why, so preprocessing can attach a PROVEN exclusion reason instead of guessing.
        emit_split_audit_event(
            "purge",
            where="_purge_train_near_ood",
            reference="val_test",
            radius_km=float(radius_km),
            n_purged=int(len(purged)),
            purged_train_indices=[int(i) for i in purged.tolist()],
        )
    return np.sort(train[keep])


def _split_from_domain_sets(y, groups, train_domains, val_domains, test_domains, *, label, bench, purge_km, seed):
    train = _idx_for(groups, train_domains)
    val = _idx_for(groups, val_domains)
    test = _idx_for(groups, test_domains)
    train = _purge_train_near_ood(train, val, test, bench, purge_km)
    _check_split(y, train, val, test, label)
    train, source_val, source_test = _source_diag_indices(y, train, seed)
    return Split(label, train, np.sort(test), np.sort(val), source_val, source_test)


def make_strict_holdout_splits(
    y: np.ndarray,
    groups: np.ndarray,
    heldout_group: str,
    seed: int,
    val_group: str | None = None,
    require_domain_val: bool = False,
    *,
    allow_one_class_test: bool = False,
    val_pool: list[str] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    idx = np.arange(len(y))
    groups_s = np.asarray(groups).astype(str)
    heldout_group = str(heldout_group)
    test = idx[groups_s == heldout_group]
    if len(test) == 0:
        raise ValueError(f"No samples found for strict holdout group: {heldout_group}")
    if not allow_one_class_test and len(np.unique(y[test])) < 2:
        raise ValueError(f"Strict holdout group is one-class: {heldout_group}")
    if val_group is not None and np.any(groups_s == str(val_group)) and str(val_group) != heldout_group:
        val_group = str(val_group)
        val = idx[groups_s == val_group]
        train = idx[(groups_s != heldout_group) & (groups_s != val_group)]
        train_val = idx[groups_s != heldout_group]
        _check_split(
            y, train, val, test, f"{val_group}->{heldout_group}", allow_one_class_test=allow_one_class_test
        )
        return np.sort(train), np.sort(val), np.sort(test), np.sort(train_val)
    if require_domain_val:
        domains = _valid_domains(groups_s)
        start = domains.index(heldout_group)
        # Deterministic alphabetical rotation: the validation region is a property of the split,
        # not of the seed, so every seed and model sees the identical fold.
        rotated = domains[start + 1:] + domains[:start]
        allowed = set(val_pool) if val_pool is not None else set(domains)
        candidates = [d for d in rotated if d in allowed and d != heldout_group]
        for candidate in candidates:
            val = idx[groups_s == candidate]
            train = idx[(groups_s != heldout_group) & (groups_s != candidate)]
            try:
                _check_split(
                    y, train, val, test, f"{candidate}->{heldout_group}",
                    allow_one_class_test=allow_one_class_test,
                )
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
        emit_split_audit_event(
            "stratification_fallback", where="make_strict_holdout_splits", stage="train_vs_val",
            holdout=str(heldout_group),
        )
        train, val = train_test_split(train_val, test_size=0.10, random_state=seed, stratify=None)
    return np.sort(train), np.sort(val), np.sort(test), np.sort(train_val)


def _iter_lodo_splits(y, groups, *, seed, holdouts, bench, purge_km):
    """Leave-one-domain-out over the full canonical domain census.

    One fold per declared-valid domain: that domain is the target, one different valid whole
    domain is the validation region, and every remaining domain is the source training pool.
    Each fold is isolated -- a fold that cannot be built is reported and skipped here, and the
    completeness check in regimes.base then fails the run because the domain was declared valid.
    """
    purge_km = float(holdouts.get("purge_km", purge_km))
    allow_one_class = bool(holdouts.get("allow_one_class_target", True))
    census = domain_census(y, groups, holdouts)
    bench_name = str(getattr(bench, "name", "?"))
    seen = {(r["benchmark"], r["domain"]) for r in DOMAIN_CENSUS if r["regime"] == NAME}
    for row in census:
        if (bench_name, row["domain"]) not in seen:
            DOMAIN_CENSUS.append({"benchmark": bench_name, "regime": NAME, **row})

    val_pool = [r["domain"] for r in census if r["valid_val"]]
    for row in census:
        domain = row["domain"]
        if not row["valid_target"]:
            emit_split_audit_event(
                "dropped_holdout", regime=NAME, holdout=str(domain), reason="ineligible_target",
                excluded_because=list(row["excluded_because"]), n=int(row["n"]), n_classes=int(row["n_classes"]),
            )
            print(
                f"   !! geographic_ood: domain {domain!r} (n={row['n']}, "
                f"{row['n_classes']} class(es)) is not an eligible target: "
                f"{row['excluded_because']}",
                flush=True,
            )
            continue
        try:
            train, val, test, _train_val = make_strict_holdout_splits(
                y, groups, domain, seed,
                require_domain_val=True,
                allow_one_class_test=allow_one_class,
                val_pool=val_pool,
            )
            train = _purge_train_near_ood(train, val, test, bench, purge_km)
            _check_split(y, train, val, test, str(domain), allow_one_class_test=allow_one_class)
            train, source_val, source_test = _source_diag_indices(y, train, seed)
        except ValueError as exc:
            emit_split_audit_event(
                "dropped_holdout", regime=NAME, holdout=str(domain), reason=str(exc), stage="lodo_fold",
            )
            print(f"   !! geographic_ood: LODO fold {domain!r} dropped ({exc})", flush=True)
            continue
        yield Split(str(domain), train, np.sort(test), np.sort(val), source_val, source_test, domain=str(domain))


def iter_splits(y, groups, *, seed, holdouts, n_folds=None, val_group=None, bench=None, **_):
    del n_folds, val_group
    mod = _bench_mod(bench) if bench is not None else None
    purge_km = float(getattr(mod, "GEOGRAPHIC_PURGE_KM", 0.0)) if mod is not None else 0.0
    if _is_lodo(holdouts):
        yield from _iter_lodo_splits(y, groups, seed=seed, holdouts=holdouts, bench=bench, purge_km=purge_km)
        return
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
                seed=seed,
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
                seed=seed,
            )
            return
    except ValueError as exc:
        emit_split_audit_event("dropped_split", regime=NAME, reason=str(exc), stage="strategy_split")
        print(f"   !! geographic_ood: split dropped ({exc})", flush=True)
        return
    for holdout in holdouts:
        try:
            train, val, test, _train_val = make_strict_holdout_splits(
                y, groups, holdout, seed, require_domain_val=True
            )
            train = _purge_train_near_ood(train, val, test, bench, purge_km)
            _check_split(y, train, val, test, str(holdout))
            train, source_val, source_test = _source_diag_indices(y, train, seed)
        except ValueError as exc:
            emit_split_audit_event(
                "dropped_holdout", regime=NAME, holdout=str(holdout), reason=str(exc), stage="curated_holdout",
            )
            print(f"   !! geographic_ood: holdout {holdout!r} dropped ({exc})", flush=True)
            continue
        yield Split(str(holdout), train, test, val, source_val, source_test)


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


def iter_dense_splits(bench_mod, *, emb_dir, seed, bench=None):
    for label, train_folds, val_folds, test_folds in iter_fold_splits(bench_mod):
        patch_classes = getattr(bench, "patch_classes", None) if bench is not None else None
        # Cache-free seam: emb_dir=None sources the patch universe from the benchmark descriptor.
        # patch_classes stays as-is (currently None) so behavior is unchanged either way.
        if emb_dir is None:
            if bench is None:
                raise ValueError("cache-free dense split needs the benchmark object (bench=...)")
            fold_patches = bench.patch_ids(set(train_folds))
        else:
            fold_patches = cacheutils.dense_fold_patches(emb_dir, set(train_folds))
        train_patches, source_val, source_test = _source_diag_patches(
            fold_patches,
            seed,
            patch_classes=patch_classes,
        )
        yield DenseSplit(
            str(label),
            set(train_folds),
            set(val_folds),
            set(test_folds),
            train_patches=train_patches,
            source_val_patches=source_val,
            source_test_patches=source_test,
            has_target=HAS_TARGET,
            group_kind=GROUP_KIND,
        )
