from __future__ import annotations

import numpy as np

from evals.regimes.base import DenseSplit, Split, emit_split_audit_event, geography_domains
from evals.regimes.geographic_ood import make_strict_holdout_splits

NAME = "official"
GROUP_KIND = "geography"
HAS_TARGET = True
USES_CURATED_HOLDOUTS = True
assign_domains = geography_domains


def iter_splits(y, groups, *, seed, holdouts, n_folds=None, val_group=None, **_):
    del n_folds
    bench = _.get("bench")
    exact = getattr(bench, "official_splits", {}) if bench is not None else {}
    if exact:
        for holdout in holdouts:
            spec = exact.get(str(holdout))
            if not spec:
                emit_split_audit_event(
                    "dropped_holdout", regime="official", holdout=str(holdout), reason="not_found_in_metadata"
                )
                print(f"   !! official: split {holdout!r} not found in benchmark metadata", flush=True)
                continue
            train = np.asarray(spec.get("train", []), dtype=np.int64)
            val = np.asarray(spec.get("val", []), dtype=np.int64)
            test = np.asarray(spec.get("test", []), dtype=np.int64)
            try:
                if not len(train) or not len(val) or not len(test):
                    raise ValueError("empty train/val/test after intersecting loaded samples")
                if len(np.unique(y[train])) < 2 or len(np.unique(y[test])) < 2:
                    raise ValueError("train or test split is one-class")
            except ValueError as exc:
                emit_split_audit_event(
                    "dropped_holdout", regime="official", holdout=str(holdout), reason=str(exc)
                )
                print(f"   !! official: split {holdout!r} dropped ({exc})", flush=True)
                continue
            yield Split(str(holdout), np.sort(train), np.sort(test), np.sort(val), has_target=False)
        return
    for holdout in holdouts:
        try:
            train, val, test, _train_val = make_strict_holdout_splits(
                y, groups, holdout, seed, val_group=val_group
            )
        except ValueError as exc:
            emit_split_audit_event(
                "dropped_holdout", regime="official", holdout=str(holdout), reason=str(exc)
            )
            print(f"   !! official: holdout {holdout!r} dropped ({exc})", flush=True)
            continue
        yield Split(str(holdout), train, test, val)


def iter_fold_splits(bench_mod):
    test_fold = sorted(bench_mod.TEST_FOLDS)[0]
    yield DenseSplit(
        f"fold_{test_fold}",
        set(bench_mod.TRAIN_FOLDS),
        set(bench_mod.VAL_FOLDS),
        set(bench_mod.TEST_FOLDS),
        has_target=HAS_TARGET,
        group_kind=GROUP_KIND,
    )
