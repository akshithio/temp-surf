from __future__ import annotations

from evals.regimes.base import DenseSplit, Split, geography_domains
from evals.regimes.geographic_ood import make_strict_holdout_splits

NAME = "official"
GROUP_KIND = "geography"
HAS_TARGET = True
USES_CURATED_HOLDOUTS = True
assign_domains = geography_domains


def iter_splits(y, groups, *, seed, holdouts, n_folds=None, val_group=None, **_):
    del n_folds
    for holdout in holdouts:
        try:
            train, val, test, _train_val = make_strict_holdout_splits(
                y, groups, holdout, seed, val_group=val_group
            )
        except ValueError as exc:
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
