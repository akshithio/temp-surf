"""Official release-split regime (schema v2).

Each benchmark's OFFICIAL split is its published release evaluation: a fixed target evaluation set
plus a source pool subdivided into train/validation. It is a target-geography route with NO
target-label access -- ``has_target=True``, ``supports_target_labels=False`` -- so
``target_label_pool`` is ALWAYS empty, the runtime fits on source_train, tunes/calibrates on
source_val, and evaluates ZERO-SHOT on target_test. There is NO generic geographic fallback; each
benchmark's exact release definition is implemented explicitly:

  * EuroCropsML -- both release anchors from ``bench.official_splits`` (fixed row-index train/val/test);
    the finetune ``target_train`` is IGNORED (no target-label routing). Membership identical across seeds.
  * CropHarvest -- Togo only: the fixed 1,272-example ``togo`` provenance source pool subdivided
    seed-specifically 1,145/127 (class-preserving, via ``partition_source``) into
    source_train/source_val, and the fixed 306-example ``togo-eval`` provenance as target_test. Only
    the source subdivision varies by seed.
  * BreizhCrops -- FRH01+FRH02 source_train, FRH03 source_val, FRH04 target_test. Fixed across seeds.
  * PASTIS (dense) -- folds 1-3 source_train patches, fold 4 source_val patches, fold 5 target_test
    patches. Fixed across seeds, patch-level throughout.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from evals import partition, split_spec
from evals.regimes.base import DenseSourceTargetSplit, SourceTargetSplit, emit_split_audit_event

NAME = "official"
GROUP_KIND = "geography"
HAS_TARGET = True
# official HAS a target geography (the held-out release evaluation set) but exposes NO target-label
# access -- it is a fixed release split evaluated zero-shot, not a label-budget route.
SUPPORTS_TARGET_LABELS = False


def _split(label, source_train, source_val, target_test) -> SourceTargetSplit:
    """Build the canonical official SourceTargetSplit: source_test and target_label_pool ALWAYS empty
    (supports_target_labels=False), target_test the fixed release evaluation membership."""
    return SourceTargetSplit(
        label=str(label),
        source_train=np.sort(np.asarray(source_train, dtype=np.int64)),
        source_val=np.sort(np.asarray(source_val, dtype=np.int64)),
        source_test=np.empty(0, dtype=np.int64),
        target_label_pool=np.empty(0, dtype=np.int64),
        target_test=np.sort(np.asarray(target_test, dtype=np.int64)),
        has_target=True, supports_target_labels=False, group_kind=GROUP_KIND,
    )


# --------------------------------------------------------------------------- #
# Tabular
# --------------------------------------------------------------------------- #
def iter_source_target_splits(bench, bench_mod, seed: int) -> Iterator[SourceTargetSplit]:
    if getattr(bench, "official_splits", None):
        yield from _anchor_official(bench, bench_mod)          # EuroCropsML: fixed row-index anchors
    elif getattr(bench_mod, "OFFICIAL_PROVENANCE", None):
        yield from _provenance_official(bench, bench_mod, seed)  # CropHarvest: togo / togo-eval provenance
    else:
        yield from _region_official(bench, bench_mod)          # BreizhCrops: fixed region holdout


def _anchor_official(bench, bench_mod) -> Iterator[SourceTargetSplit]:
    """EuroCropsML: both release anchors (fixed row-index train/val/test), identical across seeds."""
    for holdout in bench_mod.OFFICIAL_HOLDOUTS:
        spec = bench.official_splits.get(str(holdout))
        if not spec:
            emit_split_audit_event("dropped_holdout", regime="official", holdout=str(holdout), reason="not_found_in_metadata")
            continue
        train, val, test = spec.get("train", []), spec.get("val", []), spec.get("test", [])
        if not len(train) or not len(val) or not len(test):
            emit_split_audit_event("dropped_holdout", regime="official", holdout=str(holdout), reason="empty source_train/source_val/target_test")
            continue
        # spec["target_train"] (finetune labels) is deliberately IGNORED -- supports_target_labels=False.
        yield _split(holdout, train, val, test)


def _provenance_official(bench, bench_mod, seed: int) -> Iterator[SourceTargetSplit]:
    """CropHarvest Togo: the ``togo`` provenance source pool subdivided seed-specifically 90/10
    (class-preserving via partition_source), ``togo-eval`` provenance as target_test. ONLY the source
    subdivision varies by seed; sizes and class marginals do not, and target_test is fixed."""
    spec = bench_mod.OFFICIAL_PROVENANCE
    prov = np.asarray(bench_mod.provenance_groups(bench), dtype=object)
    y, _groups = bench_mod.make_targets(bench)
    source_rows = np.flatnonzero(prov == str(spec["source"]))
    target_rows = np.flatnonzero(prov == str(spec["target"]))
    if not len(source_rows) or not len(target_rows):
        emit_split_audit_event(
            "dropped_holdout", regime="official", holdout=str(spec["source"]),
            reason="empty official source or target provenance",
        )
        return
    train_n, val_n = split_spec.official_source_train_val_sizes(len(source_rows))
    classes = [str(c) for c in np.asarray(y)[source_rows].tolist()]
    regions = [str(spec["source"])] * len(source_rows)  # single provenance -> pure class-stratified subdivision
    assign = partition.partition_source(
        classes, regions, [("source_train", train_n), ("source_val", val_n)], int(seed)
    )
    label = str(bench_mod.OFFICIAL_HOLDOUTS[0]) if getattr(bench_mod, "OFFICIAL_HOLDOUTS", None) else str(spec["source"])
    yield _split(label, source_rows[assign["source_train"]], source_rows[assign["source_val"]], target_rows)


def _region_official(bench, bench_mod) -> Iterator[SourceTargetSplit]:
    """BreizhCrops: FRH01+FRH02 source_train, FRH03 source_val, FRH04 target_test (fixed across seeds)."""
    groups = np.asarray(bench.groups, dtype=object)
    target_region = str(bench_mod.OFFICIAL_HOLDOUTS[0])
    val_region = str(bench_mod.OFFICIAL_VAL_HOLDOUT)
    target_rows = np.flatnonzero(groups == target_region)
    val_rows = np.flatnonzero(groups == val_region)
    train_rows = np.flatnonzero(~np.isin(groups, np.asarray([target_region, val_region], dtype=object)))
    if not len(train_rows) or not len(val_rows) or not len(target_rows):
        emit_split_audit_event(
            "dropped_holdout", regime="official", holdout=target_region,
            reason="empty source_train/source_val/target_test region",
        )
        return
    yield _split(target_region, train_rows, val_rows, target_rows)


def sample_domains(bench, bench_mod) -> np.ndarray:
    """Per-sample domain basis: the benchmark's native geographic groups (country / NUTS-3 region /
    canonical region), recorded for worst-group scoring on the assigned source + target_test samples."""
    del bench_mod
    return np.asarray(bench.groups, dtype=object)


# --------------------------------------------------------------------------- #
# Dense (PASTIS) -- fixed published folds, patch-level
# --------------------------------------------------------------------------- #
def iter_dense_source_target_splits(bench, bench_mod, seed: int) -> Iterator[DenseSourceTargetSplit]:
    """PASTIS: folds 1-3 -> source_train, fold 4 -> source_val, fold 5 -> target_test. Fixed across
    seeds; allocation is over patch IDs only."""
    del seed  # the published folds are fixed -- official dense membership never varies with the seed
    fold_of = {int(p.patch_id): int(p.fold) for p in bench.patches}
    train_folds = {int(f) for f in bench_mod.TRAIN_FOLDS}
    val_folds = {int(f) for f in bench_mod.VAL_FOLDS}
    test_folds = {int(f) for f in bench_mod.TEST_FOLDS}

    def patches_in(folds: set[int]) -> frozenset[int]:
        return frozenset(p for p, f in fold_of.items() if f in folds)

    yield DenseSourceTargetSplit(
        label=f"fold_{sorted(test_folds)[0]}",
        source_train_patches=patches_in(train_folds),
        source_val_patches=patches_in(val_folds),
        source_test_patches=frozenset(),
        target_label_pool_patches=frozenset(),
        target_test_patches=patches_in(test_folds),
        has_target=True, supports_target_labels=False, group_kind=GROUP_KIND,
    )


def patch_domains(bench, bench_mod) -> dict[int, str]:
    """Per-patch domain basis: the published FOLD -- the official split's defining metadata, and the
    basis the runtime structural check (load_dense_split, kind='published fold') validates against."""
    del bench_mod
    return {int(p.patch_id): str(p.fold) for p in bench.patches}
