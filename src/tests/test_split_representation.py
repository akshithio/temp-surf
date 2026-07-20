"""Schema-v2 explicit source/target split representation + route capabilities.

The dependency-order foundation of the schema-v2 clean break: the explicit partition vocabulary
(source_train/source_val/source_test/target_label_pool/target_test), the SourceTargetSplit /
DenseSourceTargetSplit dataclasses that replace the overloaded v1 Split contract, and the
first-class route capabilities (has_target / supports_target_labels) -- including official's
target-geography-but-no-target-labels case. No data, no artifacts.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from evals import partition as P
from evals.regimes import base as RB
from evals.regimes import geographic_ood, official, random_id, spatial_cluster_ood


# --------------------------------------------------------------------------- #
# Partition vocabulary
# --------------------------------------------------------------------------- #
def test_v2_partition_vocabulary():
    assert RB.SOURCE_PARTITIONS == ("source_train", "source_val", "source_test")
    assert RB.TARGET_PARTITIONS == ("target_label_pool", "target_test")
    assert RB.V2_PARTITIONS == RB.SOURCE_PARTITIONS + RB.TARGET_PARTITIONS
    # aligns with the partitioner's target-partition names
    assert P.TARGET_LABEL_POOL == "target_label_pool" and P.TARGET_TEST == "target_test"
    assert {P.TARGET_LABEL_POOL, P.TARGET_TEST} == set(RB.TARGET_PARTITIONS)


# --------------------------------------------------------------------------- #
# Route capabilities
# --------------------------------------------------------------------------- #
def test_route_capabilities_per_regime():
    assert RB.route_capabilities(random_id) == (False, False)          # source-only, no target
    assert RB.route_capabilities(official) == (True, False)            # target geography, NO target labels
    assert RB.route_capabilities(geographic_ood) == (True, True)
    assert RB.route_capabilities(spatial_cluster_ood) == (True, False)  # sensitivity split, zero-shot


def test_route_capabilities_is_fail_closed():
    # both flags REQUIRED (no default inference)
    with pytest.raises(ValueError, match="must declare HAS_TARGET"):
        RB.route_capabilities(SimpleNamespace(SUPPORTS_TARGET_LABELS=True, NAME="x"))
    with pytest.raises(ValueError, match="must declare SUPPORTS_TARGET_LABELS"):
        RB.route_capabilities(SimpleNamespace(HAS_TARGET=True, NAME="x"))
    with pytest.raises(ValueError, match="must declare"):
        RB.route_capabilities(SimpleNamespace(NAME="x"))
    # each must be a real bool
    with pytest.raises(ValueError, match="must be a bool"):
        RB.route_capabilities(SimpleNamespace(HAS_TARGET=1, SUPPORTS_TARGET_LABELS=True, NAME="x"))
    # supports=True requires has_target=True
    with pytest.raises(ValueError, match="requires HAS_TARGET=True"):
        RB.route_capabilities(SimpleNamespace(HAS_TARGET=False, SUPPORTS_TARGET_LABELS=True, NAME="x"))
    # valid combinations
    assert RB.route_capabilities(SimpleNamespace(HAS_TARGET=True, SUPPORTS_TARGET_LABELS=False)) == (True, False)
    assert RB.route_capabilities(SimpleNamespace(HAS_TARGET=False, SUPPORTS_TARGET_LABELS=False)) == (False, False)


# --------------------------------------------------------------------------- #
# SourceTargetSplit
# --------------------------------------------------------------------------- #
def test_source_target_split_fields_and_as_partitions():
    s = RB.SourceTargetSplit(
        label="kenya",
        source_train=np.array([0, 1]), source_val=np.array([2]), source_test=np.array([3]),
        target_label_pool=np.array([4, 5]), target_test=np.array([6]),
        domain="kenya", has_target=True, supports_target_labels=True,
    )
    parts = s.as_partitions()
    assert list(parts) == list(RB.V2_PARTITIONS)  # canonical order
    assert parts["source_train"].tolist() == [0, 1]
    assert parts["target_label_pool"].tolist() == [4, 5]
    assert parts["target_test"].tolist() == [6]
    assert s.group_kind == "geography"


def test_source_target_split_source_only_defaults():
    s = RB.SourceTargetSplit(
        label="random_id",
        source_train=np.array([0]), source_val=np.array([1]), source_test=np.array([2]),
        has_target=False, supports_target_labels=False,
    )
    assert s.target_label_pool.size == 0 and s.target_test.size == 0
    assert s.has_target is False and s.supports_target_labels is False
    assert s.domain is None


# --------------------------------------------------------------------------- #
# DenseSourceTargetSplit (patch-level)
# --------------------------------------------------------------------------- #
def test_dense_source_target_split_fields_and_partitions():
    d = RB.DenseSourceTargetSplit(
        label="T31TFM",
        source_train_patches=frozenset({1, 2}),
        source_val_patches=frozenset({3}),
        source_test_patches=frozenset({4}),
        target_label_pool_patches=frozenset({5, 6}),
        target_test_patches=frozenset({7}),
    )
    assert d.has_target and d.supports_target_labels  # defaults
    parts = d.as_partitions()
    assert list(parts) == list(RB.V2_PARTITIONS)
    assert parts["target_label_pool"] == frozenset({5, 6})
    assert parts["target_test"] == frozenset({7})


def test_dense_source_target_split_is_hashable_frozen():
    d = RB.DenseSourceTargetSplit(
        label="x", source_train_patches=frozenset({1}),
        source_val_patches=frozenset(), source_test_patches=frozenset(),
        has_target=False, supports_target_labels=False,  # source-only -> empty target partitions
    )
    assert hash(d) == hash(d)  # frozenset fields -> a genuinely hashable frozen dataclass
    with pytest.raises(Exception):  # noqa: B017 -- frozen: fields cannot be reassigned
        d.label = "y"  # type: ignore[misc]


@pytest.mark.parametrize("has_t,supp,pool,test,match", [
    (True, False, [1], [2], "empty target_label_pool"),        # supports=False needs empty pool
    (True, True, [], [2], "non-empty target_label_pool"),      # supports=True needs non-empty pool
    (True, True, [1], [], "non-empty target_test"),            # has_target needs non-empty test
    (False, True, [], [], "requires has_target=True"),         # supports=True needs has_target
    (False, False, [1], [], "both target partitions empty"),   # no target -> both empty
])
def test_source_target_split_route_invariants_enforced_at_construction(has_t, supp, pool, test, match):
    with pytest.raises(ValueError, match=match):
        RB.SourceTargetSplit(
            label="x", source_train=np.array([0]), source_val=np.array([1]), source_test=np.array([2]),
            target_label_pool=np.array(pool, dtype=np.int64), target_test=np.array(test, dtype=np.int64),
            has_target=has_t, supports_target_labels=supp,
        )


def test_dense_split_route_invariants_enforced_at_construction():
    with pytest.raises(ValueError, match="non-empty target_test"):
        RB.DenseSourceTargetSplit(
            label="x", source_train_patches=frozenset({1}), source_val_patches=frozenset({2}),
            source_test_patches=frozenset({3}), has_target=True, supports_target_labels=True,
            target_label_pool_patches=frozenset({4}), target_test_patches=frozenset(),
        )


@pytest.mark.parametrize("flag", ["has_target", "supports_target_labels"])
@pytest.mark.parametrize("bad", [1, 0, "true", None])
def test_source_target_split_rejects_non_bool_flags(flag, bad):
    kw = dict(has_target=False, supports_target_labels=False)
    kw[flag] = bad
    with pytest.raises(ValueError, match=f"{flag} must be a bool"):
        RB.SourceTargetSplit(
            label="x", source_train=np.array([0]), source_val=np.array([1]), source_test=np.array([2]), **kw
        )


@pytest.mark.parametrize("flag", ["has_target", "supports_target_labels"])
@pytest.mark.parametrize("bad", [1, 0, "true"])
def test_dense_split_rejects_non_bool_flags(flag, bad):
    kw = dict(has_target=False, supports_target_labels=False)
    kw[flag] = bad
    with pytest.raises(ValueError, match=f"{flag} must be a bool"):
        RB.DenseSourceTargetSplit(
            label="x", source_train_patches=frozenset({1}), source_val_patches=frozenset(),
            source_test_patches=frozenset(), **kw
        )
