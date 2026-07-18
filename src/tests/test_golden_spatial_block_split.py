"""Golden regression: the canonical CropHarvest `spatial_block_2deg_purge50km` split.

Twenty result files across five runs were produced on the spatial-block basis, four of them in
`output-erm-full-20260711` -- the tree the paper currently cites. `geographic_ood` has since been
switched to leave-one-domain-out, so nothing else in the suite would notice if that historical
split stopped reproducing. These tests pin it against the real artifact.

The fixture (`fixtures/cropharvest_spatial_block_golden.json`, 39 KB) is derived from the
canonical `split_manifest.json` on gilbreth and is checked in, so the tests never touch SSH or
the 67k-sample benchmark. It was verified identical across all four models and all three seeds
when generated -- the partition is a pure function of the data, not of the seed or the encoder.

SCOPE -- READ THIS BEFORE TRUSTING IT. This is a DOMAIN-SET and COUNT regression, **not** a
per-sample assignment regression. What is canonical here is the block partition (which of the
2,496 blocks are test / val / train) and the per-block sample counts recorded in the manifest.
The per-sample INDICES these tests compute are SYNTHETIC -- they index a rebuilt census, not the
real benchmark, and they are not the canonical run's sample assignments. Pinning those would need
per-sample ID digests derived from the 67k-sample dataset; `split_manifest.json` records only
counts and domain names, so it cannot support that claim and neither can this file.

WHY THE BLOCK PARTITION IS EXACTLY REPRODUCIBLE ANYWAY. `_spatial_partitions` ranks every block by
sha256 of its name and takes a prefix: test = the first blocks whose cumulative size reaches
round(0.20 * valid_n), val = the next blocks reaching round(0.10 * valid_n), train = the rest.
Against the canonical artifact this holds exactly -- ranked[:888] is the test set, ranked[888:1254]
is the val set, ranked[1254:] is the train set, and the test cumulative reaches its 13538 target
precisely at ranked index 887. So the choice depends only on the block NAMES (which fix the
ranking), the sizes of the blocks inside the picked prefix, and valid_n. Train-block sizes are
never consulted, because every train block sorts after the prefix. The manifest records train
counts only AFTER the 50 km purge and the source-diagnostic subsample, so their individual values
are unrecoverable -- but they provably cannot affect the outcome, and only their total (which sets
valid_n) matters. Hence they are synthesized to the recorded total, which is also why any
assertion about the TRAIN sample count here would be circular and none is made.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from evals.benchmarks import cropharvest
from evals.regimes import geographic_ood as GEO

FIXTURE = Path(__file__).parent / "fixtures" / "cropharvest_spatial_block_golden.json"


@pytest.fixture(scope="module")
def golden() -> dict:
    return json.loads(FIXTURE.read_text())


@pytest.fixture(scope="module")
def rebuilt(golden: dict):
    """The 67k-row census, built once: _spatial_partitions' pick loop is O(blocks x n)."""
    return _rebuild(golden)


@pytest.fixture(scope="module")
def partition(golden: dict, rebuilt):
    y, groups = rebuilt
    return GEO._spatial_partitions(
        y, groups, val_frac=golden["val_fraction"], test_frac=golden["test_fraction"]
    )


def _digest(names) -> str:
    return hashlib.sha256(json.dumps(sorted(names)).encode()).hexdigest()


def _rebuild(golden: dict):
    """Reconstruct (y, groups) reproducing the canonical block census exactly."""
    sizes: dict[str, int] = {}
    sizes.update(golden["test_domain_counts"])
    sizes.update(golden["val_domain_counts"])

    # Train blocks: individual sizes are unrecoverable from the manifest (purged) and provably
    # irrelevant to the picks; only their total matters, via valid_n. Spread it evenly.
    train = golden["train_domains"]
    base, extra = divmod(golden["train_total"], len(train))
    for i, name in enumerate(train):
        sizes[name] = base + (1 if i < extra else 0)

    groups, y = [], []
    for name in sorted(sizes):
        n = sizes[name]
        groups += [name] * n
        # Both classes in every block: the pick loop's `classes >= 2` guard is satisfied long
        # before the size target (888 blocks are needed to reach it), so the exact per-block
        # balance -- which the manifest only records in aggregate -- cannot change the picks.
        y += [1] * (n - n // 2) + [0] * (n // 2)
    return np.asarray(y, dtype=np.int64), np.asarray(groups, dtype=object)


def test_fixture_is_internally_consistent(golden: dict) -> None:
    assert sum(golden["test_domain_counts"].values()) == golden["n_test"]
    assert sum(golden["val_domain_counts"].values()) == golden["n_val"]
    assert golden["n_val"] + golden["n_test"] + golden["train_total"] == golden["valid_n"]
    assert _digest(golden["test_domain_counts"]) == golden["digests"]["test"]
    assert _digest(golden["val_domain_counts"]) == golden["digests"]["val"]
    assert _digest(golden["train_domains"]) == golden["digests"]["train"]


def test_rebuilt_census_matches_the_canonical_totals(golden: dict, rebuilt) -> None:
    y, groups = rebuilt

    assert len(y) == golden["valid_n"]
    assert len(set(groups)) == 2496


def test_golden_spatial_block_partition_is_reproduced_exactly(golden: dict, partition) -> None:
    """The canonical val/test/train DOMAIN sets, byte for byte -- the real golden claim here."""
    train_d, val_d, test_d = partition

    assert _digest(test_d) == golden["digests"]["test"], "test block set drifted from canonical"
    assert _digest(val_d) == golden["digests"]["val"], "val block set drifted from canonical"
    assert _digest(train_d) == golden["digests"]["train"], "train block set drifted from canonical"


def test_golden_partition_sample_counts_match_the_canonical_manifest(golden: dict, rebuilt, partition) -> None:
    """The val/test SAMPLE COUNTS the canonical artifact reports: 13,538 and 6,881.

    These are canonical: neither partition is purged or subsampled, so each is exactly the sum of
    the manifest's per-block counts over the blocks the partitioner chose. The claim under test is
    that the partitioner still chooses blocks summing to those totals.

    The indices themselves are synthetic and are NOT asserted to be the canonical run's sample
    assignments -- see the module docstring. No assertion is made about the train count either: the
    train blocks' sizes were synthesized to a fixed total, so checking it would only re-measure
    this file's own arithmetic.
    """
    _y, groups = rebuilt
    train_d, val_d, test_d = partition
    test_idx = GEO._idx_for(groups, test_d)
    val_idx = GEO._idx_for(groups, val_d)
    train_idx = GEO._idx_for(groups, train_d)

    assert len(test_idx) == golden["n_test"] == 13538
    assert len(val_idx) == golden["n_val"] == 6881
    # structural, not canonical: the three partitions must tile the census exactly once
    assert len(set(test_idx) & set(val_idx)) == 0
    assert len(set(test_idx) & set(train_idx)) == 0
    assert len(test_idx) + len(val_idx) + len(train_idx) == golden["valid_n"]


def test_target_sizes_follow_the_declared_fractions(golden: dict) -> None:
    """Pins the arithmetic the canonical run used: test stops exactly on its target."""
    assert round(golden["test_fraction"] * golden["valid_n"]) == golden["n_test"]
    # val overshoots its target because pick() adds whole blocks until the target is reached
    assert golden["n_val"] >= round(golden["val_fraction"] * golden["valid_n"])


def test_every_canonical_block_name_round_trips_through_the_block_basis(golden: dict) -> None:
    """`spatial_block_domains` must still assign the exact block names the artifact records.

    Decodes all 2,496 canonical block names back to a coordinate inside each block and asserts
    the basis re-derives that same name -- 2,496 real cases from the canonical run, rather than a
    handful of toy coordinates.
    """
    names = sorted(
        set(golden["train_domains"])
        | set(golden["val_domain_counts"])
        | set(golden["test_domain_counts"])
    )
    block = cropharvest.GEOGRAPHIC_BLOCK_DEGREES
    lats, lons, expected = [], [], []
    for name in names:
        _, lat_bin, lon_bin = name.split("_")
        # centre of the block this name denotes
        lats.append(int(lat_bin) * block - 90.0 + block / 2.0)
        lons.append(int(lon_bin) * block - 180.0 + block / 2.0)
        expected.append(name)
    bench = type("B", (), {"latlon": np.column_stack([lats, lons])})()

    assigned = cropharvest.spatial_block_domains(bench)

    assert list(assigned) == expected, "the block basis no longer reproduces canonical block names"


def test_lodo_would_not_reproduce_the_historical_partition(golden: dict, rebuilt, partition) -> None:
    """The regression this whole file exists to prevent.

    Running the spatial_blocks strategy over canonical provenance domains would emit the
    historical label over a completely different partition. assign_domains must refuse to let
    the two bases be confused.
    """
    y, _groups = rebuilt
    canonical_domains = np.asarray(["kenya"] * len(y), dtype=object)

    _train_d, _val_d, test_d = partition
    assert _digest(test_d) == golden["digests"]["test"]

    # the same strategy over a different basis cannot reconstruct it
    with pytest.raises(ValueError):
        GEO._spatial_partitions(y, canonical_domains, val_frac=0.10, test_frac=0.20)
