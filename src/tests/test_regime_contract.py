"""Contract tests over the REAL regime modules.

Each of the four regimes must speak the schema-v2 contract on its actual module object (not a
hand-written stub): fail-closed ``route_capabilities`` (HAS_TARGET / SUPPORTS_TARGET_LABELS) and an
explicit ``iter_source_target_splits`` emitter that yields SourceTargetSplit without recording a
regime_problem. Each test also asserts the retired v1 contract (``iter_splits`` / ``assign_domains``
/ ``iter_dense_splits``) is genuinely gone, so no silent alias can be reintroduced.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from evals.regimes import base as RB

#: All four regimes are schema v2 (explicit source/target emitters + fail-closed route
#: capabilities); the ``test_*_speaks_the_v2_regime_contract`` tests below pin each contract, and no
#: regime remains on the retired v1 iter_splits / assign_domains path.


#: Real CropHarvest domain names -- `official` holds out kenya/lem-brazil/togo by name, so a
#: bench with invented domain labels would fail that regime for the wrong reason.
DOMAINS = ("kenya", "lem-brazil", "togo", "ethiopia")


def _bench():
    """Four real domains, two classes each, coordinates far enough apart to cluster."""
    groups = np.asarray(sum(([d] * 12 for d in DOMAINS), []), dtype=object)
    y = np.asarray(([1] * 6 + [0] * 6) * 4, dtype=np.int64)
    latlon = np.concatenate([
        np.tile([0.0, 0.0], (12, 1)),
        np.tile([20.0, 20.0], (12, 1)),
        np.tile([-20.0, 40.0], (12, 1)),
        np.tile([40.0, -20.0], (12, 1)),
    ]) + np.linspace(0, 0.05, 48).reshape(-1, 1)  # jitter so clusters are non-degenerate
    return SimpleNamespace(
        name="cropharvest", groups=groups, latlon=latlon, labels=y,
        sample_ids=np.arange(len(y)), years=np.zeros(len(y), dtype=np.int64),
    ), y


@pytest.fixture(autouse=True)
def _clean():
    RB.clear_regime_problems()
    RB.clear_domain_census()
    yield
    RB.clear_regime_problems()
    RB.clear_domain_census()


def test_random_id_speaks_the_v2_regime_contract() -> None:
    """random_id is migrated to schema v2: it exposes the explicit source/target emitter + fail-closed
    route capabilities, and NO LONGER the retired v1 iter_splits / assign_domains contract the three
    parametrized tests above pin for the not-yet-migrated regimes."""
    from evals.regimes import random_id

    bench, y = _bench()
    bench_mod = _bench_mod()

    assert RB.route_capabilities(random_id) == (False, False)  # source-only, fail-closed
    splits = list(random_id.iter_source_target_splits(bench, bench_mod, seed=0))
    assert len(splits) == 1
    s = splits[0]
    assert s.has_target is False and s.supports_target_labels is False
    assert not RB.REGIME_PROBLEMS  # the v2 emitter raises on infeasibility; it never records a problem
    assert len(random_id.sample_domains(bench, bench_mod)) == len(y)
    # the retired v1 contract is genuinely gone (no silent alias left behind)
    assert not hasattr(random_id, "iter_splits")
    assert not hasattr(random_id, "assign_domains")


def test_official_speaks_the_v2_regime_contract() -> None:
    """official is migrated to schema v2: explicit source/target emitter + fail-closed route
    capabilities (target geography, NO target labels), NOT the retired v1 iter_splits/assign_domains."""
    from evals.benchmarks import breizhcrops as bz
    from evals.regimes import official

    assert RB.route_capabilities(official) == (True, False)  # has_target, but no target-label access
    # a BreizhCrops-style region official on a synthetic bench (FRH01+FRH02 / FRH03 / FRH04)
    groups = np.asarray(["frh01"] * 8 + ["frh02"] * 8 + ["frh03"] * 6 + ["frh04"] * 6, dtype=object)
    bench = SimpleNamespace(
        groups=groups, labels=np.arange(len(groups)) % 3, sample_ids=np.arange(len(groups)),
    )
    [split] = list(official.iter_source_target_splits(bench, bz, 0))
    assert split.label == "frh04"
    assert split.has_target is True and split.supports_target_labels is False
    assert split.source_test.size == 0 and split.target_label_pool.size == 0
    assert not RB.REGIME_PROBLEMS
    # the retired v1 contract is genuinely gone (no silent alias/fallback left behind)
    assert not hasattr(official, "iter_splits")
    assert not hasattr(official, "iter_fold_splits")
    assert not hasattr(official, "assign_domains")


def test_geographic_ood_speaks_the_v2_regime_contract() -> None:
    """geographic_ood is migrated to schema v2: true LODO emitter + fail-closed route capabilities
    (target geography WITH target labels), NOT the retired v1 iter_splits/assign_domains/census."""
    from evals.benchmarks import breizhcrops as bz
    from evals.regimes import geographic_ood as geo

    assert RB.route_capabilities(geo) == (True, True)  # has_target AND supports target labels
    groups = np.asarray(["frh01"] * 12 + ["frh02"] * 12 + ["frh03"] * 12 + ["frh04"] * 12, dtype=object)
    bench = SimpleNamespace(
        name="breizhcrops", groups=groups, labels=np.arange(len(groups)) % 3,
        latlon=np.column_stack([np.linspace(48, 49, len(groups)), np.linspace(-4, -2, len(groups))]),
        sample_ids=np.arange(len(groups)),
    )
    splits = {s.label: s for s in geo.iter_source_target_splits(bench, bz, 0)}
    assert set(splits) == {"frh01", "frh02", "frh03", "frh04"}   # LODO over all four regions
    sp = splits["frh04"]
    assert sp.has_target is True and sp.supports_target_labels is True
    assert sp.target_label_pool.size > 0 and sp.target_test.size > 0
    assert not RB.REGIME_PROBLEMS
    # the retired v1 contract is genuinely gone
    assert not hasattr(geo, "iter_splits")
    assert not hasattr(geo, "iter_fold_splits")
    assert not hasattr(geo, "assign_domains")
    assert not hasattr(geo, "make_strict_holdout_splits")


def test_spatial_cluster_ood_speaks_the_v2_regime_contract() -> None:
    """spatial_cluster_ood is migrated to schema v2: coordinate-only cell emitter + fail-closed route
    capabilities. It carries target geography but NO target labels -- it is a split-sensitivity
    analysis scored zero-shot, not a second deployment setting."""
    from evals.regimes import spatial_cluster_ood as sc

    assert RB.route_capabilities(sc) == (True, False)  # has_target, but NO target-label routes
    # five far-apart coordinate blobs (each two-class) -> five clean spherical-K-means cells
    centers = [(0.0, 0.0), (0.0, 40.0), (40.0, 0.0), (-40.0, 0.0), (0.0, -40.0)]
    groups, y, latlon = [], [], []
    for ci, (la, lo) in enumerate(centers):
        for i in range(12):
            groups.append(f"blob{ci}")
            y.append(i % 2)
            latlon.append((la + 0.01 * i, lo - 0.01 * i))
    bench = SimpleNamespace(
        name="cropharvest", groups=np.asarray(groups, dtype=object), labels=np.asarray(y, dtype=np.int64),
        latlon=np.asarray(latlon, dtype=float), sample_ids=np.arange(len(y)),
    )
    bench_mod = _bench_mod()

    splits = list(sc.iter_source_target_splits(bench, bench_mod, 0))
    assert [s.label for s in splits] == list(sc._CELL_NAMES)  # exactly the five fixed cells, in order
    assert len(splits) == sc.N_CLUSTERS == 5
    sp = splits[0]
    assert sp.has_target is True and sp.supports_target_labels is False
    assert sp.target_label_pool.size == 0 and sp.target_test.size > 0  # whole cell is zero-shot eval
    assert sp.target_role == RB.TARGET_ROLE_HEADLINE
    assert not RB.REGIME_PROBLEMS  # the v2 emitter raises on infeasibility; it never records a problem
    assert len(sc.sample_domains(bench, bench_mod)) == len(y)
    # the retired v1 contract is genuinely gone (no silent alias/fallback left behind)
    assert not hasattr(sc, "iter_splits")
    assert not hasattr(sc, "iter_dense_splits")
    assert not hasattr(sc, "assign_domains")


def _bench_mod():
    from evals.benchmarks import cropharvest

    return cropharvest
