"""Contract tests over the REAL regime modules.

These exist because a signature change to `assign_domains` broke `random_id` and `official` --
which alias `base.geography_domains` directly rather than defining their own function -- and the
whole suite still passed. Every regime test in the tree used a hand-written stub with the new
signature, so nothing exercised the actual aliases. The failure mode was maximally quiet:
`base.iter_splits` catches the TypeError in a broad `except Exception`, downgrades it to a
`regime_problem`, and (before the exit-code fix) the run reported success having silently
evaluated 2 of 4 declared regimes.

So: parametrize over the real module names, and assert the contract holds for each.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from evals.regimes import base as RB

TABULAR_REGIMES = ["random_id", "official", "geographic_ood", "spatial_cluster_ood"]


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


@pytest.mark.parametrize("regime_name", TABULAR_REGIMES)
def test_real_regime_assign_domains_accepts_the_strategy(regime_name: str) -> None:
    """Every regime -- including the ones that ALIAS a shared function -- takes (bench, holdouts)."""
    bench, y = _bench()
    regime = RB.load_regime(regime_name)
    holdouts = RB.holdouts_for(_bench_mod(), regime_name)

    domains = regime.assign_domains(bench, holdouts)

    assert len(domains) == len(y)


@pytest.mark.parametrize("regime_name", TABULAR_REGIMES)
def test_real_regime_yields_splits_and_reports_no_problems(regime_name: str) -> None:
    """The end-to-end path the run actually takes: a declared regime must produce splits."""
    bench, y = _bench()
    holdouts = RB.holdouts_for(_bench_mod(), regime_name)

    splits = list(RB.iter_splits(
        regime_name, bench, y, holdouts, seed=0,
        strict_mode=False, val_group=RB.val_group_for(_bench_mod(), regime_name),
    ))

    assert splits, f"{regime_name} produced ZERO splits"
    assert not RB.REGIME_PROBLEMS, f"{regime_name} reported: {RB.REGIME_PROBLEMS}"


@pytest.mark.parametrize("regime_name", TABULAR_REGIMES)
def test_real_regime_under_strict_mode_does_not_raise(regime_name: str) -> None:
    """STRICT_MODE turns any regime problem into a hard failure -- there must be none to turn."""
    bench, y = _bench()
    holdouts = RB.holdouts_for(_bench_mod(), regime_name)

    list(RB.iter_splits(
        regime_name, bench, y, holdouts, seed=0,
        strict_mode=True, val_group=RB.val_group_for(_bench_mod(), regime_name),
    ))


def _bench_mod():
    from evals.benchmarks import cropharvest

    return cropharvest
