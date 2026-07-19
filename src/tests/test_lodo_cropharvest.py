"""CropHarvest geographic_ood (schema v2): the benchmark-specific frozen geography.

The generic LODO mechanics (source coverage, purge-before-partition, 80/10/10 + 80/20, one-class
stress) are pinned in test_regime_geographic.py. This file pins CropHarvest's specifics: the headline
rotation is the canonical LOCALIZED regions only, GeoWiki and Croplands stay source-only global
collections, the Mali provenance merge produces a single ``mali`` canonical target, one-class regions
are zero-shot stress targets (no target-label route), and the purge radius is 50 km. Synthetic data.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from evals import split_spec
from evals.regimes import base as RB
from evals.regimes import geographic_ood as geo

# headline localized regions (subset), a one-class supplementary region, and the two source-only
# global collections -- all real CropHarvest canonical names.
_HEADLINE = ("kenya", "togo", "mali", "ethiopia")
_SUPPLEMENTARY = ("central-asia",)  # one-class stress
_SOURCE_ONLY = ("geowiki-landcover-2017", "croplands")

_CENTERS = {
    "kenya": (0.5, 37.0), "togo": (8.0, 1.0), "mali": (17.0, -4.0), "ethiopia": (9.0, 40.0),
    "central-asia": (43.0, 68.0), "geowiki-landcover-2017": (20.0, 20.0), "croplands": (-10.0, -50.0),
}


def _ch_bench(per=40, seed=0):
    rng = np.random.default_rng(seed)
    groups, y, latlon = [], [], []
    for dom in (*_HEADLINE, *_SUPPLEMENTARY, *_SOURCE_ONLY):
        la, lo = _CENTERS[dom]
        for i in range(per):
            groups.append(dom)
            y.append(0 if dom in _SUPPLEMENTARY else i % 2)  # supplementary regions are one-class
            latlon.append((la + rng.normal(0, 0.05), lo + rng.normal(0, 0.05)))
    bench = SimpleNamespace(
        name="cropharvest", groups=np.asarray(groups, dtype=object), labels=np.asarray(y, dtype=np.int64),
        latlon=np.asarray(latlon, dtype=float), sample_ids=np.asarray([f"s{i}" for i in range(len(y))], dtype=object),
    )
    bench_mod = SimpleNamespace(BENCHMARK="cropharvest", make_targets=lambda b: (b.labels, b.groups))
    return bench, bench_mod


def _splits(bench, bench_mod, seed=0):
    return {s.label: s for s in geo.iter_source_target_splits(bench, bench_mod, seed)}


def test_headline_targets_are_canonical_localized_regions_only():
    bench, bench_mod = _ch_bench()
    labels = set(_splits(bench, bench_mod))
    # every present headline + supplementary region rotates as a target
    assert set(_HEADLINE) | set(_SUPPLEMENTARY) <= labels
    # the source-only global collections NEVER rotate as targets
    for unit in _SOURCE_ONLY:
        assert unit not in labels
    assert set(split_spec.CROPHARVEST.source_only_units) == {"croplands", "geowiki-landcover-2017"}


def test_geowiki_and_croplands_stay_in_the_source_never_a_target():
    bench, bench_mod = _ch_bench()
    groups = np.asarray(bench.groups).astype(str)
    sp = _splits(bench, bench_mod)["kenya"]
    src = np.concatenate([sp.source_train, sp.source_val, sp.source_test])
    src_domains = set(groups[src])
    # both global collections are present in the source pool for a localized target
    assert {"geowiki-landcover-2017", "croplands"} <= src_domains


def test_mali_rotates_as_a_headline_target_with_complete_non_mali_source():
    bench, bench_mod = _ch_bench()
    groups = np.asarray(bench.groups).astype(str)
    sp = _splits(bench, bench_mod)["mali"]
    assert sp.has_target is True and sp.supports_target_labels is True
    target_rows = set(np.flatnonzero(groups == "mali").tolist())
    assert set(sp.target_label_pool.tolist()) | set(sp.target_test.tolist()) == target_rows
    src = np.concatenate([sp.source_train, sp.source_val, sp.source_test])
    # source is the COMPLETE non-mali population (every other present domain, incl. source-only)
    assert set(groups[src]) == set(groups.tolist()) - {"mali"}


def test_one_class_region_is_zero_shot_stress_not_a_label_route():
    bench, bench_mod = _ch_bench()
    groups = np.asarray(bench.groups).astype(str)
    sp = _splits(bench, bench_mod)["central-asia"]
    assert sp.has_target is True
    assert sp.supports_target_labels is False        # excluded from target-label routes
    assert sp.target_label_pool.size == 0            # no few-shot pool
    assert set(sp.target_test.tolist()) == set(np.flatnonzero(groups == "central-asia").tolist())


def test_purge_radius_is_50km_and_fires_before_partitioning():
    assert split_spec.CROPHARVEST.purge_km == 50.0
    bench, bench_mod = _ch_bench()
    groups = np.asarray(bench.groups).astype(str)
    # co-locate togo ONTO kenya so the 50 km purge removes togo source rows for the kenya target
    bench.latlon[groups == "togo"] = bench.latlon[groups == "kenya"][0] + 1e-4

    RB.clear_split_audit_events()
    events = RB.SPLIT_AUDIT_EVENTS
    windows, prev, splits = {}, 0, {}
    for s in geo.iter_source_target_splits(bench, bench_mod, 0):
        windows[s.label] = list(events[prev:len(events)])
        prev = len(events)
        splits[s.label] = s
    purges = [e for e in windows["kenya"] if e["kind"] == "purge"]
    assert purges and all(e["radius_km"] == 50.0 for e in purges)
    purged = set().union(*(set(e["purged_indices"]) for e in purges))
    src = set(splits["kenya"].source_train.tolist()) | set(splits["kenya"].source_val.tolist()) \
        | set(splits["kenya"].source_test.tolist())
    assert purged and src.isdisjoint(purged)  # purge happened BEFORE partitioning
    assert purged.issubset(set(np.flatnonzero(groups == "togo").tolist()))
