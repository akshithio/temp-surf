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
import pytest

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


# --------------------------------------------------------------------------- #
# Territorial exclusion: the distance purge alone does not establish LODO
# --------------------------------------------------------------------------- #
def _interleaved_bench(per=40, ring=24, radius_deg=1.5, seed=0):
    """A target sampled around a RING plus global-collection points at its centre.

    The centre points are ~167 km from every labelled target sample, so they clear the 50 km purge
    outright -- yet they sit squarely inside the target's territory. This is the real CropHarvest
    failure mode (globally distributed collections interleaved with every region); the tight
    single-blob fixtures elsewhere in this file structurally cannot express it.
    """
    rng = np.random.default_rng(seed)
    groups, y, latlon = [], [], []
    la0, lo0 = _CENTERS["kenya"]
    for i in range(ring):                                   # kenya: a ring, not a blob
        ang = 2 * np.pi * i / ring
        groups.append("kenya")
        y.append(i % 2)
        latlon.append((la0 + radius_deg * np.cos(ang), lo0 + radius_deg * np.sin(ang)))
    for i in range(6):                                      # geowiki points INSIDE the ring
        groups.append("geowiki-landcover-2017")
        y.append(i % 2)
        latlon.append((la0 + 0.02 * (i - 3), lo0 + 0.02 * (i - 3)))
    for dom in ("togo", "mali", "ethiopia", "croplands"):   # ordinary far-away source domains
        la, lo = _CENTERS[dom]
        for i in range(per):
            groups.append(dom)
            y.append(i % 2)
            latlon.append((la + rng.normal(0, 0.05), lo + rng.normal(0, 0.05)))
    bench = SimpleNamespace(
        name="cropharvest", groups=np.asarray(groups, dtype=object), labels=np.asarray(y, dtype=np.int64),
        latlon=np.asarray(latlon, dtype=float), sample_ids=np.asarray([f"s{i}" for i in range(len(y))], dtype=object),
    )
    bench_mod = SimpleNamespace(BENCHMARK="cropharvest", make_targets=lambda b: (b.labels, b.groups))
    return bench, bench_mod


def test_in_footprint_source_points_are_excluded_even_though_they_pass_the_purge():
    bench, bench_mod = _interleaved_bench()
    kenya = _splits(bench, bench_mod)["kenya"]
    src = np.concatenate([kenya.source_train, kenya.source_val, kenya.source_test])
    retained = set(np.asarray(bench.groups)[src].tolist())
    assert "geowiki-landcover-2017" not in retained, (
        "geowiki points at the centre of the Kenya ring cleared the 50 km purge and were retained: "
        "the distance purge does not establish territorial exclusion"
    )
    # the far-away collections are untouched -- the mask excludes territory, not provenance
    assert {"togo", "mali", "ethiopia", "croplands"} <= retained


def test_the_distance_purge_alone_would_have_retained_them():
    """Pins WHY the mask is needed: every excluded point is >50 km from all labelled target samples."""
    from sklearn.neighbors import BallTree

    bench, _bench_mod = _interleaved_bench()
    groups = np.asarray(bench.groups)
    tgt = bench.latlon[groups == "kenya"]
    inside = bench.latlon[groups == "geowiki-landcover-2017"]
    tree = BallTree(np.deg2rad(tgt), metric="haversine")
    dist_km = tree.query(np.deg2rad(inside), k=1, return_distance=True)[0].ravel() * geo.EARTH_RADIUS_KM
    assert (dist_km > split_spec.CROPHARVEST.purge_km).all()


def test_footprint_exclusion_is_declared_only_where_domains_are_not_territories():
    """CropHarvest domains are provenance labels and need the mask; country/fold benchmarks do not."""
    assert split_spec.CROPHARVEST.footprint_exclusion is True
    for name in ("eurocropsml", "breizhcrops", "pastis"):
        assert split_spec.ALL_SPECS[name].footprint_exclusion is False


def test_footprint_exclusion_records_an_auditable_reason():
    bench, bench_mod = _interleaved_bench()
    RB.REGIME_PROBLEMS.clear()
    events = []
    orig = geo.emit_split_audit_event
    try:
        geo.emit_split_audit_event = lambda kind, **kw: events.append((kind, kw))
        _splits(bench, bench_mod)
    finally:
        geo.emit_split_audit_event = orig
    hits = [kw for kind, kw in events if kind == "footprint_exclusion"]
    # one spec per target, INCLUDING targets that excluded nothing -- "masked, zero hits" and
    # "never masked" are different claims and must be distinguishable in the artifact
    assert len(hits) >= 2
    assert any(h["n_excluded"] == 0 for h in hits)
    assert all(len(h["footprint_sha256"]) == 64 for h in hits)
    # the event carries everything needed to RECONSTRUCT and re-verify the decision boundary
    ev = next(h for h in hits if h["n_excluded"] == 6)
    assert ev["buffer_m"] == split_spec.CROPHARVEST.purge_km * 1000.0
    assert "+proj=aeqd" in ev["crs"] and "+units=m" in ev["crs"]
    assert ev["hull_wkt"].startswith("POLYGON")
    assert len(ev["footprint_sha256"]) == 64
    assert sorted(ev["excluded_indices"]) == ev["excluded_indices"]


# --------------------------------------------------------------------------- #
# The buffer is a real metric buffer, not degree-space radial scaling
# --------------------------------------------------------------------------- #
def _aeqd_offsets(lat0, lon0, target_pts, probe_pts):
    """Run the production footprint against explicit probe coordinates. Returns the inside mask."""
    import shapely

    footprint, _hull, transformer, _crs = geo.target_footprint(
        np.asarray(target_pts, dtype=float), 50_000.0, where="test"
    )
    probe = np.asarray(probe_pts, dtype=float)
    px, py = transformer.transform(probe[:, 1].tolist(), probe[:, 0].tolist())
    return np.asarray(shapely.intersects_xy(footprint, np.asarray(px), np.asarray(py)), dtype=bool)


def test_east_west_buffer_is_metric_at_high_latitude():
    """At 60 deg N a longitude degree is ~55 km, half its equatorial width.

    Degree-space expansion by 50/111.32 deg would reach ~2x too far east/west here. The metric buffer
    must extend ~50 km in EVERY direction: a point 45 km east is inside, one 55 km east is outside.
    """
    lat0 = 60.0
    km_per_deg_lon = 111.32 * np.cos(np.deg2rad(lat0))       # ~55.7 km at 60N
    target = [(lat0, 0.0), (lat0 + 0.001, 0.0), (lat0, 0.001)]   # a tiny cluster -> essentially a disc
    east_45 = (lat0, 45.0 / km_per_deg_lon)
    east_55 = (lat0, 55.0 / km_per_deg_lon)
    north_45 = (lat0 + 45.0 / 111.32, 0.0)
    north_55 = (lat0 + 55.0 / 111.32, 0.0)
    inside = _aeqd_offsets(lat0, 0.0, target, [east_45, east_55, north_45, north_55])
    assert inside.tolist() == [True, False, True, False], (
        "buffer is not isotropic in metres: east/west and north/south reach must both be ~50 km"
    )


def test_degree_space_expansion_would_have_been_wrong_here():
    """Pins WHY the projection matters: the old 50/111.32-degree reach is ~0.449 deg of longitude,
    which at 60N is only ~25 km -- so the naive rule under-buffers east/west by about half."""
    lat0 = 60.0
    deg = 50.0 / 111.32
    km_east_of_one_deg = 111.32 * np.cos(np.deg2rad(lat0))
    assert abs(deg * km_east_of_one_deg - 25.0) < 1.0        # ~25 km, not the intended 50 km


def test_buffer_offsets_edges_and_rounds_corners():
    """A true polygon buffer is constant-width along EDGES and rounds CORNERS.

    Radial vertex scaling fails both: it over-reaches along the diagonals and under-reaches at edge
    midpoints. Probes are placed just inside/outside 50 km from an edge midpoint and from a corner.
    """
    lat0, lon0 = 0.0, 0.0
    d = 1.0                                                   # ~111 km square, well beyond the buffer
    target = [(lat0 - d, lon0 - d), (lat0 - d, lon0 + d), (lat0 + d, lon0 + d), (lat0 + d, lon0 - d)]
    km = 111.32
    # due north of the TOP EDGE midpoint
    edge_in = (lat0 + d + 45.0 / km, lon0)
    edge_out = (lat0 + d + 55.0 / km, lon0)
    # diagonally beyond the NE CORNER: 45 / 55 km along the 45-degree bearing
    diag_in = (lat0 + d + (45.0 / np.sqrt(2)) / km, lon0 + d + (45.0 / np.sqrt(2)) / km)
    diag_out = (lat0 + d + (55.0 / np.sqrt(2)) / km, lon0 + d + (55.0 / np.sqrt(2)) / km)
    inside = _aeqd_offsets(lat0, lon0, target, [edge_in, edge_out, diag_in, diag_out])
    assert inside.tolist() == [True, False, True, False]


def test_a_point_just_outside_a_corner_diagonal_is_not_swept_in():
    """Radial scaling from the centroid would drag the whole diagonal outward. A rounded corner must
    exclude a point 70 km out along the diagonal even though it is only ~50 km from the corner axes."""
    lat0, lon0, d, km = 0.0, 0.0, 1.0, 111.32
    target = [(lat0 - d, lon0 - d), (lat0 - d, lon0 + d), (lat0 + d, lon0 + d), (lat0 + d, lon0 - d)]
    far_diag = (lat0 + d + (70.0 / np.sqrt(2)) / km, lon0 + d + (70.0 / np.sqrt(2)) / km)
    assert _aeqd_offsets(lat0, lon0, target, [far_diag]).tolist() == [False]


def test_single_target_point_yields_a_disc_not_a_failure():
    """One coordinate is not degenerate: its hull is a Point and the footprint is a 50 km disc."""
    inside = _aeqd_offsets(0.0, 0.0, [(0.0, 0.0)], [(0.0, 45.0 / 111.32), (0.0, 55.0 / 111.32)])
    assert inside.tolist() == [True, False]


def test_collinear_target_points_yield_a_capsule_not_a_failure():
    """Collinear coordinates give a LineString hull; buffering it is well defined (a capsule)."""
    target = [(0.0, 0.0), (0.0, 0.5), (0.0, 1.0)]
    inside = _aeqd_offsets(0.0, 0.5, target, [(45.0 / 111.32, 0.5), (55.0 / 111.32, 0.5)])
    assert inside.tolist() == [True, False]


# --------------------------------------------------------------------------- #
# Fail closed: a footprint that cannot be built must never silently disable exclusion
# --------------------------------------------------------------------------- #
def test_undefined_footprint_fails_closed():
    with pytest.raises(geo.FootprintError, match="no target coordinate is finite"):
        geo.target_footprint(np.array([[np.nan, np.nan]]), 50_000.0, where="test")
    with pytest.raises(geo.FootprintError, match="no target coordinate is finite"):
        geo.target_footprint(np.empty((0, 2)), 50_000.0, where="test")
    with pytest.raises(geo.FootprintError, match="buffer"):
        geo.target_footprint(np.array([[0.0, 0.0]]), 0.0, where="test")


def test_footprint_exclusion_without_coordinates_fails_closed():
    with pytest.raises(geo.FootprintError, match="requires coordinates"):
        geo._footprint_exclude(np.array([0]), np.array([1]), None, 50.0, where="test")


# --------------------------------------------------------------------------- #
# Event -> artifact: the exclusion reaches assignments.csv and the log with its own reason
# --------------------------------------------------------------------------- #
def test_footprint_rows_reach_assignments_csv_with_a_specific_reason():
    """The regression the generic-`unassigned` bug would have failed: footprint-masked ids must be
    status=purged / reason=inside_buffered_target_footprint, never excluded/unassigned."""
    from evals import split_artifacts as SA

    bench, bench_mod = _interleaved_bench()
    RB.REGIME_PROBLEMS.clear()
    events = []
    orig = geo.emit_split_audit_event
    try:
        geo.emit_split_audit_event = lambda kind, **kw: (events.append({"kind": kind, **kw}), None)[1]
        split = _splits(bench, bench_mod)["kenya"]
    finally:
        geo.emit_split_audit_event = orig

    rows, summary = SA.build_tabular_leaf(
        "cropharvest", "geographic_ood", 0, split=split,
        domains=np.asarray(bench.groups), labels=bench.labels, sample_ids=bench.sample_ids,
        audit_events=events, purge_km=50.0,
    )
    by_reason = {}
    for r in rows:
        by_reason.setdefault(r["reason"], []).append(r)

    footprint_rows = by_reason.get(SA.REASON_INSIDE_FOOTPRINT, [])
    assert len(footprint_rows) == 6
    assert all(r["status"] == SA.STATUS_PURGED and r["partition"] == "" for r in footprint_rows)
    # they are the geowiki points, and they did NOT fall through to the generic bucket
    assert all(r["domain"] == "geowiki-landcover-2017" for r in footprint_rows)
    assert not any(r["domain"] == "geowiki-landcover-2017" for r in by_reason.get("unassigned", []))

    # ...and the central-log summary counts the reason separately from the distance purge
    assert summary["purge_counts"][SA.REASON_INSIDE_FOOTPRINT] == 6
    assert summary["purge_count"] == sum(summary["purge_counts"].values())
    assert SA.REASON_INSIDE_FOOTPRINT not in summary["exclusion_counts"]


def test_distance_purge_and_footprint_reasons_stay_distinct_in_the_log():
    from evals import split_artifacts as SA

    bench, bench_mod = _interleaved_bench()
    events = [
        {"kind": "purge", "purged_indices": [0]},
        {"kind": "footprint_exclusion", "excluded_indices": [1]},
    ]
    split = _splits(bench, bench_mod)["kenya"]
    _rows, summary = SA.build_tabular_leaf(
        "cropharvest", "geographic_ood", 0, split=split,
        domains=np.asarray(bench.groups), labels=bench.labels, sample_ids=bench.sample_ids,
        audit_events=events, purge_km=50.0,
    )
    assert set(summary["purge_counts"]) <= set(SA.PURGE_REASONS)
