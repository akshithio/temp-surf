"""PASTIS-R metadata repair: EPSG:2154 -> EPSG:4326 coordinates + Sentinel-tile storage.

The old loader treated Lambert-93 easting/northing as lon/lat, so every real patch fell outside the
valid degree range and became NaN, and it stored no Sentinel tile at all -- which the geographic
tile-LODO regime needs. These tests pin the transform (against an independent pyproj call) and the
tile parsing on a synthetic Lambert-93 fixture; no real data.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

import evals.benchmarks.pastis as pastis
from evals import split_spec as S


# --------------------------------------------------------------------------- #
# _canonical_tile
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("raw,expected", [
    ("30UXV", "T30UXV"),
    ("T31TFM", "T31TFM"),
    (" t32ulu ", "T32ULU"),
    ("31tfj", "T31TFJ"),
    (None, None),   # absent -> None (a MISSING tile, hard-failed by assert_geographic_ready)
    ("", None),
    ("   ", None),
])
def test_canonical_tile(raw, expected):
    assert pastis._canonical_tile(raw) == expected


@pytest.mark.parametrize("bad", ["T123", "XYZ", "T31TF", "T31TFMM", "3UXV", "T311FM", "hello"])
def test_malformed_tile_raises(bad):
    with pytest.raises(ValueError, match="canonical Sentinel granule tile"):
        pastis._canonical_tile(bad)


# --------------------------------------------------------------------------- #
# _geometry_latlon
# --------------------------------------------------------------------------- #
def _square(cx, cy, half=32.0):
    return {"type": "Polygon", "coordinates": [[
        [cx - half, cy - half], [cx + half, cy - half],
        [cx + half, cy + half], [cx - half, cy + half], [cx - half, cy - half],
    ]]}


def test_lambert93_geometry_is_transformed_to_wgs84():
    from pyproj import Transformer
    t = Transformer.from_crs("EPSG:2154", "EPSG:4326", always_xy=True)
    # a Lambert-93 easting/northing well outside the lon/lat degree range
    east, north = 600_000.0, 6_300_000.0
    lat, lon = pastis._geometry_latlon(_square(east, north))
    exp_lon, exp_lat = t.transform(east, north)
    assert np.isfinite(lat) and np.isfinite(lon)
    assert lat == pytest.approx(exp_lat, abs=1e-6)
    assert lon == pytest.approx(exp_lon, abs=1e-6)
    # sanity: lands in metropolitan France
    assert 41.0 <= lat <= 51.5 and -5.5 <= lon <= 9.5


def test_wgs84_geometry_is_kept_as_is():
    # synthetic fixtures declare geometries directly in degrees; must pass through unchanged
    lat, lon = pastis._geometry_latlon(_square(2.5, 48.8, half=0.01))
    assert lat == pytest.approx(48.8) and lon == pytest.approx(2.5)


def test_empty_geometry_is_nan():
    assert all(np.isnan(v) for v in pastis._geometry_latlon(None))
    assert all(np.isnan(v) for v in pastis._geometry_latlon({"type": "Polygon", "coordinates": []}))


# --------------------------------------------------------------------------- #
# Full loader integration (Lambert-93 metadata + TILE property)
# --------------------------------------------------------------------------- #
def _make_pastis_l93(base, patches):
    """Minimal PASTIS release: metadata.geojson with Lambert-93 geometry + TILE + the .npy files."""
    for d in ("DATA_S2", "DATA_S1A", "ANNOTATIONS"):
        (base / d).mkdir(parents=True, exist_ok=True)
    feats = []
    for pid, fold, tile, east, north in patches:
        feats.append({
            "type": "Feature",
            "properties": {
                "ID_PATCH": pid, "Fold": fold, "TILE": tile,
                "dates-S2": {"0": 20190115, "1": 20190215},
                "dates-S1A": {"0": 20190110, "1": 20190210},
            },
            "geometry": _square(east, north),
        })
        np.save(base / "DATA_S2" / f"S2_{pid}.npy", np.ones((2, 10, 128, 128), dtype=np.int16))
        np.save(base / "DATA_S1A" / f"S1A_{pid}.npy", np.ones((2, 3, 128, 128), dtype=np.float16))
        target = np.zeros((3, 128, 128), dtype=np.uint8)
        target[0, :64, :64] = 1
        np.save(base / "ANNOTATIONS" / f"TARGET_{pid}.npy", target)
    (base / "metadata.geojson").write_text(json.dumps({"type": "FeatureCollection", "features": feats}))


def test_loader_stores_tile_and_finite_transformed_coords(tmp_path):
    base = tmp_path / "pastis"
    # four patches over two tiles, Lambert-93 easting/northing (raw TILE unprefixed on some)
    patches = [
        (1001, 1, "30UXV", 610_000.0, 6_650_000.0),
        (1002, 2, "T30UXV", 611_000.0, 6_651_000.0),
        (1003, 3, "31TFM", 640_000.0, 6_300_000.0),
        (1004, 4, "T31TFM", 641_000.0, 6_301_000.0),
    ]
    _make_pastis_l93(base, patches)
    bench = pastis.load_benchmark(root=tmp_path, shuffle=False)

    by_id = {p.patch_id: p for p in bench.patches}
    assert set(by_id) == {1001, 1002, 1003, 1004}
    # tiles canonicalized (both "30UXV" and "T30UXV" -> "T30UXV")
    assert by_id[1001].tile == by_id[1002].tile == "T30UXV"
    assert by_id[1003].tile == by_id[1004].tile == "T31TFM"
    # coordinates transformed and finite (NOT NaN, the old bug), inside metropolitan France
    for p in bench.patches:
        lat, lon = p.latlon
        assert np.isfinite(lat) and np.isfinite(lon), p.patch_id
        assert 41.0 <= lat <= 51.5 and -5.5 <= lon <= 9.5

    assert bench.tiles() == ["T30UXV", "T31TFM"]
    assert bench.patch_tiles == {1001: "T30UXV", 1002: "T30UXV", 1003: "T31TFM", 1004: "T31TFM"}


def test_missing_tile_metadata_is_not_geographic_ready(tmp_path):
    """A patch with no TILE loads (tile=None) but is NOT a geographic-ready state: split generation
    must hard-fail via assert_geographic_ready rather than silently drop it."""
    base = tmp_path / "pastis"
    for d in ("DATA_S2", "DATA_S1A", "ANNOTATIONS"):
        (base / d).mkdir(parents=True, exist_ok=True)
    feat = {
        "type": "Feature",
        "properties": {"ID_PATCH": 7, "Fold": 1,
                       "dates-S2": {"0": 20190115}, "dates-S1A": {"0": 20190110}},
        "geometry": _square(2.5, 48.8, half=0.01),  # degrees, finite coords
    }
    np.save(base / "DATA_S2" / "S2_7.npy", np.ones((1, 10, 128, 128), dtype=np.int16))
    np.save(base / "DATA_S1A" / "S1A_7.npy", np.ones((1, 3, 128, 128), dtype=np.float16))
    np.save(base / "ANNOTATIONS" / "TARGET_7.npy", np.zeros((3, 128, 128), dtype=np.uint8))
    (base / "metadata.geojson").write_text(json.dumps({"type": "FeatureCollection", "features": [feat]}))

    bench = pastis.load_benchmark(root=tmp_path, shuffle=False)
    assert bench.patches[0].tile is None and bench.tiles() == []
    with pytest.raises(ValueError, match="no Sentinel tile"):
        pastis.assert_geographic_ready(bench)


def test_assert_geographic_ready_passes_and_rejects(tmp_path):
    base = tmp_path / "pastis"
    _make_pastis_l93(base, [
        (1001, 1, "30UXV", 610_000.0, 6_650_000.0),
        (1002, 2, "T31TFM", 640_000.0, 6_300_000.0),
    ])
    bench = pastis.load_benchmark(root=tmp_path, shuffle=False)
    pastis.assert_geographic_ready(bench)  # all valid -> no raise

    # a patch whose coordinate is NaN (missing geometry) is rejected
    from dataclasses import replace
    broken = replace(bench, patches=(replace(bench.patches[0], latlon=(float("nan"), float("nan"))),
                                     bench.patches[1]))
    with pytest.raises(ValueError, match="non-finite coordinates"):
        pastis.assert_geographic_ready(broken)


# --------------------------------------------------------------------------- #
# Frozen tile universe + counts validation (before geographic/spatial splits)
# --------------------------------------------------------------------------- #
def _p(pid, tile):
    return SimpleNamespace(patch_id=pid, tile=tile, latlon=(46.0, 2.0))


def _bench(patches):
    return SimpleNamespace(patches=patches)


def test_frozen_tile_counts_are_exactly_the_spec_numbers():
    assert S.PASTIS_TILE_PATCHES == {"T30UXV": 531, "T31TFJ": 623, "T31TFM": 723, "T32ULU": 556}


def test_frozen_tile_universe_accepts_exact_match():
    exp = {"T30UXV": 2, "T31TFM": 3}
    patches = [_p(1, "T30UXV"), _p(2, "T30UXV"), _p(3, "T31TFM"), _p(4, "T31TFM"), _p(5, "T31TFM")]
    pastis.assert_frozen_tile_universe(_bench(patches), expected=exp)  # no raise


def test_frozen_tile_universe_rejects_wrong_count():
    exp = {"T30UXV": 2, "T31TFM": 3}
    patches = [_p(1, "T30UXV"), _p(2, "T31TFM"), _p(3, "T31TFM"), _p(4, "T31TFM")]  # T30UXV=1 != 2
    with pytest.raises(ValueError, match=r"T30UXV: 1 patch\(es\) != expected 2"):
        pastis.assert_frozen_tile_universe(_bench(patches), expected=exp)


def test_frozen_tile_universe_rejects_unexpected_tile():
    patches = [_p(1, "T30UXV"), _p(2, "T31TFJ")]
    with pytest.raises(ValueError, match="unexpected tile"):
        pastis.assert_frozen_tile_universe(_bench(patches), expected={"T30UXV": 1})


def test_frozen_tile_universe_rejects_missing_tile():
    patches = [_p(1, "T30UXV"), _p(2, None)]
    with pytest.raises(ValueError, match="no tile"):
        pastis.assert_frozen_tile_universe(_bench(patches), expected={"T30UXV": 1})


def test_frozen_tile_universe_rejects_duplicate_patch_id():
    patches = [_p(1, "T30UXV"), _p(1, "T30UXV")]  # duplicate patch id 1
    with pytest.raises(ValueError, match="duplicate patch id"):
        pastis.assert_frozen_tile_universe(_bench(patches), expected={"T30UXV": 2})
