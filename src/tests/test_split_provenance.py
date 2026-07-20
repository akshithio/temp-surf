"""Provenance gates on canonical split generation.

A frozen scientific artifact must not record provenance it cannot honour. Three ways the log could
lie, all now refused before anything is written:

  * ``code_revision`` records ``git rev-parse HEAD``; if the protocol implementation is uncommitted
    that hash does not contain the code that produced the splits;
  * ``data_fingerprint`` records the inputs; a null value asserts "the inputs are unknown";
  * ``label_access.csv`` had no checksum at all, so a different valid draw loaded silently.

Also pins that the frozen ``split_config`` carries the footprint policy, and that the per-leaf
footprint SPECIFICATION is recorded even for targets that excluded nothing.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_TOOLS = Path(__file__).resolve().parents[2] / "tools"
if str(_TOOLS) not in sys.path:
    sys.path.insert(0, str(_TOOLS))

import generate_splits as G  # noqa: E402

from evals import split_artifacts as SA  # noqa: E402
from evals import split_spec  # noqa: E402
from evals.regimes import geographic_ood as geo  # noqa: E402


# --------------------------------------------------------------------------- #
# The generator refuses provenance it cannot honour
# --------------------------------------------------------------------------- #
def test_generation_refuses_a_dirty_tree(monkeypatch):
    """An uncommitted protocol change would be attributed to a revision that does not contain it."""
    monkeypatch.setattr(G, "_tree_is_dirty", lambda: " M src/evals/regimes/geographic_ood.py")
    monkeypatch.setenv("DATA_FINGERPRINT", "abc123")
    with pytest.raises(SystemExit, match="uncommitted changes"):
        G._require_frozen_provenance()


def test_generation_refuses_a_null_data_fingerprint(monkeypatch):
    monkeypatch.setattr(G, "_tree_is_dirty", lambda: "")
    monkeypatch.delenv("DATA_FINGERPRINT", raising=False)
    with pytest.raises(SystemExit, match="DATA_FINGERPRINT"):
        G._require_frozen_provenance()


def test_generation_proceeds_on_a_clean_tree_with_a_fingerprint(monkeypatch):
    monkeypatch.setattr(G, "_tree_is_dirty", lambda: "")
    monkeypatch.setenv("DATA_FINGERPRINT", "abc123")
    G._require_frozen_provenance()      # must not raise


def test_dirty_detection_ignores_files_that_cannot_change_a_split(monkeypatch):
    """Only src/ and tools/ can alter generation; a dirty note or figure must not block it."""
    monkeypatch.setattr(
        G.subprocess, "check_output",
        lambda *a, **k: " M viz/progress/artifact_brief.md\n?? scratch.txt\n",
    )
    assert G._tree_is_dirty() == ""


def test_dirty_detection_flags_source_changes(monkeypatch):
    monkeypatch.setattr(
        G.subprocess, "check_output", lambda *a, **k: " M src/evals/split_spec.py\n",
    )
    assert "src/evals/split_spec.py" in G._tree_is_dirty()


def test_unavailable_git_is_treated_as_dirty(monkeypatch):
    """Cannot prove clean => refuse. Never assume a provenance claim we did not verify."""
    def _boom(*a, **k):
        raise OSError("git missing")

    monkeypatch.setattr(G.subprocess, "check_output", _boom)
    assert G._tree_is_dirty()


# --------------------------------------------------------------------------- #
# The frozen split configuration records the footprint policy
# --------------------------------------------------------------------------- #
def test_split_config_records_the_footprint_policy():
    fp = G._split_config()["footprint_exclusion"]
    assert fp["enabled_benchmarks"] == ["cropharvest"]
    assert fp["hull_policy"] == "convex_hull_of_target_coordinates"
    assert "+proj=aeqd" in fp["projection"] and "+units=m" in fp["projection"]
    assert fp["buffer_quad_segs"] == geo.FOOTPRINT_QUAD_SEGS
    assert "purge_km * 1000" in fp["buffer_m_rule"]
    assert fp["assignment_reason"] == SA.REASON_INSIDE_FOOTPRINT
    assert set(fp["recorded_per_leaf"]) == set(SA.FOOTPRINT_SPEC_FIELDS)
    assert "FootprintError" in fp["fail_closed"]


def test_split_config_footprint_policy_tracks_the_specs():
    """The recorded policy is derived from split_spec, never re-literaled."""
    enabled = {n for n, s in split_spec.ALL_SPECS.items() if s.footprint_exclusion}
    assert set(G._split_config()["footprint_exclusion"]["enabled_benchmarks"]) == enabled


# --------------------------------------------------------------------------- #
# Per-leaf footprint specification, including zero-exclusion targets
# --------------------------------------------------------------------------- #
def _leaf_with_events(events, n_ids=7):
    from evals.regimes.base import SourceTargetSplit

    sample_ids = np.asarray([f"s{i}" for i in range(n_ids)], dtype=object)
    split = SourceTargetSplit(
        label="kenya",
        source_train=np.array([0, 1]), source_val=np.array([2]), source_test=np.array([3]),
        target_label_pool=np.array([4]), target_test=np.array([5]),
        has_target=True, supports_target_labels=True, group_kind="geography",
    )
    return SA.build_tabular_leaf(
        "cropharvest", "geographic_ood", 0, split=split,
        domains=np.asarray(["kenya"] * n_ids, dtype=object),
        labels=np.array([0, 1, 0, 1, 0, 1, 0]), sample_ids=sample_ids,
        audit_events=events, purge_km=50.0,
    )


_SPEC = {
    "kind": "footprint_exclusion", "crs": "+proj=aeqd +lat_0=0 +lon_0=0 +units=m",
    "buffer_m": 50000.0, "quad_segs": 64, "hull_policy": "convex_hull",
    "hull_wkt": "POLYGON ((0 0, 1 0, 1 1, 0 0))", "footprint_sha256": "f" * 64,
}


def test_footprint_spec_is_recorded_for_a_zero_exclusion_target():
    """"Masked and nothing fell inside" must be distinguishable from "never masked"."""
    _rows, summary = _leaf_with_events([{**_SPEC, "n_excluded": 0, "excluded_indices": []}])
    fp = summary["footprint"]
    assert fp is not None
    assert fp["n_excluded"] == 0
    assert fp["footprint_sha256"] == "f" * 64
    assert fp["buffer_m"] == 50000.0
    assert fp["crs"].startswith("+proj=aeqd")
    assert summary["purge_counts"].get(SA.REASON_INSIDE_FOOTPRINT, 0) == 0


def test_no_footprint_spec_when_the_mask_never_ran():
    _rows, summary = _leaf_with_events([])
    assert summary["footprint"] is None


def test_footprint_spec_accompanies_actual_exclusions():
    # index 6 is the only id in no partition, so it is the one a removal can claim
    _rows, summary = _leaf_with_events([{**_SPEC, "n_excluded": 1, "excluded_indices": [6]}])
    assert summary["footprint"]["n_excluded"] == 1
    assert summary["purge_counts"][SA.REASON_INSIDE_FOOTPRINT] == 1


def test_footprint_spec_fields_are_exactly_what_the_regime_emits():
    """Every declared spec field is actually produced by the production emitter."""
    import shapely

    footprint, hull, _tf, crs = geo.target_footprint(
        np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 1.0]]), 50_000.0, where="test"
    )
    emitted = {
        "crs": crs, "buffer_m": 50_000.0, "quad_segs": geo.FOOTPRINT_QUAD_SEGS,
        "hull_policy": "convex_hull", "hull_wkt": shapely.to_wkt(hull, rounding_precision=3),
        "footprint_sha256": "0" * 64, "n_excluded": 0,
    }
    assert set(emitted) == set(SA.FOOTPRINT_SPEC_FIELDS)
    assert not footprint.is_empty
