"""Frozen-run result manifest: exact-match resume, readable diffs, embedding-identity binding."""

from __future__ import annotations

import pytest

from utils import cacheutils as C
from utils import perfutils as perf
from utils import runstate


def _m(**over):
    base = {
        "schema": 1,
        "final_commit": None,
        "benchmark": "cropharvest",
        "model": "galileo",
        "embedding": {"artifact": "baseline", "digest": "abc"},
        "seeds": [0, 1],
        "regimes": ["official"],
        "probe_cap": 0,
    }
    base.update(over)
    return base


def test_exact_match_resumes(tmp_path):
    m = _m()
    runstate.publish_run_manifest(tmp_path, m)
    runstate.check_run_manifest(tmp_path, m, overwrite_mode=False)  # no raise


def test_seed_change_refuses_with_readable_diff(tmp_path):
    runstate.publish_run_manifest(tmp_path, _m(seeds=[0, 1]))
    with pytest.raises(RuntimeError, match=r"seeds: \[0, 1\] != \[0, 1, 2\]"):
        runstate.check_run_manifest(tmp_path, _m(seeds=[0, 1, 2]), overwrite_mode=False)


def test_regime_change_refuses(tmp_path):
    runstate.publish_run_manifest(tmp_path, _m())
    with pytest.raises(RuntimeError, match="regimes:"):
        runstate.check_run_manifest(tmp_path, _m(regimes=["official", "geographic_ood"]), overwrite_mode=False)


def test_embedding_identity_binding(tmp_path):
    """A change in the embedding (its recorded content digest) flips the run manifest -> refuse."""
    runstate.publish_run_manifest(tmp_path, _m(embedding={"artifact": "baseline", "digest": "abc"}))
    with pytest.raises(RuntimeError, match=r"embedding\.digest: 'abc' != 'zzz'"):
        runstate.check_run_manifest(
            tmp_path, _m(embedding={"artifact": "baseline", "digest": "zzz"}), overwrite_mode=False
        )


def test_probe_cap_change_refuses(tmp_path):
    runstate.publish_run_manifest(tmp_path, _m(probe_cap=0))
    with pytest.raises(RuntimeError, match="probe_cap:"):
        runstate.check_run_manifest(tmp_path, _m(probe_cap=25000), overwrite_mode=False)


def test_rows_without_manifest_refuses(tmp_path):
    (tmp_path / "probe_results.jsonl").write_text('{"x":1}\n')
    with pytest.raises(RuntimeError, match="NO run_manifest.json"):
        runstate.check_run_manifest(tmp_path, _m(), overwrite_mode=False)


def test_overwrite_mode_bypasses(tmp_path):
    runstate.publish_run_manifest(tmp_path, _m(seeds=[0]))
    runstate.check_run_manifest(tmp_path, _m(seeds=[9]), overwrite_mode=True)  # no raise


def test_digest_is_deterministic(tmp_path):
    a = runstate.run_manifest_digest(_m())
    b = runstate.run_manifest_digest(_m())
    assert a == b and len(a) == 64
    assert runstate.run_manifest_digest(_m(seeds=[0])) != a


def test_build_run_manifest_wires_embedding_and_probe_cap(tmp_path, monkeypatch):
    monkeypatch.setattr(C, "DATASET_DIGEST_DIR", tmp_path / "dd")
    (tmp_path / "dd").mkdir()
    monkeypatch.setattr(C, "_FROZEN_IDENTITY", {"final_commit": "deadbeef", "clean": True, "tree_identity": "t"})
    monkeypatch.setattr(perf, "PROBE_CAP", 50000)
    man = runstate.build_run_manifest(
        "galileo", "cropharvest", "baseline", "EMB_DIGEST_123",
        ["official"], [0, 1, 2], {"device": "cpu"},
        active_probes=["logistic"], budget_regimes=[1.0], max_dense_pixels=0, write_predictions=True,
    )
    assert man["embedding"] == {"artifact": "baseline", "digest": "EMB_DIGEST_123"}
    assert man["final_commit"] == "deadbeef" and man["seeds"] == [0, 1, 2]
    assert man["probe_cap"] == 50000  # PROBE_CAP is recorded in the RUN manifest, not embedding identity
    assert "device" not in man["enc"]  # device is not result-defining
