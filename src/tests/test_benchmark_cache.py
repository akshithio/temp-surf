"""Canonical benchmark-object pickle cache.

One fixed path per benchmark (``data/cache/benchmark/<bench>.pkl``), always built with the frozen
production loader contract (max_samples=None, shuffle=True, seed=0). No tagging, fingerprinting,
loader kwargs, or per-parameter fallback files.
"""

from __future__ import annotations

import inspect
import threading
import time
from types import SimpleNamespace

import pytest

from utils import cacheutils as C

BENCHES = ["cropharvest", "eurocropsml", "breizhcrops", "pastis"]


@pytest.fixture
def cache(tmp_path, monkeypatch):
    """Point the cache at a temp dir and stub the loader with a call counter."""
    monkeypatch.setattr(C, "CACHE_DIR", tmp_path)
    calls = {"n": 0}

    def fake_get_input(name, root=None, **kwargs):
        calls["n"] += 1
        assert kwargs == C._CANONICAL_LOADER  # always the fixed production contract
        return SimpleNamespace(name=name, payload=list(range(3)))

    monkeypatch.setattr(C.GI, "get_input", fake_get_input)
    return SimpleNamespace(dir=tmp_path / "benchmark", calls=calls)


def _tmp_residue(d):
    """Leftover atomic temp pickles (.<name>.<pid>.<uuid>.tmp) -- the .lock file is NOT one."""
    return sorted(p.name for p in d.glob("*.tmp")) + sorted(p.name for p in d.glob(".*.tmp"))


def test_canonical_path_for_all_four_benchmarks(cache):
    for b in BENCHES:
        assert C.benchmark_cache_path(b) == cache.dir / f"{b}.pkl"


def test_unknown_benchmark_rejected(cache):
    with pytest.raises(KeyError):
        C.benchmark_cache_path("not_a_benchmark")


def test_cache_hit_avoids_loading_source_again(cache):
    first = C.cached_bench("cropharvest")
    second = C.cached_bench("cropharvest")
    assert cache.calls["n"] == 1  # source loaded exactly once
    assert first.name == second.name == "cropharvest"


def test_missing_cache_builds_atomically(cache):
    bench = C.cached_bench("pastis")
    path = C.benchmark_cache_path("pastis")
    assert path.exists() and bench.name == "pastis"
    assert _tmp_residue(cache.dir) == []  # no leftover temporary pickle


def test_corrupt_pickle_rebuilds_safely(cache, capsys):
    path = C.benchmark_cache_path("breizhcrops")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"not a valid pickle")
    bench = C.cached_bench("breizhcrops")
    out = capsys.readouterr().out
    assert bench.name == "breizhcrops"
    assert cache.calls["n"] == 1  # rebuilt from source
    assert "unreadable" in out and str(path) in out  # reported clearly (path + exception)
    assert _tmp_residue(cache.dir) == []


def test_lock_file_persists_and_no_tmp_remains(cache):
    C.cached_bench("eurocropsml")
    names = {p.name for p in cache.dir.iterdir()}
    assert "eurocropsml.pkl" in names
    # A persistent advisory ``.eurocropsml.pkl.lock`` is acceptable and expected -- assert only that
    # no temporary pickle survives, never that the lock file is absent.
    assert _tmp_residue(cache.dir) == []


def test_concurrent_cold_callers_produce_one_pickle(cache, monkeypatch):
    orig = C.GI.get_input

    def slow_get_input(name, root=None, **kwargs):
        time.sleep(0.05)  # widen the contention window so writers genuinely race
        return orig(name, root=root, **kwargs)

    monkeypatch.setattr(C.GI, "get_input", slow_get_input)

    results = []
    barrier = threading.Barrier(6)

    def worker():
        barrier.wait()
        results.append(C.cached_bench("cropharvest"))

    threads = [threading.Thread(target=worker) for _ in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert cache.calls["n"] == 1  # built exactly once despite the race
    assert all(r.name == "cropharvest" for r in results) and len(results) == 6
    assert [p.name for p in cache.dir.glob("*.pkl")] == ["cropharvest.pkl"]
    assert _tmp_residue(cache.dir) == []


def test_no_legacy_tagged_filename_is_generated(cache):
    C.cached_bench("cropharvest")
    names = [p.name for p in cache.dir.glob("*.pkl")]
    assert names == ["cropharvest.pkl"]
    assert all("__" not in n for n in names)  # no params/code/data tag suffix


def test_api_is_kwarg_less_and_rejects_noncanonical_settings(cache):
    sig = inspect.signature(C.cached_bench)
    assert list(sig.parameters) == ["benchmark"]
    assert not any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
    with pytest.raises(TypeError):
        C.cached_bench("cropharvest", max_samples=100)  # subsets must use dataio.get_input directly
    assert not C.benchmark_cache_path("cropharvest").exists()  # nothing was written


def test_tagging_and_fingerprint_subsystem_removed():
    for gone in (
        "bench_tag",
        "_input_fingerprint",
        "_pastis_input_fingerprint",
        "_benchmark_input_fingerprint",
        "_hash_files",
        "_update_file_content_hash",
    ):
        assert not hasattr(C, gone), f"{gone} should have been deleted"
