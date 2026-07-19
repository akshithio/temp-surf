"""The one shared cache-metadata file (data/logs/cache.json): creation, atomic merge under a single
lock, run-provenance digest lookup, and the absence of the retired sidecar/dataset-digest helpers."""

from __future__ import annotations

import json
import threading

import pytest

from utils import cacheutils as C


@pytest.fixture
def cache_json(tmp_path, monkeypatch):
    path = tmp_path / "logs" / "cache.json"
    monkeypatch.setattr(C, "CACHE_JSON_PATH", path)
    return path


def test_absent_cache_reads_as_empty_doc(cache_json):
    doc = C._read_cache_doc()
    assert doc["datasets"] == {} and doc["embeddings"] == {}
    assert not cache_json.exists()  # reading never creates the file


def test_update_cache_creates_file_and_persists(cache_json):
    C.update_cache(datasets={"cropharvest": "d" * 64})
    assert cache_json.exists()
    assert json.loads(cache_json.read_text())["datasets"]["cropharvest"] == "d" * 64
    assert C.dataset_digest("cropharvest") == "d" * 64


def test_dataset_digest_absent_is_hard_error(cache_json):
    with pytest.raises(C.MissingEmbeddingCache, match="No frozen dataset digest"):
        C.dataset_digest("cropharvest")


def test_corrupt_cache_is_hard_error_not_silent_reset(cache_json):
    cache_json.parent.mkdir(parents=True)
    cache_json.write_text("{ not json")
    with pytest.raises(json.JSONDecodeError):
        C._read_cache_doc()  # a shared metadata file is never silently reset


def test_update_cache_merges_not_clobbers(cache_json):
    # Each writer read-modify-writes the WHOLE doc, so a later writer preserves earlier entries
    # (datasets and embedding records from other cells) rather than replacing the file wholesale.
    C.update_cache(datasets={"cropharvest": "a" * 64})
    C.update_cache(embeddings={"cropharvest/raw/baseline": {"artifact_sha256": "s1"}})
    C.update_cache(embeddings={"pastis/galileo/baseline": {"tile_set_digest": "s2"}})
    C.update_cache(datasets={"pastis": "b" * 64})

    doc = C._read_cache_doc()
    assert doc["datasets"] == {"cropharvest": "a" * 64, "pastis": "b" * 64}
    assert set(doc["embeddings"]) == {"cropharvest/raw/baseline", "pastis/galileo/baseline"}


def test_concurrent_writers_all_survive(cache_json):
    # The single lock + read-modify-write is what stops concurrent model writers from dropping each
    # other's records: N threads each add a distinct cell under maximal contention; all must survive.
    n = 8
    barrier = threading.Barrier(n)

    def writer(i):
        barrier.wait()
        C.update_cache(embeddings={f"bench/model{i}/baseline": {"artifact_sha256": f"s{i}"}})

    threads = [threading.Thread(target=writer, args=(i,)) for i in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert set(C._read_cache_doc()["embeddings"]) == {f"bench/model{i}/baseline" for i in range(n)}


def test_update_cache_is_atomic_on_failure(cache_json, monkeypatch):
    # A failed replacement leaves the prior document intact (temp-file + os.replace): no partial or
    # empty cache.json can survive a crash mid-write.
    C.update_cache(datasets={"cropharvest": "d" * 64})
    prior = cache_json.read_text()
    monkeypatch.setattr(C.os, "replace", lambda *a, **k: (_ for _ in ()).throw(OSError("disk full")))
    with pytest.raises(OSError, match="disk full"):
        C.update_cache(datasets={"eurocropsml": "e" * 64})
    assert cache_json.read_text() == prior


def test_embedding_digest_lookup_for_run_provenance(cache_json):
    # build_run_manifest binds results to the embedding via embedding_digest(): the artifact SHA for
    # tabular cells, the tile-set digest for dense cells, and None when the cell is absent.
    C.update_cache(embeddings={
        "cropharvest/raw/baseline": {"artifact_sha256": "TAB_SHA"},
        "pastis/galileo/baseline": {"tile_set_digest": "DENSE_DIGEST"},
    })
    assert C.embedding_digest("cropharvest", "raw", "baseline") == "TAB_SHA"
    assert C.embedding_digest("pastis", "galileo", "baseline", dense=True) == "DENSE_DIGEST"
    assert C.embedding_digest("cropharvest", "olmoearth", "baseline") is None


def test_retired_sidecar_and_digest_helpers_are_gone():
    # No production code path depends on the per-cell baseline.manifest.json sidecars or the
    # per-benchmark dataset_digests/*.txt files any more.
    for name in ("embedding_manifest_path", "dense_manifest_path", "_read_manifest",
                 "_write_manifest", "_build_dense_manifest", "DATASET_DIGEST_DIR"):
        assert not hasattr(C, name), f"{name} should be removed"
