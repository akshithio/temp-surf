"""Frozen-run tabular embedding cache: fixed paths, cache.json identity, refuse-on-mismatch."""

from __future__ import annotations

import numpy as np
import pytest

from utils import cacheutils as C


class FakeModel:
    embedding_dim = 4

    def __init__(self, arr):
        self.arr = arr

    def encode(self, bench):
        return self.arr

    def compute_macs(self):
        return 0


class FakeBench:
    def __init__(self, n, prefix="s"):
        self.sample_ids = np.array([f"{prefix}{i}" for i in range(n)], dtype=object)

    @property
    def n_samples(self):
        return len(self.sample_ids)


@pytest.fixture
def cache_env(tmp_path, monkeypatch):
    monkeypatch.setattr(C, "EMBEDDINGS_DIR", tmp_path / "emb")
    monkeypatch.setattr(C, "CACHE_JSON_PATH", tmp_path / "logs" / "cache.json")
    C.update_cache(datasets={"cropharvest": "d" * 64})
    return tmp_path


def _patch_model(monkeypatch, arr):
    monkeypatch.setattr(C, "build_model", lambda name, **k: FakeModel(arr))


def test_extract_then_load_roundtrip_fixed_path(cache_env, monkeypatch):
    bench = FakeBench(3)
    arr = np.arange(12, dtype=np.float32).reshape(3, 4)
    _patch_model(monkeypatch, arr)

    out = C.extract_and_cache(bench, "cropharvest", "fakemodel", "baseline")
    assert np.array_equal(out, arr)

    art = C.embedding_cache_path("cropharvest", "fakemodel", "baseline")
    assert art.name == "baseline.npy" and art.exists()

    record = C._cache_record("cropharvest", "fakemodel", "baseline")
    assert record["shape"] == [3, 4] and record["dtype"] == "float32"
    assert len(record["artifact_sha256"]) == 64 and len(record["checkpoint_sha256"]) == 64
    assert "sample_ids_digest" in record and "byte_size" not in record  # slim record

    assert np.array_equal(C.load_cached_embeddings(bench, "cropharvest", "fakemodel", "baseline"), arr)


def test_load_absent_is_not_built(cache_env):
    bench = FakeBench(3)
    with pytest.raises(C.MissingEmbeddingCache, match="not built"):
        C.load_cached_embeddings(bench, "cropharvest", "fakemodel", "baseline")


def test_load_rejects_wrong_sample_order(cache_env, monkeypatch):
    bench = FakeBench(3)
    _patch_model(monkeypatch, np.zeros((3, 4), np.float32))
    C.extract_and_cache(bench, "cropharvest", "fakemodel", "baseline")
    other = FakeBench(3, prefix="z")  # same count, different IDs
    with pytest.raises(C.MissingEmbeddingCache, match="sample_ids_digest"):
        C.load_cached_embeddings(other, "cropharvest", "fakemodel", "baseline")


def test_load_rejects_dataset_digest_change(cache_env, monkeypatch):
    bench = FakeBench(2)
    _patch_model(monkeypatch, np.zeros((2, 4), np.float32))
    C.extract_and_cache(bench, "cropharvest", "fakemodel", "baseline")
    C.update_cache(datasets={"cropharvest": "e" * 64})  # inputs changed underneath
    with pytest.raises(C.MissingEmbeddingCache, match="dataset_digest"):
        C.load_cached_embeddings(bench, "cropharvest", "fakemodel", "baseline")


def test_load_rejects_checkpoint_change(cache_env, monkeypatch):
    bench = FakeBench(2)
    _patch_model(monkeypatch, np.zeros((2, 4), np.float32))
    C.extract_and_cache(bench, "cropharvest", "fakemodel", "baseline")
    monkeypatch.setattr(C, "checkpoint_sha256", lambda *a, **k: "f" * 64)  # a different checkpoint
    with pytest.raises(C.MissingEmbeddingCache, match="checkpoint_sha256"):
        C.load_cached_embeddings(bench, "cropharvest", "fakemodel", "baseline")


def test_load_rejects_shape_mismatch(cache_env, monkeypatch):
    bench = FakeBench(3)
    _patch_model(monkeypatch, np.zeros((3, 4), np.float32))
    C.extract_and_cache(bench, "cropharvest", "fakemodel", "baseline")
    art = C.embedding_cache_path("cropharvest", "fakemodel", "baseline")
    with open(art, "wb") as f:  # a different-shaped array under the same path
        np.save(f, np.zeros((3, 5), np.float32))
    with pytest.raises(C.MissingEmbeddingCache, match="shape"):
        C.load_cached_embeddings(bench, "cropharvest", "fakemodel", "baseline")


def test_extract_refuses_to_replace_a_mismatched_cache(cache_env, monkeypatch):
    """The frozen run never auto-replaces a completed artifact -- it refuses and asks the operator."""
    bench = FakeBench(2)
    _patch_model(monkeypatch, np.zeros((2, 4), np.float32))
    C.extract_and_cache(bench, "cropharvest", "fakemodel", "baseline")
    C.update_cache(datasets={"cropharvest": "e" * 64})  # inputs changed underneath
    with pytest.raises(C.MissingEmbeddingCache, match="REFUSING|Delete the leaf"):
        C.extract_and_cache(bench, "cropharvest", "fakemodel", "baseline")


def test_record_is_written_only_after_the_array(cache_env, monkeypatch):
    """If publication fails at the record step, there is an array but NO cache.json record, so the
    cache reads as incomplete (not built) rather than certified."""
    bench = FakeBench(2)
    _patch_model(monkeypatch, np.zeros((2, 4), np.float32))
    monkeypatch.setattr(C, "update_cache", lambda **k: (_ for _ in ()).throw(RuntimeError("boom")))
    with pytest.raises(RuntimeError, match="boom"):
        C.extract_and_cache(bench, "cropharvest", "fakemodel", "baseline")
    assert C.embedding_cache_path("cropharvest", "fakemodel", "baseline").exists()
    assert C._cache_record("cropharvest", "fakemodel", "baseline") is None
    with pytest.raises(C.MissingEmbeddingCache, match="not built"):
        C.load_cached_embeddings(bench, "cropharvest", "fakemodel", "baseline")


def test_checkpoint_sha256_is_full_and_dir_aware(tmp_path):
    f = tmp_path / "ckpt.pt"
    f.write_bytes(b"weights")
    assert len(C.checkpoint_sha256("agrifm", weights_override=str(f))) == 64

    d = tmp_path / "modeldir"
    (d / "sub").mkdir(parents=True)
    (d / "config.json").write_bytes(b"{}")
    (d / "sub" / "weights.pth").write_bytes(b"w")
    got = C.checkpoint_sha256("olmoearth", weights_override=str(d))
    assert len(got) == 64
    (d / "config.json").write_bytes(b'{"x":1}')  # content change -> identity moves
    C._CHECKPOINT_SHA_CACHE.clear()
    assert C.checkpoint_sha256("olmoearth", weights_override=str(d)) != got
