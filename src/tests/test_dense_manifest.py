"""Frozen-run dense PASTIS cache: slim cache.json record, exact completeness, resumable, refuse-on-mismatch."""

from __future__ import annotations

import numpy as np
import pytest

from utils import cacheutils as C


class FakeDenseTile:
    def __init__(self, n):
        self._n = n

    def pixel_benchmark(self):
        return self


class FakeDensePatch:
    def __init__(self, patch_id, fold, target_path):
        self.patch_id, self.fold, self.target_path = patch_id, fold, target_path


class FakeDenseBench:
    """One patch, tile_size 64 -> 2x2 = 4 non-void tiles (target is all label 0)."""

    def __init__(self, target_path):
        self.tile_size = 64
        self.ignore_index = 255
        self.patches = (FakeDensePatch(7, 1, target_path),)

    def iter_tiles(self, cache_root=None, overwrite=False):
        for r in range(2):
            for c in range(2):
                tile_id = f"7_{r}_{c}"
                if cache_root is not None and not overwrite:
                    fp = cache_root / "fold_1" / f"{tile_id}.npy"
                    lp = cache_root / "fold_1" / f"{tile_id}.labels.npy"
                    if fp.exists() and lp.exists():
                        continue
                yield tile_id, 1, FakeDenseTile(10), np.arange(10, dtype=np.int64) % 3


class FakeDenseModel:
    embedding_dim = 4

    def encode_dense(self, tile):
        return np.zeros((tile._n, 4), np.float32)

    def compute_macs(self):
        return 0


@pytest.fixture
def dense_env(tmp_path, monkeypatch):
    monkeypatch.setattr(C, "EMBEDDINGS_DIR", tmp_path / "emb")
    monkeypatch.setattr(C, "CACHE_JSON_PATH", tmp_path / "logs" / "cache.json")
    C.update_cache(datasets={"pastis": "p" * 64})
    monkeypatch.setattr(C, "build_model", lambda name, **k: FakeDenseModel())
    tpath = tmp_path / "target.npy"
    np.save(tpath, np.zeros((1, 128, 128), np.uint8))
    return tmp_path, FakeDenseBench(tpath)


def test_dense_roundtrip_slim_record(dense_env):
    _tmp, bench = dense_env
    C.extract_dense_and_cache(bench, "pastis", "fakemodel", "baseline")
    m = C._cache_record("pastis", "fakemodel", "baseline")
    assert m["feature_tile_count"] == 4 and m["label_tile_count"] == 4 and m["feature_dim"] == 4
    assert len(m["tile_set_digest"]) == 64
    assert "expected_feature_tiles" not in m and "aggregate_feature_digest" not in m  # no arrays / content hashes
    assert C.require_dense_cache(bench, "pastis", "fakemodel", "baseline") == C.dense_embedding_cache_dir("pastis", "fakemodel", "baseline")


def test_dense_require_absent_is_not_built(dense_env):
    _tmp, bench = dense_env
    with pytest.raises(C.MissingEmbeddingCache, match="not built"):
        C.require_dense_cache(bench, "pastis", "fakemodel", "baseline")


def test_dense_require_rejects_missing_tile(dense_env):
    _tmp, bench = dense_env
    root = C.extract_dense_and_cache(bench, "pastis", "fakemodel", "baseline")
    (root / "fold_1" / "7_0_0.npy").unlink()
    with pytest.raises(C.MissingEmbeddingCache, match="missing"):
        C.require_dense_cache(bench, "pastis", "fakemodel", "baseline")


def test_dense_require_rejects_extra_stale_tile(dense_env):
    _tmp, bench = dense_env
    root = C.extract_dense_and_cache(bench, "pastis", "fakemodel", "baseline")
    np.save(root / "fold_1" / "999_0_0.npy", np.zeros((10, 4), np.float32))
    np.save(root / "fold_1" / "999_0_0.labels.npy", np.zeros((10,), np.uint8))
    with pytest.raises(C.MissingEmbeddingCache, match="UNEXPECTED"):
        C.require_dense_cache(bench, "pastis", "fakemodel", "baseline")


def test_dense_require_rejects_extra_feature_only_tile(dense_env):
    # A stray FEATURE file with no matching label -- the earlier label-only glob missed this entirely.
    _tmp, bench = dense_env
    root = C.extract_dense_and_cache(bench, "pastis", "fakemodel", "baseline")
    np.save(root / "fold_1" / "999_0_0.npy", np.zeros((10, 4), np.float32))
    with pytest.raises(C.MissingEmbeddingCache, match="UNEXPECTED feature"):
        C.require_dense_cache(bench, "pastis", "fakemodel", "baseline")


def test_dense_require_rejects_tampered_feature_dim(dense_env):
    _tmp, bench = dense_env
    C.extract_dense_and_cache(bench, "pastis", "fakemodel", "baseline")
    record = C._cache_record("pastis", "fakemodel", "baseline")
    record["feature_dim"] = 999
    C.update_cache(embeddings={C._embedding_key("pastis", "fakemodel", "baseline"): record})
    with pytest.raises(C.MissingEmbeddingCache, match="feature_dim"):
        C.require_dense_cache(bench, "pastis", "fakemodel", "baseline")


def test_dense_require_rejects_dataset_digest_change(dense_env):
    _tmp, bench = dense_env
    C.extract_dense_and_cache(bench, "pastis", "fakemodel", "baseline")
    C.update_cache(datasets={"pastis": "q" * 64})
    with pytest.raises(C.MissingEmbeddingCache, match="dataset_digest"):
        C.require_dense_cache(bench, "pastis", "fakemodel", "baseline")


def test_dense_extract_resumes_after_interruption(dense_env):
    _tmp, bench = dense_env
    root = C.extract_dense_and_cache(bench, "pastis", "fakemodel", "baseline")
    # Simulate a crash BEFORE the record, having lost one tile: absent record = incomplete.
    doc = C._read_cache_doc()
    doc["embeddings"].pop(C._embedding_key("pastis", "fakemodel", "baseline"), None)
    C._atomic_write_json(C.CACHE_JSON_PATH, doc)
    (root / "fold_1" / "7_0_0.npy").unlink()
    (root / "fold_1" / "7_0_0.labels.npy").unlink()
    # Re-run resumes: re-encodes only the missing tile, then records completion.
    C.extract_dense_and_cache(bench, "pastis", "fakemodel", "baseline")
    C.require_dense_cache(bench, "pastis", "fakemodel", "baseline")
