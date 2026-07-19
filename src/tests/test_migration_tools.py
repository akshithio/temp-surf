"""Focused tests for the two TEMPORARY freeze-and-run migration tools.

``tools/adopt_embeddings.py`` (adopt a legacy embedding cache into the canonical layout WITHOUT
regenerating it) and ``tools/preflight_dataset_digests.py`` (portable dataset content digest). These
tools live outside ``src`` and edit a CONFIG block instead of taking CLI args, so the tests import
each module by path and drive its internal functions directly. Everything is synthetic -- no real
benchmark data, no real model, no real checkpoints.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from utils import artifacts

_TOOLS = Path(__file__).resolve().parents[2] / "tools"


def _load_tool(name: str):
    """Import a tools/*.py module fresh (its own CONFIG dict), sharing the live ``utils`` modules."""
    spec = importlib.util.spec_from_file_location(f"_tool_{name}", _TOOLS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ============================ tabular adoption ==============================


class _FakeTabModel:
    """Encodes the passed-through row indices back to their stored rows (+delta), so the spot check
    reproduces the legacy matrix exactly when delta == 0 and diverges otherwise."""

    def __init__(self, arr: np.ndarray, dim: int, delta: float = 0.0):
        self._arr, self.embedding_dim, self._delta = arr, dim, delta

    def encode(self, idx):  # `_subset_bench` is patched to pass the index list straight through
        return self._arr[np.asarray(idx)] + self._delta


def _setup_tabular(adopt, monkeypatch, tmp_path, *, arr, model_dim=None, delta=0.0):
    dim = arr.shape[1] if model_dim is None else model_dim
    bench = SimpleNamespace(n_samples=arr.shape[0], labels=np.zeros(arr.shape[0], dtype=np.int64),
                            sample_ids=list(range(arr.shape[0])))
    monkeypatch.setattr(adopt, "_load_bench", lambda benchmark, s2_only: (bench, benchmark))
    monkeypatch.setattr(adopt, "_subset_bench", lambda b, idx: idx)  # feed indices to the fake encode
    monkeypatch.setattr(adopt.C, "build_model", lambda name, **kw: _FakeTabModel(arr, dim, delta))
    monkeypatch.setattr(adopt.C, "EMBEDDINGS_DIR", tmp_path / "emb")
    monkeypatch.setattr(adopt.C, "checkpoint_sha256", lambda *a, **k: "CKPT")
    monkeypatch.setattr(adopt.C, "dataset_digest", lambda *a, **k: "DSET")


def _tab_cand(legacy: Path):
    return {"benchmark": "cropharvest", "model": "raw", "s2_only": False, "dense": False,
            "legacy": str(legacy), "weights_path": None}


def test_tabular_adopt_report_never_writes(tmp_path, monkeypatch):
    adopt = _load_tool("adopt_embeddings")
    arr = np.arange(24, dtype=np.float32).reshape(8, 3)
    legacy = tmp_path / "legacy" / "baseline.npy"
    legacy.parent.mkdir(parents=True)
    np.save(legacy, arr)
    _setup_tabular(adopt, monkeypatch, tmp_path, arr=arr)
    monkeypatch.setitem(adopt.CONFIG, "mode", "report")

    result = adopt._adopt_tabular(_tab_cand(legacy), {})
    assert result["status"] == "would-adopt"
    assert not (tmp_path / "emb").exists()  # report mode writes nothing


def test_tabular_adopt_publish_success_source_untouched_manifest_last(tmp_path, monkeypatch):
    adopt = _load_tool("adopt_embeddings")
    arr = np.arange(24, dtype=np.float32).reshape(8, 3)
    legacy = tmp_path / "legacy" / "baseline.npy"
    legacy.parent.mkdir(parents=True)
    np.save(legacy, arr)
    legacy_bytes = legacy.read_bytes()
    _setup_tabular(adopt, monkeypatch, tmp_path, arr=arr)
    monkeypatch.setitem(adopt.CONFIG, "mode", "publish")

    art_path = adopt.C.embedding_cache_path("cropharvest", "raw", "baseline")
    orig_write = adopt.C._write_manifest

    def _manifest_last(path, manifest):  # the array must already exist when the manifest is written
        assert art_path.exists(), "manifest written before the array -- not manifest-last"
        return orig_write(path, manifest)

    monkeypatch.setattr(adopt.C, "_write_manifest", _manifest_last)

    result = adopt._adopt_tabular(_tab_cand(legacy), {})
    assert result["status"] == "adopted"
    assert legacy.read_bytes() == legacy_bytes  # source untouched
    man = adopt.C._read_manifest(adopt.C.embedding_manifest_path("cropharvest", "raw", "baseline"))
    assert man["benchmark"] == "cropharvest" and man["checkpoint_sha256"] == "CKPT"
    assert man["dataset_digest"] == "DSET" and man["shape"] == [8, 3]
    assert man["artifact_sha256"] == artifacts.sha256_file(art_path)  # sidecar matches published array
    np.testing.assert_array_equal(np.load(art_path), arr)


def test_tabular_refuses_missing_legacy(tmp_path, monkeypatch):
    adopt = _load_tool("adopt_embeddings")
    arr = np.zeros((4, 2), dtype=np.float32)
    _setup_tabular(adopt, monkeypatch, tmp_path, arr=arr)
    result = adopt._adopt_tabular(_tab_cand(tmp_path / "nope.npy"), {})
    assert result["status"] == "refused" and "missing" in result["reason"]


def test_tabular_refuses_wrong_dtype(tmp_path, monkeypatch):
    adopt = _load_tool("adopt_embeddings")
    arr = np.zeros((4, 2), dtype=np.float64)  # not float32
    legacy = tmp_path / "legacy.npy"
    np.save(legacy, arr)
    _setup_tabular(adopt, monkeypatch, tmp_path, arr=arr.astype(np.float32), model_dim=2)
    # the legacy array on disk is float64; reload path reads its true dtype
    result = adopt._adopt_tabular(_tab_cand(legacy), {})
    assert result["status"] == "refused" and "dtype" in result["reason"]


def test_tabular_refuses_wrong_dimension(tmp_path, monkeypatch):
    adopt = _load_tool("adopt_embeddings")
    arr = np.zeros((4, 3), dtype=np.float32)
    legacy = tmp_path / "legacy.npy"
    np.save(legacy, arr)
    _setup_tabular(adopt, monkeypatch, tmp_path, arr=arr, model_dim=99)  # model expects 99, array is 3
    result = adopt._adopt_tabular(_tab_cand(legacy), {})
    assert result["status"] == "refused" and "feature width" in result["reason"]


def test_tabular_refuses_failed_spotcheck(tmp_path, monkeypatch):
    adopt = _load_tool("adopt_embeddings")
    arr = np.arange(24, dtype=np.float32).reshape(8, 3)
    legacy = tmp_path / "legacy.npy"
    np.save(legacy, arr)
    _setup_tabular(adopt, monkeypatch, tmp_path, arr=arr, delta=1.0)  # re-encode diverges by 1.0
    result = adopt._adopt_tabular(_tab_cand(legacy), {})
    assert result["status"] == "refused" and "spot check" in result["reason"]


def test_tabular_refuses_malformed_array(tmp_path, monkeypatch):
    adopt = _load_tool("adopt_embeddings")
    legacy = tmp_path / "legacy.npy"
    legacy.write_bytes(b"\x93NUMPY not-a-valid-npy")
    _setup_tabular(adopt, monkeypatch, tmp_path, arr=np.zeros((4, 2), dtype=np.float32))
    result = adopt._adopt_tabular(_tab_cand(legacy), {})
    assert result["status"] == "refused" and "unreadable" in result["reason"]


# ============================ dense PASTIS adoption =========================

_TILE_SPECS = (("11_0_0", 1), ("11_0_1", 1), ("22_0_0", 2))


class _FakeTile:
    def __init__(self, features: np.ndarray):
        self.features = features


class _FakeDenseBench:
    def __init__(self, tiles):  # tiles: list of (tile_id, fold, features)
        self._tiles = tiles

    def iter_tiles(self):
        for tile_id, fold, feats in self._tiles:
            yield tile_id, fold, _FakeTile(feats), np.zeros(feats.shape[0], dtype=np.uint8)


class _FakeDenseModel:
    def __init__(self, dim: int, delta: float = 0.0):
        self.embedding_dim, self._delta = dim, delta

    def encode_dense(self, tile):
        return tile.features + self._delta


def _write_tile(root: Path, rel: str, arr: np.ndarray) -> None:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, arr)


def _setup_dense(adopt, monkeypatch, tmp_path, *, dim=4, delta=0.0, drop=None,
                 extra_label=None, extra_feature=None,
                 malformed_feat=None, bad_dtype_feat=None, bad_dim_feat=None):
    """Build a synthetic legacy dense cache. The ``*_feat`` faults target ONE specific feature rel
    (use a NON-first rel to prove exhaustive per-tile validation)."""
    rng = np.random.default_rng(0)
    feat_rels = sorted(f"fold_{fold}/{tid}.npy" for tid, fold in _TILE_SPECS)
    lab_rels = sorted(f"fold_{fold}/{tid}.labels.npy" for tid, fold in _TILE_SPECS)
    legacy = tmp_path / "legacy"
    bench_tiles = []
    for tid, fold in _TILE_SPECS:
        frel, lrel = f"fold_{fold}/{tid}.npy", f"fold_{fold}/{tid}.labels.npy"
        feats = rng.standard_normal((5, dim)).astype(np.float32)
        if frel == malformed_feat:
            (legacy / frel).parent.mkdir(parents=True, exist_ok=True)
            (legacy / frel).write_bytes(b"\x93NUMPY broken")
        elif frel == bad_dtype_feat:
            _write_tile(legacy, frel, feats.astype(np.float64))
        elif frel == bad_dim_feat:
            _write_tile(legacy, frel, rng.standard_normal((5, dim + 3)).astype(np.float32))
        elif frel != drop:
            _write_tile(legacy, frel, feats)
        if lrel != drop:
            _write_tile(legacy, lrel, np.zeros(5, dtype=np.uint8))
        bench_tiles.append((tid, fold, feats))
    if extra_label is not None:
        _write_tile(legacy, extra_label, np.zeros(3, dtype=np.uint8))
    if extra_feature is not None:
        _write_tile(legacy, extra_feature, np.zeros((3, dim), dtype=np.float32))
    bench = _FakeDenseBench(bench_tiles)
    monkeypatch.setattr(adopt, "_load_bench", lambda benchmark, s2_only: (bench, benchmark))
    monkeypatch.setattr(adopt.C, "_dense_expected_rels", lambda b: (feat_rels, lab_rels))
    monkeypatch.setattr(adopt.C, "build_model", lambda name, **kw: _FakeDenseModel(dim, delta))
    monkeypatch.setattr(adopt.C, "EMBEDDINGS_DIR", tmp_path / "emb")
    monkeypatch.setattr(adopt.C, "checkpoint_sha256", lambda *a, **k: "CKPT")
    monkeypatch.setattr(adopt.C, "dataset_digest", lambda *a, **k: "DSET")
    return legacy, feat_rels, lab_rels


def _dense_cand(legacy: Path):
    return {"benchmark": "pastis", "model": "galileo", "s2_only": False, "dense": True,
            "legacy": str(legacy), "weights_path": None}


def test_dense_adopt_report_never_writes(tmp_path, monkeypatch):
    adopt = _load_tool("adopt_embeddings")
    legacy, _, _ = _setup_dense(adopt, monkeypatch, tmp_path)
    monkeypatch.setitem(adopt.CONFIG, "mode", "report")
    result = adopt._adopt_dense(_dense_cand(legacy), {})
    assert result["status"] == "would-adopt"
    assert not (tmp_path / "emb").exists()


def test_dense_adopt_publish_success_source_untouched_manifest_last(tmp_path, monkeypatch):
    adopt = _load_tool("adopt_embeddings")
    legacy, feat_rels, lab_rels = _setup_dense(adopt, monkeypatch, tmp_path)
    monkeypatch.setitem(adopt.CONFIG, "mode", "publish")

    before = {rel: (legacy / rel).read_bytes() for rel in feat_rels + lab_rels}
    root = adopt.C.dense_embedding_cache_dir("pastis", "galileo", "baseline")
    orig_write = adopt.C._write_manifest

    def _manifest_last(path, manifest):
        for rel in feat_rels + lab_rels:  # every expected tile present before the manifest lands
            assert (root / rel).exists(), f"manifest written before tile {rel}"
        return orig_write(path, manifest)

    monkeypatch.setattr(adopt.C, "_write_manifest", _manifest_last)

    result = adopt._adopt_dense(_dense_cand(legacy), {})
    assert result["status"] == "adopted"
    # source untouched
    for rel, data in before.items():
        assert (legacy / rel).read_bytes() == data
    # canonical tiles + slim manifest
    for rel in feat_rels + lab_rels:
        assert (root / rel).exists()
    man = adopt.C._read_manifest(adopt.C.dense_manifest_path("pastis", "galileo", "baseline"))
    assert man["feature_tile_count"] == 3 and man["label_tile_count"] == 3
    assert man["feature_dim"] == 4 and man["dtype"] == "float32"
    assert man["checkpoint_sha256"] == "CKPT" and man["dataset_digest"] == "DSET"
    assert man["tile_set_digest"] == adopt.C.tile_set_digest(feat_rels, lab_rels)


def test_dense_refuses_missing_tile(tmp_path, monkeypatch):
    adopt = _load_tool("adopt_embeddings")
    legacy, _, _ = _setup_dense(adopt, monkeypatch, tmp_path, drop="fold_1/11_0_1.npy")
    result = adopt._adopt_dense(_dense_cand(legacy), {})
    assert result["status"] == "refused" and "missing" in result["reason"]


def test_dense_refuses_extra_label_tile(tmp_path, monkeypatch):
    adopt = _load_tool("adopt_embeddings")
    legacy, _, _ = _setup_dense(adopt, monkeypatch, tmp_path, extra_label="fold_2/999_0_0.labels.npy")
    result = adopt._adopt_dense(_dense_cand(legacy), {})
    assert result["status"] == "refused" and "UNEXPECTED label" in result["reason"]


def test_dense_refuses_extra_feature_only_tile(tmp_path, monkeypatch):
    # A stray feature file with no matching label -- the label-only glob used to miss this entirely.
    adopt = _load_tool("adopt_embeddings")
    legacy, _, _ = _setup_dense(adopt, monkeypatch, tmp_path, extra_feature="fold_2/999_0_0.npy")
    result = adopt._adopt_dense(_dense_cand(legacy), {})
    assert result["status"] == "refused" and "UNEXPECTED feature" in result["reason"]


def test_dense_refuses_wrong_dtype_nonfirst_tile(tmp_path, monkeypatch):
    adopt = _load_tool("adopt_embeddings")
    legacy, _, _ = _setup_dense(adopt, monkeypatch, tmp_path, bad_dtype_feat="fold_1/11_0_1.npy")
    result = adopt._adopt_dense(_dense_cand(legacy), {})
    assert result["status"] == "refused" and "11_0_1.npy: dtype" in result["reason"]


def test_dense_refuses_wrong_dimension_nonfirst_tile(tmp_path, monkeypatch):
    adopt = _load_tool("adopt_embeddings")
    legacy, _, _ = _setup_dense(adopt, monkeypatch, tmp_path, bad_dim_feat="fold_1/11_0_1.npy")
    result = adopt._adopt_dense(_dense_cand(legacy), {})
    assert result["status"] == "refused" and "11_0_1.npy: feature width" in result["reason"]


def test_dense_refuses_malformed_nonfirst_tile(tmp_path, monkeypatch):
    adopt = _load_tool("adopt_embeddings")
    legacy, _, _ = _setup_dense(adopt, monkeypatch, tmp_path, malformed_feat="fold_1/11_0_1.npy")
    result = adopt._adopt_dense(_dense_cand(legacy), {})
    assert result["status"] == "refused" and "unreadable feature tile fold_1/11_0_1.npy" in result["reason"]


def test_dense_refuses_failed_spotcheck(tmp_path, monkeypatch):
    adopt = _load_tool("adopt_embeddings")
    legacy, _, _ = _setup_dense(adopt, monkeypatch, tmp_path, delta=1.0)
    result = adopt._adopt_dense(_dense_cand(legacy), {})
    assert result["status"] == "refused" and "spot check" in result["reason"]


def test_dense_publish_replaces_foreign_partial_tile(tmp_path, monkeypatch):
    # A partial destination left by ANOTHER candidate: a valid-looking .npy at an expected rel but
    # with foreign bytes. Publication must not trust it -- it is replaced from the selected source,
    # so the foreign content can never be certified.
    adopt = _load_tool("adopt_embeddings")
    legacy, feat_rels, _ = _setup_dense(adopt, monkeypatch, tmp_path)
    monkeypatch.setitem(adopt.CONFIG, "mode", "publish")
    root = adopt.C.dense_embedding_cache_dir("pastis", "galileo", "baseline")
    foreign_rel = feat_rels[0]
    _write_tile(root, foreign_rel, np.full((5, 4), 7.0, dtype=np.float32))  # different from legacy
    foreign_bytes = (root / foreign_rel).read_bytes()
    legacy_bytes = (legacy / foreign_rel).read_bytes()
    assert foreign_bytes != legacy_bytes

    result = adopt._adopt_dense(_dense_cand(legacy), {})
    assert result["status"] == "adopted"
    published = (root / foreign_rel).read_bytes()
    assert published == legacy_bytes and published != foreign_bytes  # foreign tile replaced


def test_dense_publish_keeps_byte_identical_tile(tmp_path, monkeypatch):
    # The safe-resume happy path: a destination tile that already matches the source is a no-op.
    adopt = _load_tool("adopt_embeddings")
    legacy, feat_rels, _ = _setup_dense(adopt, monkeypatch, tmp_path)
    monkeypatch.setitem(adopt.CONFIG, "mode", "publish")
    root = adopt.C.dense_embedding_cache_dir("pastis", "galileo", "baseline")
    rel = feat_rels[0]
    _write_tile(root, rel, np.load(legacy / rel))  # identical to source
    result = adopt._adopt_dense(_dense_cand(legacy), {})
    assert result["status"] == "adopted"
    assert (root / rel).read_bytes() == (legacy / rel).read_bytes()


# ============================ dataset preflight =============================


def _mk_files(root: Path, mapping: dict[str, bytes]) -> None:
    for rel, data in mapping.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)


def test_preflight_digest_is_traversal_order_independent(tmp_path):
    pre = _load_tool("preflight_dataset_digests")
    root = tmp_path / "bench"
    _mk_files(root, {"a.npy": b"AAA", "sub/b.npy": b"BBB", "c.txt": b"CCC"})
    rels = ["a.npy", "sub/b.npy", "c.txt"]
    d1 = pre._aggregate_sha256(root, rels)
    d2 = pre._aggregate_sha256(root, list(reversed(rels)))  # different input order
    assert d1 == d2 and len(d1) == 64
    (root / "a.npy").write_bytes(b"ZZZ")  # content actually matters
    assert pre._aggregate_sha256(root, rels) != d1


def test_preflight_refuses_missing_input(tmp_path):
    pre = _load_tool("preflight_dataset_digests")
    root = tmp_path / "bench"
    _mk_files(root, {"a.npy": b"AAA"})
    with pytest.raises(FileNotFoundError):
        pre._aggregate_sha256(root, ["a.npy", "missing.npy"])


def _prep_preflight(pre, monkeypatch, tmp_path, benches):
    """benches: {benchmark: {relpath: bytes}}. Returns (digest_dir, {benchmark: good_digest})."""
    input_root, digest_dir = tmp_path / "benchmarks", tmp_path / "digests"
    consumed = {b: sorted(files) for b, files in benches.items()}
    for bench, files in benches.items():
        _mk_files(input_root / bench, files)
    monkeypatch.setattr(pre.C, "INPUT_ROOT", input_root)
    monkeypatch.setattr(pre.C, "DATASET_DIGEST_DIR", digest_dir)
    monkeypatch.setattr(pre, "_consumed_files", lambda benchmark, r: consumed[benchmark])
    monkeypatch.setitem(pre.CONFIG, "benchmarks", list(benches))
    good = {b: pre._aggregate_sha256(input_root / b, consumed[b]) for b in benches}
    return digest_dir, good


def test_preflight_reference_match_writes_digest(tmp_path, monkeypatch):
    pre = _load_tool("preflight_dataset_digests")
    digest_dir, good = _prep_preflight(pre, monkeypatch, tmp_path, {"fake": {"a.npy": b"AAA", "b.npy": b"BBB"}})
    monkeypatch.setitem(pre.CONFIG, "reference", {"fake": good["fake"]})
    monkeypatch.setitem(pre.CONFIG, "write", True)
    assert pre.main() == 0
    assert (digest_dir / "fake.txt").read_text().strip() == good["fake"]


def test_preflight_write_enabled_mismatch_leaves_files_unchanged(tmp_path, monkeypatch):
    # write=True, but one benchmark's reference mismatches -> transactional abort: the matching
    # benchmark's file is NOT written and a pre-existing digest file is left untouched.
    pre = _load_tool("preflight_dataset_digests")
    digest_dir, good = _prep_preflight(
        pre, monkeypatch, tmp_path, {"good": {"a.npy": b"AAA"}, "bad": {"b.npy": b"BBB"}})
    digest_dir.mkdir()
    (digest_dir / "good.txt").write_text("PRIOR\n")  # a prior good run's file
    monkeypatch.setitem(pre.CONFIG, "reference", {"good": good["good"], "bad": "0" * 64})
    monkeypatch.setitem(pre.CONFIG, "write", True)
    assert pre.main() == 1
    assert (digest_dir / "good.txt").read_text() == "PRIOR\n"  # untouched despite matching + write=True
    assert not (digest_dir / "bad.txt").exists()               # mismatch never written


def test_preflight_dry_run_is_write_free(tmp_path, monkeypatch):
    pre = _load_tool("preflight_dataset_digests")
    digest_dir, _good = _prep_preflight(pre, monkeypatch, tmp_path, {"fake": {"a.npy": b"AAA"}})
    monkeypatch.setitem(pre.CONFIG, "reference", {})  # no reference -> no failure
    monkeypatch.setitem(pre.CONFIG, "write", False)
    assert pre.main() == 0
    assert not digest_dir.exists()  # zero filesystem writes, including no directory creation


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
