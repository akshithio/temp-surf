"""Focused tests for the two TEMPORARY freeze-and-run migration tools.

``tools/adopt_embeddings.py`` (adopt a legacy embedding cache into the canonical layout WITHOUT
regenerating it) and ``tools/preflight_dataset_digests.py`` (portable dataset content digest). These
tools live outside ``src`` and edit a CONFIG block instead of taking CLI args, so the tests import
each module by path and drive its internal functions directly. Everything is synthetic -- no real
benchmark data, no real model, no real checkpoints.
"""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass
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

    def __init__(self, arr: np.ndarray, dim: int, delta: float = 0.0, enc_width=None):
        self._arr, self.embedding_dim, self._delta, self._enc_width = arr, dim, delta, enc_width

    def encode(self, idx):  # `_subset_bench` is patched to pass the index list straight through
        idx = np.asarray(idx)
        if self._enc_width is not None:  # encoder outputs a different width than the legacy array
            return np.zeros((len(idx), self._enc_width), dtype=np.float32)
        return self._arr[idx] + self._delta


def _setup_tabular(adopt, monkeypatch, tmp_path, *, arr, model_dim=None, delta=0.0, enc_width=None):
    dim = arr.shape[1] if model_dim is None else model_dim
    bench = SimpleNamespace(n_samples=arr.shape[0], labels=np.zeros(arr.shape[0], dtype=np.int64),
                            sample_ids=list(range(arr.shape[0])))
    monkeypatch.setattr(adopt, "_load_bench", lambda benchmark, s2_only: (bench, benchmark))
    monkeypatch.setattr(adopt, "_subset_bench", lambda b, idx: idx)  # feed indices to the fake encode
    monkeypatch.setattr(adopt.C, "build_model", lambda name, **kw: _FakeTabModel(arr, dim, delta, enc_width))
    monkeypatch.setattr(adopt.C, "EMBEDDINGS_DIR", tmp_path / "emb")
    monkeypatch.setattr(adopt.C, "CACHE_JSON_PATH", tmp_path / "logs" / "cache.json")
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
    assert result["status"] == "ready"
    assert not (tmp_path / "emb").exists()  # report mode writes nothing


def test_tabular_apply_renames_source_and_writes_record_last(tmp_path, monkeypatch):
    adopt = _load_tool("adopt_embeddings")
    arr = np.arange(24, dtype=np.float32).reshape(8, 3)
    legacy = tmp_path / "legacy" / "baseline.npy"
    legacy.parent.mkdir(parents=True)
    np.save(legacy, arr)
    legacy_inode = legacy.stat().st_ino
    _setup_tabular(adopt, monkeypatch, tmp_path, arr=arr)
    monkeypatch.setitem(adopt.CONFIG, "mode", "apply")

    art_path = adopt.C.embedding_cache_path("cropharvest", "raw", "baseline")
    orig_update = adopt.C.update_cache

    def _record_last(**kwargs):  # the array must already exist when the record is written
        assert art_path.exists(), "record written before the array -- not record-last"
        return orig_update(**kwargs)

    monkeypatch.setattr(adopt.C, "update_cache", _record_last)

    result = adopt._adopt_tabular(_tab_cand(legacy), {})
    assert result["status"] == "adopted"
    assert not legacy.exists()
    assert not legacy.parent.exists()
    assert art_path.stat().st_ino == legacy_inode  # same file, renamed rather than copied
    record = adopt.C._cache_record("cropharvest", "raw", "baseline")
    assert record["checkpoint_sha256"] == "CKPT" and record["dataset_digest"] == "DSET"
    assert record["shape"] == [8, 3]
    assert record["artifact_sha256"] == artifacts.sha256_file(art_path)  # record matches published array
    np.testing.assert_array_equal(np.load(art_path), arr)


def test_tabular_apply_does_not_serialize_embedding_data(tmp_path, monkeypatch):
    adopt = _load_tool("adopt_embeddings")
    arr = np.arange(24, dtype=np.float32).reshape(8, 3)
    legacy = tmp_path / "legacy" / "baseline.npy"
    legacy.parent.mkdir(parents=True)
    np.save(legacy, arr)
    _setup_tabular(adopt, monkeypatch, tmp_path, arr=arr)
    monkeypatch.setitem(adopt.CONFIG, "mode", "apply")
    monkeypatch.setattr(
        adopt.np,
        "save",
        lambda *_args, **_kwargs: pytest.fail("embedding data must be renamed, not serialized"),
    )

    assert adopt._adopt_tabular(_tab_cand(legacy), {})["status"] == "adopted"


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
    # The encoder OUTPUT width (5) differs from the legacy array width (3) -> refused.
    adopt = _load_tool("adopt_embeddings")
    arr = np.zeros((8, 3), dtype=np.float32)
    legacy = tmp_path / "legacy.npy"
    np.save(legacy, arr)
    _setup_tabular(adopt, monkeypatch, tmp_path, arr=arr, enc_width=5)
    result = adopt._adopt_tabular(_tab_cand(legacy), {})
    assert result["status"] == "refused" and "feature width" in result["reason"]


def test_tabular_adopts_zero_embedding_dim_model(tmp_path, monkeypatch):
    # RawModel reports embedding_dim=0 until its first encode; adoption must take the width from the
    # encoder OUTPUT, not the attribute -- otherwise every RawModel cell is wrongly refused "width != 0".
    adopt = _load_tool("adopt_embeddings")
    arr = np.arange(24, dtype=np.float32).reshape(8, 3)
    legacy = tmp_path / "legacy.npy"
    np.save(legacy, arr)
    _setup_tabular(adopt, monkeypatch, tmp_path, arr=arr, model_dim=0)  # embedding_dim=0 like RawModel
    monkeypatch.setitem(adopt.CONFIG, "mode", "apply")
    result = adopt._adopt_tabular(_tab_cand(legacy), {})
    assert result["status"] == "adopted"


def test_tabular_refuses_cross_filesystem_move(tmp_path, monkeypatch):
    adopt = _load_tool("adopt_embeddings")
    arr = np.arange(24, dtype=np.float32).reshape(8, 3)
    legacy = tmp_path / "legacy" / "baseline.npy"
    legacy.parent.mkdir(parents=True)
    np.save(legacy, arr)
    _setup_tabular(adopt, monkeypatch, tmp_path, arr=arr)
    monkeypatch.setattr(adopt, "_device_id", lambda path: 1 if path == legacy else 2)

    result = adopt._adopt_tabular(_tab_cand(legacy), {})
    assert result["status"] == "refused" and "cross-filesystem" in result["reason"]
    assert legacy.exists()


def test_tabular_refuses_unexpected_destination(tmp_path, monkeypatch):
    adopt = _load_tool("adopt_embeddings")
    arr = np.arange(24, dtype=np.float32).reshape(8, 3)
    legacy = tmp_path / "legacy" / "baseline.npy"
    legacy.parent.mkdir(parents=True)
    np.save(legacy, arr)
    _setup_tabular(adopt, monkeypatch, tmp_path, arr=arr)
    destination = adopt.C.embedding_cache_path("cropharvest", "raw", "baseline")
    destination.parent.mkdir(parents=True)
    np.save(destination, arr)

    result = adopt._adopt_tabular(_tab_cand(legacy), {})
    assert result["status"] == "refused" and "already exists" in result["reason"]
    assert legacy.exists()


def test_tabular_valid_canonical_is_checked_before_noop(tmp_path, monkeypatch):
    adopt = _load_tool("adopt_embeddings")
    arr = np.arange(24, dtype=np.float32).reshape(8, 3)
    legacy = tmp_path / "legacy" / "baseline.npy"
    legacy.parent.mkdir(parents=True)
    np.save(legacy, arr)
    _setup_tabular(adopt, monkeypatch, tmp_path, arr=arr)
    monkeypatch.setitem(adopt.CONFIG, "mode", "apply")
    assert adopt._adopt_tabular(_tab_cand(legacy), {})["status"] == "adopted"

    result = adopt._adopt_tabular(_tab_cand(legacy), {})
    assert result["status"] == "already-adopted"


def test_tabular_invalid_canonical_is_not_accepted_as_noop(tmp_path, monkeypatch):
    adopt = _load_tool("adopt_embeddings")
    arr = np.arange(24, dtype=np.float32).reshape(8, 3)
    legacy = tmp_path / "legacy" / "baseline.npy"
    legacy.parent.mkdir(parents=True)
    np.save(legacy, arr)
    _setup_tabular(adopt, monkeypatch, tmp_path, arr=arr)
    monkeypatch.setitem(adopt.CONFIG, "mode", "apply")
    assert adopt._adopt_tabular(_tab_cand(legacy), {})["status"] == "adopted"
    key = adopt.C._embedding_key("cropharvest", "raw", "baseline")
    record = adopt.C._cache_record("cropharvest", "raw", "baseline")
    record["checkpoint_sha256"] = "wrong"  # a validated identity field now disagrees
    adopt.C.update_cache(embeddings={key: record})

    result = adopt._adopt_tabular(_tab_cand(legacy), {})
    assert result["status"] == "refused" and "invalid canonical" in result["reason"]


def test_tabular_changed_canonical_array_is_not_accepted_as_noop(tmp_path, monkeypatch):
    adopt = _load_tool("adopt_embeddings")
    arr = np.arange(24, dtype=np.float32).reshape(8, 3)
    legacy = tmp_path / "legacy" / "baseline.npy"
    legacy.parent.mkdir(parents=True)
    np.save(legacy, arr)
    _setup_tabular(adopt, monkeypatch, tmp_path, arr=arr)
    monkeypatch.setitem(adopt.CONFIG, "mode", "apply")
    assert adopt._adopt_tabular(_tab_cand(legacy), {})["status"] == "adopted"
    destination = adopt.C.embedding_cache_path("cropharvest", "raw", "baseline")
    changed = arr.copy()
    changed[0, 0] += 1
    np.save(destination, changed)

    result = adopt._adopt_tabular(_tab_cand(legacy), {})
    assert result["status"] == "refused"


def test_tabular_rerun_finishes_record_after_rename_interruption(tmp_path, monkeypatch):
    adopt = _load_tool("adopt_embeddings")
    arr = np.arange(24, dtype=np.float32).reshape(8, 3)
    legacy = tmp_path / "legacy" / "baseline.npy"
    legacy.parent.mkdir(parents=True)
    np.save(legacy, arr)
    _setup_tabular(adopt, monkeypatch, tmp_path, arr=arr)
    monkeypatch.setitem(adopt.CONFIG, "mode", "apply")
    original_update = adopt.C.update_cache
    monkeypatch.setattr(
        adopt.C,
        "update_cache",
        lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("interrupted")),
    )
    with pytest.raises(RuntimeError, match="interrupted"):
        adopt._adopt_tabular(_tab_cand(legacy), {})
    destination = adopt.C.embedding_cache_path("cropharvest", "raw", "baseline")
    assert destination.exists() and not legacy.exists()

    monkeypatch.setattr(adopt.C, "update_cache", original_update)
    assert adopt._adopt_tabular(_tab_cand(legacy), {})["status"] == "adopted"
    assert not legacy.parent.exists()


def test_real_rawmodel_reports_zero_embedding_dim_before_encode():
    # The exact property that broke adoption: RawModel.embedding_dim is 0 until the first encode.
    from models.raw import RawModel

    assert RawModel().embedding_dim == 0


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


@dataclass(frozen=True)
class _TrackedPatch:
    patch_id: int
    fold: int


@dataclass(frozen=True)
class _TrackedDenseBench:
    patches: tuple[_TrackedPatch, ...]
    visited: list[int]

    def iter_tiles(self):
        for patch in self.patches:
            self.visited.append(patch.patch_id)
            for col in (0, 1):
                tile_id = f"{patch.patch_id}_0_{col}"
                yield tile_id, patch.fold, tile_id, np.zeros(1, dtype=np.uint8)


def test_dense_spotcheck_loads_only_selected_patches():
    adopt = _load_tool("adopt_embeddings")
    visited = []
    bench = _TrackedDenseBench(
        patches=tuple(_TrackedPatch(patch_id, 1) for patch_id in range(100, 110)),
        visited=visited,
    )
    picks = {"fold_1/102_0_1.npy", "fold_1/108_0_0.npy"}

    selected = list(adopt._iter_selected_dense_tiles(bench, picks))

    assert [tile_id for tile_id, _fold, _tile, _labels in selected] == ["102_0_1", "108_0_0"]
    assert visited == [102, 108]


def _write_tile(root: Path, rel: str, arr: np.ndarray) -> None:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, arr)


def _setup_dense(adopt, monkeypatch, tmp_path, *, dim=4, model_dim=None, delta=0.0, drop=None,
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
    monkeypatch.setattr(adopt.C, "build_model",
                        lambda name, **kw: _FakeDenseModel(dim if model_dim is None else model_dim, delta))
    monkeypatch.setattr(adopt.C, "EMBEDDINGS_DIR", tmp_path / "emb")
    monkeypatch.setattr(adopt.C, "CACHE_JSON_PATH", tmp_path / "logs" / "cache.json")
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
    assert result["status"] == "ready"
    assert not (tmp_path / "emb").exists()


def test_dense_adopts_zero_embedding_dim_model(tmp_path, monkeypatch):
    # Dense RawModel also reports embedding_dim=0 until its first encode; dense adoption must derive
    # the reference width from the legacy tiles, not the attribute, and let the spot-check confirm it.
    adopt = _load_tool("adopt_embeddings")
    legacy, feat_rels, lab_rels = _setup_dense(adopt, monkeypatch, tmp_path, model_dim=0)
    monkeypatch.setitem(adopt.CONFIG, "mode", "apply")
    result = adopt._adopt_dense(_dense_cand(legacy), {})
    assert result["status"] == "adopted"
    root = adopt.C.dense_embedding_cache_dir("pastis", "galileo", "baseline")
    for rel in feat_rels + lab_rels:
        assert (root / rel).exists()


def test_dense_apply_renames_source_and_writes_record_last(tmp_path, monkeypatch):
    adopt = _load_tool("adopt_embeddings")
    legacy, feat_rels, lab_rels = _setup_dense(adopt, monkeypatch, tmp_path)
    monkeypatch.setitem(adopt.CONFIG, "mode", "apply")

    legacy_inode = legacy.stat().st_ino
    root = adopt.C.dense_embedding_cache_dir("pastis", "galileo", "baseline")
    orig_update = adopt.C.update_cache

    def _record_last(**kwargs):
        for rel in feat_rels + lab_rels:  # every expected tile present before the record lands
            assert (root / rel).exists(), f"record written before tile {rel}"
        return orig_update(**kwargs)

    monkeypatch.setattr(adopt.C, "update_cache", _record_last)

    result = adopt._adopt_dense(_dense_cand(legacy), {})
    assert result["status"] == "adopted"
    assert not legacy.exists()
    assert root.stat().st_ino == legacy_inode  # whole tree renamed without per-tile copies
    for rel in feat_rels + lab_rels:
        assert (root / rel).exists()
    record = adopt.C._cache_record("pastis", "galileo", "baseline")
    assert record["feature_tile_count"] == 3 and record["label_tile_count"] == 3
    assert record["feature_dim"] == 4 and record["dtype"] == "float32"
    assert record["checkpoint_sha256"] == "CKPT" and record["dataset_digest"] == "DSET"
    assert record["tile_set_digest"] == adopt.C.tile_set_digest(feat_rels, lab_rels)


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


def test_dense_refuses_unexpected_destination(tmp_path, monkeypatch):
    adopt = _load_tool("adopt_embeddings")
    legacy, feat_rels, _ = _setup_dense(adopt, monkeypatch, tmp_path)
    root = adopt.C.dense_embedding_cache_dir("pastis", "galileo", "baseline")
    foreign_rel = feat_rels[0]
    _write_tile(root, foreign_rel, np.full((5, 4), 7.0, dtype=np.float32))

    result = adopt._adopt_dense(_dense_cand(legacy), {})
    assert result["status"] == "refused" and "already exists" in result["reason"]
    assert legacy.exists()


def test_dense_refuses_cross_filesystem_move(tmp_path, monkeypatch):
    adopt = _load_tool("adopt_embeddings")
    legacy, _, _ = _setup_dense(adopt, monkeypatch, tmp_path)
    monkeypatch.setattr(adopt, "_device_id", lambda path: 1 if path == legacy else 2)

    result = adopt._adopt_dense(_dense_cand(legacy), {})
    assert result["status"] == "refused" and "cross-filesystem" in result["reason"]
    assert legacy.exists()


def test_dense_canonical_noop_runs_full_validation(tmp_path, monkeypatch):
    adopt = _load_tool("adopt_embeddings")
    legacy, _, label_rels = _setup_dense(adopt, monkeypatch, tmp_path)
    monkeypatch.setitem(adopt.CONFIG, "mode", "apply")
    assert adopt._adopt_dense(_dense_cand(legacy), {})["status"] == "adopted"
    assert adopt._adopt_dense(_dense_cand(legacy), {})["status"] == "already-adopted"
    root = adopt.C.dense_embedding_cache_dir("pastis", "galileo", "baseline")
    np.save(root / label_rels[-1], np.zeros(5, dtype=np.float32))

    result = adopt._adopt_dense(_dense_cand(legacy), {})
    assert result["status"] == "refused" and "dtype" in result["reason"]


def test_main_applies_valid_candidate_before_later_refusal(monkeypatch):
    adopt = _load_tool("adopt_embeddings")
    applied = []
    candidates = [{"benchmark": "a", "model": "m"}, {"benchmark": "b", "model": "m"}]
    monkeypatch.setitem(adopt.CONFIG, "mode", "apply")
    monkeypatch.setitem(adopt.CONFIG, "candidates", candidates)

    def prepare(candidate, _enc_kwargs):
        if candidate["benchmark"] == "b":
            return adopt._Plan({"candidate": "b/m", "status": "refused", "reason": "bad"})
        return adopt._Plan(
            {"candidate": "a/m", "status": "ready"},
            lambda: applied.append("a") or {"candidate": "a/m", "status": "adopted"},
        )

    monkeypatch.setattr(adopt, "_prepare_candidate", prepare)
    assert adopt.main() == 1
    assert applied == ["a"]


def test_main_validates_each_candidate_once(monkeypatch):
    adopt = _load_tool("adopt_embeddings")
    prepared, applied = [], []
    candidates = [{"benchmark": "a", "model": "m"}, {"benchmark": "b", "model": "m"}]
    monkeypatch.setitem(adopt.CONFIG, "mode", "apply")
    monkeypatch.setitem(adopt.CONFIG, "candidates", candidates)

    def prepare(candidate, _enc_kwargs):
        name = candidate["benchmark"]
        prepared.append(name)
        return adopt._Plan(
            {"candidate": f"{name}/m", "status": "ready"},
            lambda: applied.append(name) or {"candidate": f"{name}/m", "status": "adopted"},
        )

    monkeypatch.setattr(adopt, "_prepare_candidate", prepare)
    assert adopt.main() == 0
    assert prepared == ["a", "b"] and applied == ["a", "b"]


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
    """benches: {benchmark: {relpath: bytes}}. Returns (cache_json_path, {benchmark: good_digest})."""
    input_root = tmp_path / "benchmarks"
    cache_json = tmp_path / "logs" / "cache.json"
    consumed = {b: sorted(files) for b, files in benches.items()}
    for bench, files in benches.items():
        _mk_files(input_root / bench, files)
    monkeypatch.setattr(pre.C, "INPUT_ROOT", input_root)
    monkeypatch.setattr(pre.C, "CACHE_JSON_PATH", cache_json)
    monkeypatch.setattr(pre, "_consumed_files", lambda benchmark, r: consumed[benchmark])
    monkeypatch.setitem(pre.CONFIG, "benchmarks", list(benches))
    good = {b: pre._aggregate_sha256(input_root / b, consumed[b]) for b in benches}
    return cache_json, good


def test_preflight_reference_match_writes_digest(tmp_path, monkeypatch):
    pre = _load_tool("preflight_dataset_digests")
    _cache_json, good = _prep_preflight(pre, monkeypatch, tmp_path, {"fake": {"a.npy": b"AAA", "b.npy": b"BBB"}})
    monkeypatch.setitem(pre.CONFIG, "reference", {"fake": good["fake"]})
    monkeypatch.setitem(pre.CONFIG, "write", True)
    assert pre.main() == 0
    assert pre.C._read_cache_doc()["datasets"]["fake"] == good["fake"]


def test_preflight_write_enabled_mismatch_leaves_records_unchanged(tmp_path, monkeypatch):
    # write=True, but one benchmark's reference mismatches -> transactional abort: nothing is merged
    # into cache.json and a pre-existing dataset record is left untouched.
    pre = _load_tool("preflight_dataset_digests")
    _cache_json, good = _prep_preflight(
        pre, monkeypatch, tmp_path, {"good": {"a.npy": b"AAA"}, "bad": {"b.npy": b"BBB"}})
    pre.C.update_cache(datasets={"good": "PRIOR"})  # a prior good run's record
    monkeypatch.setitem(pre.CONFIG, "reference", {"good": good["good"], "bad": "0" * 64})
    monkeypatch.setitem(pre.CONFIG, "write", True)
    assert pre.main() == 1
    datasets = pre.C._read_cache_doc()["datasets"]
    assert datasets["good"] == "PRIOR"  # untouched despite matching + write=True
    assert "bad" not in datasets        # mismatch never written


def test_preflight_dry_run_is_write_free(tmp_path, monkeypatch):
    pre = _load_tool("preflight_dataset_digests")
    cache_json, _good = _prep_preflight(pre, monkeypatch, tmp_path, {"fake": {"a.npy": b"AAA"}})
    monkeypatch.setitem(pre.CONFIG, "reference", {})  # no reference -> no failure
    monkeypatch.setitem(pre.CONFIG, "write", False)
    assert pre.main() == 0
    assert not cache_json.exists()  # zero filesystem writes


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
