from __future__ import annotations

import contextlib
import fcntl
import functools
import hashlib
import importlib
import inspect
import json
import os
import pickle
import re
import uuid
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from dataio import get_input as GI
from evals.benchmarks.pastis import PastisBenchmark
from utils import perfutils as perf

REPO = Path(__file__).resolve().parents[2]
SCRATCH = REPO / "data"
INPUT_ROOT = Path(os.environ.get("ROBUSTNESS_INPUT", REPO / "data" / "input")) / "benchmarks"
CACHE_DIR = SCRATCH / "cache"
OUTPUT_DIR = SCRATCH / "output"
EMBEDDINGS_DIR = CACHE_DIR / "embeddings"
BENCH_SRC = REPO / "src" / "dataio" / "get_input.py"

BENCHMARK_MODULES: dict[str, str] = {
    "cropharvest": "evals/benchmarks/cropharvest",
    "eurocropsml": "evals/benchmarks/eurocropsml",
    "breizhcrops": "evals/benchmarks/breizhcrops",
    "pastis": "evals/benchmarks/pastis",
}

MODELS: dict[str, tuple[str, str]] = {
    "raw": ("models.raw", "RawModel"),  # reality-check control (not a foundation model)
    "presto": ("models.presto", "PrestoModel"),
    "olmoearth": ("models.olmoearth", "OlmoEarthModel"),
    "galileo": ("models.galileo", "GalileoModel"),
    "agrifm": ("models.agrifm", "AgriFMModel"),
    "tessera": ("models.tessera", "TesseraModel"),
}

EMB_DTYPE = "float32"


class MissingEmbeddingCache(FileNotFoundError):
    pass


class ModelWrapper(Protocol):
    embedding_dim: int

    def encode(self, bench: Any) -> np.ndarray: ...


def _hash_files(*paths: str | Path) -> str:
    h = hashlib.sha256()
    for p in sorted(map(str, paths)):
        try:
            h.update(Path(p).read_bytes())
        except OSError:
            h.update(b"<missing>")
    return h.hexdigest()[:10]


def _hash_str(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()[:10]


def _atomic_tmp(path: Path) -> Path:
    return path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")


@contextlib.contextmanager
def _cache_lock(path: Path):
    lock_path = path.with_name(f".{path.name}.lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with open(lock_path, "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def _update_file_content_hash(path: str | Path, h: Any) -> None:
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)


def _input_fingerprint(bench_dir: Path, mode: str | None = None) -> str:
    h = hashlib.sha256()
    mode = (mode or os.environ.get("DATA_FINGERPRINT", "deep")).strip().lower()
    try:
        if mode == "top":
            for e in sorted(bench_dir.iterdir(), key=lambda x: x.name):
                st = e.stat()
                h.update(f"{e.name}:{st.st_size}:{int(st.st_mtime)}".encode())
        else:
            for root, dirs, files in os.walk(bench_dir):
                dirs.sort()
                rel = os.path.relpath(root, bench_dir)
                for name in sorted(files):
                    fp = os.path.join(root, name)
                    try:
                        h.update(f"{rel}/{name}\0".encode())
                        _update_file_content_hash(fp, h)
                    except OSError:
                        h.update(f"{rel}/{name}:<missing>".encode())
    except OSError:
        h.update(b"<missing>")
    return h.hexdigest()[:10]


def _pastis_input_fingerprint(bench_dir: Path, mode: str | None = None) -> str:
    """Fingerprint the PASTIS inputs actually consumed by the loader.

    The release directory also contains large unused products such as descending-orbit
    S1D and instance annotations. Walking and hashing every byte forces a tens-of-GB
    reread before every resumed run, even when the dense tile cache is already complete.
    PASTIS release arrays are immutable staged inputs here, so the manifest uses
    metadata content plus array size/mtime identity for the S2, S1A, and TARGET files.
    """
    mode = (mode or os.environ.get("DATA_FINGERPRINT", "deep")).strip().lower()
    if mode == "top":
        return _input_fingerprint(bench_dir, mode)

    h = hashlib.sha256()
    metadata = bench_dir / "metadata.geojson"
    try:
        h.update(b"metadata.geojson\0")
        _update_file_content_hash(metadata, h)
        geo = json.loads(metadata.read_text())
        patch_ids = sorted({int(feature["properties"]["ID_PATCH"]) for feature in geo["features"]})
    except Exception:
        h.update(b"metadata.geojson:<missing-or-invalid>")
        patch_ids = []

    required = (
        ("DATA_S2", "S2_{}.npy"),
        ("DATA_S1A", "S1A_{}.npy"),
        ("ANNOTATIONS", "TARGET_{}.npy"),
    )
    for patch_id in patch_ids:
        for subdir, template in required:
            rel = Path(subdir) / template.format(patch_id)
            path = bench_dir / rel
            try:
                st = path.stat()
                h.update(f"{rel.as_posix()}:{st.st_size}:{st.st_mtime_ns}".encode())
            except OSError:
                h.update(f"{rel.as_posix()}:<missing>".encode())
    return h.hexdigest()[:10]


@functools.lru_cache(maxsize=32)
def _benchmark_input_fingerprint(benchmark: str, mode: str, _data_version: str) -> str:
    if benchmark == "pastis":
        return _pastis_input_fingerprint(INPUT_ROOT / benchmark, mode)
    return _input_fingerprint(INPUT_ROOT / benchmark, mode)


def bench_tag(benchmark: str, kwargs: dict) -> str:
    params = "_".join(f"{k}-{kwargs[k]}" for k in sorted(kwargs)) or "default"
    spec_path = REPO / "src" / (BENCHMARK_MODULES[benchmark].replace(".", "/") + ".py")
    code = _hash_files(BENCH_SRC, spec_path)
    data_version = os.environ.get("DATA_VERSION", "").strip()
    data = _benchmark_input_fingerprint(benchmark, os.environ.get("DATA_FINGERPRINT", "deep").strip().lower(), data_version)
    suffix = f"__dv-{data_version}" if data_version else ""
    if os.environ.get("STRICT_MODE", "").strip().lower() not in ("", "0", "false", "no"):
        suffix += "__strict"
    return f"{params}__code-{code}__data-{data}{suffix}"


def cached_bench(benchmark: str, tag: str, **kwargs):
    path = CACHE_DIR / "benchmark" / f"{benchmark}__{tag}.pkl"
    if path.exists():
        try:
            return pickle.loads(path.read_bytes())
        except Exception:
            pass  # degraded/partial cache -> rebuild
    with _cache_lock(path):
        if path.exists():
            try:
                return pickle.loads(path.read_bytes())
            except Exception:
                pass
        with perf.measure(f"bench.load/{benchmark}", tag=tag):
            bench = GI.get_input(benchmark, root=INPUT_ROOT, **kwargs)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = _atomic_tmp(path)
        try:
            with open(tmp, "wb") as f:
                pickle.dump(bench, f)
            os.replace(tmp, path)
        finally:
            tmp.unlink(missing_ok=True)
        return bench


def build_model(name: str, **kwargs) -> ModelWrapper:
    mod_path, cls_name = MODELS[name]
    cls = getattr(importlib.import_module(mod_path), cls_name)
    sig = inspect.signature(cls)
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
        return cls(**kwargs)
    accepted = set(sig.parameters)
    return cls(**{k: v for k, v in kwargs.items() if k in accepted})


def _model_source_files(model_name: str) -> list[Path]:
    mod_src = REPO / "src" / (MODELS[model_name][0].replace(".", "/") + ".py")
    files = [mod_src]
    try:
        text = mod_src.read_text()
    except OSError:
        return files
    names = set(re.findall(r"utils\.models\.([A-Za-z_]\w*)", text))
    for imports in re.findall(r"from\s+utils\.models\s+import\s+([^\n]+)", text):
        names.update(part.strip().split(" as ", 1)[0] for part in imports.split(","))
    for name in sorted(names):
        util_path = REPO / "src" / "utils" / "models" / f"{name}.py"
        if util_path.exists() and util_path not in files:
            files.append(util_path)
            try:
                util_text = util_path.read_text()
            except OSError:
                util_text = ""
            if re.search(r"from\s+utils\.gputils\s+import|import\s+utils\.gputils", util_text):
                gputils_path = REPO / "src" / "utils" / "gputils.py"
                if gputils_path not in files:
                    files.append(gputils_path)
    return files


# Per-model checkpoint identity folded into embedding cache keys.
_CHECKPOINT_SPECS: dict[str, tuple[str, str | None, str | None, str | None]] = {
    "presto": ("hf", None, None,
               "torchgeo/presto@44835fba5116ed5f000d5eea3973655985bf765b:model-f317d103.pth"
               "+code@11e207a668a34336ced1d8e492a1bd5849b96c4a"),
    "olmoearth": ("hf", None, None,
                  "allenai/OlmoEarth-v1_1-Base@4ef31d45f80c1d4fcce18f9cde40c1b5e4d96cf4"
                  "+code@0e11e448946f8ca259593435194e9faa14a58c77"),
    "galileo": ("hf", None, None, "nasaharvest/galileo@f039dd5dde966a931baeda47eb680fa89b253e4e:base"),
    "agrifm": ("local", "AGRIFM_WEIGHTS", "models/agrifm/AgriFM.pth", None),
    "tessera": ("local", "TESSERA_WEIGHTS", "models/tessera/tessera_v1_1_mpc_encoder.pt", None),
    "raw": ("raw", None, None, None),  # no checkpoint; identity is the RAW_MODE featurization
}
_HF_DEFAULTS = {
    "presto": ("models/presto/model-f317d103.pth", ("model-f317d103.pth",)),
    "olmoearth": ("models/olmoearth-v1_1-base", ("config.json", "weights.pth")),
    "galileo": ("models/galileo/base", ("config.json", "model.pt")),
}


@functools.lru_cache(maxsize=128)
def _hash_file_content(path_str: str, size: int, mtime_ns: int) -> str:
    h = hashlib.sha256()
    h.update(str(size).encode())
    with open(path_str, "rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):  # 8 MiB blocks
            h.update(block)
    return h.hexdigest()[:16]


def _local_weight_id(path: Path) -> str:
    if path.is_dir():
        files = sorted(p for p in path.rglob("*") if p.is_file())
        return _hash_str("|".join(f"{f.relative_to(path)}:{_local_weight_id(f)}" for f in files))
    st = path.stat()
    return _hash_file_content(str(path), st.st_size, st.st_mtime_ns)


_INPUT_BASE = INPUT_ROOT.parent


def _checkpoint_fingerprint(model_name: str, weights_override: str | Path | None = None) -> str:
    if weights_override:
        p = Path(weights_override).expanduser()
        if p.exists():
            return _hash_str(f"override:{p.name}:{_local_weight_id(p)}")[:10]
        return _hash_str(f"override-missing:{p}")[:10]
    spec = _CHECKPOINT_SPECS.get(model_name)
    if not spec:
        return ""
    kind, env_name, default_rel, hf_pin = spec
    if kind == "hf":
        local = _HF_DEFAULTS.get(model_name)
        if local:
            rel, required = local
            path = _INPUT_BASE / rel
            if all((path / r).exists() for r in required) if path.is_dir() else path.exists():
                return _hash_str(f"hf:{hf_pin}:local:{_local_weight_id(path)}")[:10]
        return _hash_str(f"hf:{hf_pin}")[:10]
    if kind == "raw":  # raw baseline has no weights; its output depends on the featurization mode
        return _hash_str(f"raw:{os.environ.get('RAW_MODE', 'flatten')}")[:10]
    if kind == "local":
        env_val = os.environ.get(env_name) if env_name else None
        path = Path(env_val).expanduser() if env_val else _INPUT_BASE / default_rel
        if not path.exists():
            return _hash_str(f"local-missing:{path}")[:10]  # stable even before the file is staged
        return _hash_str(f"local:{path.name}:{_local_weight_id(path)}")[:10]
    return ""


def _emb_sig(bench: Any, model_name: str, tag: str, weights_override=None) -> str:
    base = f"n{bench.n_samples}_d{EMB_DTYPE}_b{_hash_str(tag)}_e{_hash_files(*_model_source_files(model_name))}"
    fp = _checkpoint_fingerprint(model_name, weights_override)
    return f"{base}_w{fp}" if fp else base


def embedding_cache_path(bench: Any, benchmark: str, model_name: str, tag: str, weights_override=None) -> Path:
    """Checkpoint-aware path for one benchmark-level frozen embedding matrix."""
    return EMBEDDINGS_DIR / benchmark / model_name / _emb_sig(bench, model_name, tag, weights_override) / "baseline.npy"


def dense_embedding_cache_dir(
    bench: PastisBenchmark, benchmark: str, model_name: str, tag: str, weights_override=None
) -> Path:
    """Checkpoint-aware root directory for one dense frozen-feature tile cache."""
    return EMBEDDINGS_DIR / benchmark / model_name / _emb_sig(bench, model_name, tag, weights_override) / "baseline"


def load_cached_embeddings(bench: Any, benchmark: str, model_name: str, tag: str, weights_override=None) -> np.ndarray:
    """Load an existing frozen embedding matrix, failing if it has not been built."""
    path = embedding_cache_path(bench, benchmark, model_name, tag, weights_override)
    if not path.exists():
        raise MissingEmbeddingCache(
            f"Embedding cache not found for {model_name}/{benchmark}: {path}. "
            "Run with RUN_STAGES including 'gen_embeddings' first."
        )
    return np.load(path).astype(np.float32, copy=False)


def require_dense_cache(bench: PastisBenchmark, benchmark: str, model_name: str, tag: str, weights_override=None) -> Path:
    """Return an existing dense tile cache root, failing if it is absent OR INCOMPLETE.

    Validates the EXACT expected set of tile IDs (every patch × every subtile) -- both the
    feature and the label file must exist for each. A bare count would let a missing expected
    tile be masked by an extra stale one; matching identities catches that.
    """
    root = dense_embedding_cache_dir(bench, benchmark, model_name, tag, weights_override)
    if not root.exists():
        raise MissingEmbeddingCache(
            f"Dense embedding cache not found for {model_name}/{benchmark}: {root}. "
            "Run with RUN_STAGES including 'gen_embeddings' first."
        )
    tiles_per_axis = 128 // bench.tile_size
    expected: set[Path] = set()
    missing: list[str] = []
    for patch in bench.patches:
        fold_dir = root / f"fold_{patch.fold}"
        target = None
        target_path = getattr(patch, "target_path", None)
        if target_path is not None:
            target = np.load(target_path, mmap_mode="r")[0]
        for r in range(tiles_per_axis):
            for c in range(tiles_per_axis):
                tile_id = f"{patch.patch_id}_{r}_{c}"
                if target is not None:
                    row = r * bench.tile_size
                    col = c * bench.tile_size
                    labels = target[row:row + bench.tile_size, col:col + bench.tile_size]
                    if not np.any(labels != bench.ignore_index):
                        continue
                label_path = fold_dir / f"{tile_id}.labels.npy"
                expected.add(label_path)
                if not ((fold_dir / f"{tile_id}.npy").exists() and label_path.exists()):
                    missing.append(tile_id)
    if missing:
        raise MissingEmbeddingCache(
            f"Dense cache INCOMPLETE for {model_name}/{benchmark}: {len(missing)} expected tiles "
            f"are missing a feature/label file in {root} (e.g. {missing[:3]}). Extraction was "
            "interrupted -- re-run RUN_STAGES with 'gen_embeddings'."
        )
    # Reject EXTRA tiles too: load_dense_samples globs every *.labels.npy, so a stale tile left from
    # a different descriptor (e.g. a changed max_samples) would silently be evaluated.
    extra = sorted(str(p.relative_to(root)) for p in root.glob("fold_*/*.labels.npy") if p not in expected)
    if extra:
        raise MissingEmbeddingCache(
            f"Dense cache for {model_name}/{benchmark} has {len(extra)} UNEXPECTED tile(s) not in the "
            f"descriptor in {root} (e.g. {extra[:3]}). These would contaminate evaluation -- clear the "
            "cache dir and re-run 'gen_embeddings'."
        )
    return root


def extract_and_cache(
    bench: Any, benchmark, model_name, tag, overwrite=False, **enc_kwargs
) -> np.ndarray:
    """Cache and return the frozen-model embedding matrix for a benchmark."""
    path = embedding_cache_path(bench, benchmark, model_name, tag, enc_kwargs.get("weights_path"))
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        return np.load(path).astype(np.float32, copy=False)
    with _cache_lock(path):
        if path.exists() and not overwrite:
            return np.load(path).astype(np.float32, copy=False)
        model = build_model(model_name, **enc_kwargs)
        print(f"  encoding {model_name} ...", flush=True)
        with perf.measure(f"encode/{model_name}", n_samples=bench.n_samples):
            arr = model.encode(bench)
        if not hasattr(model, "_macs"):
            try:
                model._macs = model.compute_macs()
            except Exception as exc:
                print(f"  (compute_macs failed for {model_name}: {type(exc).__name__}; recording 0)", flush=True)
                model._macs = 0
        perf.log_static(
            f"encode/{model_name}",
            macs=model._macs * bench.n_samples,
            n_samples=bench.n_samples,
            n_features=model.embedding_dim,
        )
        arr = arr.astype(EMB_DTYPE, copy=False)
        tmp = _atomic_tmp(path)
        try:
            with open(tmp, "wb") as f:
                np.save(f, arr)
            os.replace(tmp, path)
        finally:
            tmp.unlink(missing_ok=True)
        return arr.astype(np.float32, copy=False)


def extract_dense_and_cache(
    bench: PastisBenchmark,
    benchmark: str,
    model_name: str,
    tag: str,
    overwrite: bool = False,
    **enc_kwargs,
) -> Path:
    """Encode a lazy spatial benchmark one tile at a time."""
    root = dense_embedding_cache_dir(bench, benchmark, model_name, tag, enc_kwargs.get("weights_path"))
    root.mkdir(parents=True, exist_ok=True)
    model = None
    for tile_id, fold, tile, labels in bench.iter_tiles(cache_root=root, overwrite=overwrite):
        if len(labels) == 0:
            continue
        fold_dir = root / f"fold_{fold}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        feature_path = fold_dir / f"{tile_id}.npy"
        label_path = fold_dir / f"{tile_id}.labels.npy"
        if feature_path.exists() and label_path.exists() and not overwrite:
            continue
        with _cache_lock(feature_path):
            if feature_path.exists() and label_path.exists() and not overwrite:
                continue
            if model is None:
                model = build_model(model_name, **enc_kwargs)
            with perf.measure(f"encode_dense/{model_name}", tile=tile_id, fold=fold, n_pixels=len(labels)):
                features = model.encode_dense(tile) if hasattr(model, "encode_dense") else model.encode(tile.pixel_benchmark())
            if features.shape[0] != labels.shape[0]:
                raise ValueError(f"Dense model returned {features.shape[0]} rows for {labels.shape[0]} valid pixels")
            for path, values, dtype in ((feature_path, features, EMB_DTYPE), (label_path, labels, "uint8")):
                tmp = _atomic_tmp(path)
                try:
                    with open(tmp, "wb") as handle:
                        np.save(handle, np.asarray(values, dtype=dtype))
                    os.replace(tmp, path)
                finally:
                    tmp.unlink(missing_ok=True)
    return root


def _dense_label_paths(emb_dir: Path, folds: set[int], patch_ids: set[int] | None = None) -> list[Path]:
    """Sorted ``*.labels.npy`` paths for ``folds``, optionally restricted to an original-patch set."""
    paths = sorted(
        path
        for fold in sorted(folds)
        for path in (emb_dir / f"fold_{fold}").glob("*.labels.npy")
    )
    if patch_ids is not None:
        wanted = {int(p) for p in patch_ids}
        paths = [p for p in paths if int(p.name.split("_", 1)[0]) in wanted]
    return paths


def dense_fold_patches(emb_dir: Path, folds: set[int]) -> list[int]:
    return sorted({int(p.name.split("_", 1)[0]) for p in _dense_label_paths(emb_dir, folds)})


def load_dense_samples(
    emb_dir: Path,
    folds: set[int],
    max_pixels: int,
    seed: int,
    patch_ids: set[int] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    label_paths = _dense_label_paths(emb_dir, folds, patch_ids)
    if not label_paths:
        raise FileNotFoundError(f"No dense caches for folds {sorted(folds)} (patches={patch_ids}) under {emb_dir}")
    rng = np.random.default_rng(seed)
    lengths = [len(np.load(p, mmap_mode="r")) for p in label_paths]
    total = int(sum(lengths))
    chosen = np.arange(total) if total <= max_pixels else np.sort(rng.choice(total, size=max_pixels, replace=False))
    feature_parts: list[np.ndarray] = []
    label_parts: list[np.ndarray] = []
    group_parts: list[np.ndarray] = []
    tile_parts: list[np.ndarray] = []
    patch_parts: list[np.ndarray] = []
    start = 0
    for tile_idx, (label_path, n) in enumerate(zip(label_paths, lengths, strict=True)):
        stop = start + n
        take = chosen[(chosen >= start) & (chosen < stop)] - start
        start = stop
        if len(take) == 0:
            continue
        feature_path = label_path.with_name(label_path.name.replace(".labels.npy", ".npy"))
        labels = np.load(label_path, mmap_mode="r")
        features = np.load(feature_path, mmap_mode="r")
        fold = int(label_path.parent.name.removeprefix("fold_"))
        patch_id = int(label_path.name.split("_", 1)[0])  # filename is "{patch}_{row}_{col}.labels.npy"
        feature_parts.append(np.asarray(features[take], dtype=np.float32))
        label_parts.append(np.asarray(labels[take], dtype=np.int64))
        group_parts.append(np.full(len(take), fold, dtype=np.int64))
        tile_parts.append(np.full(len(take), tile_idx, dtype=np.int64))
        patch_parts.append(np.full(len(take), patch_id, dtype=np.int64))
    x = np.concatenate(feature_parts)
    y = np.concatenate(label_parts)
    groups = np.concatenate(group_parts)
    tile_ids = np.concatenate(tile_parts)
    patch_ids_out = np.concatenate(patch_parts)
    return x, y, groups, tile_ids, patch_ids_out


def iter_dense_tiles(emb_dir: Path, folds: set[int], patch_ids: set[int] | None = None):
    for label_path in _dense_label_paths(emb_dir, folds, patch_ids):
        feature_path = label_path.with_name(label_path.name.replace(".labels.npy", ".npy"))
        labels = np.asarray(np.load(label_path), dtype=np.int64)
        features = np.asarray(np.load(feature_path), dtype=np.float32)
        yield features, labels
