from __future__ import annotations

import functools
import hashlib
import importlib
import inspect
import os
import pickle
import re
import uuid
from pathlib import Path
from typing import Any

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

EMB_DTYPE = "float16"


class MissingEmbeddingCache(FileNotFoundError):
    """Raised when a probing run requests an embedding cache that is absent."""


def _hash_files(*paths: str | Path) -> str:
    """Short hash of source-file contents -- folds CODE into a cache key."""
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


def _update_file_content_hash(path: str | Path, h: Any) -> None:
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)


def _input_fingerprint(bench_dir: Path) -> str:
    """Fingerprint a dataset dir; deep mode hashes paths and bytes, top mode hashes directory stats."""
    h = hashlib.sha256()
    mode = os.environ.get("DATA_FINGERPRINT", "deep").strip().lower()
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


def bench_tag(benchmark: str, kwargs: dict) -> str:
    """Identity of an assembled benchmark: params + loader-code hash + input-data hash."""
    params = "_".join(f"{k}-{kwargs[k]}" for k in sorted(kwargs)) or "default"
    spec_path = REPO / "src" / (BENCHMARK_MODULES[benchmark].replace(".", "/") + ".py")
    code = _hash_files(BENCH_SRC, spec_path)
    data = _input_fingerprint(INPUT_ROOT / benchmark)
    data_version = os.environ.get("DATA_VERSION", "").strip()
    suffix = f"__dv-{data_version}" if data_version else ""
    # OVERWRITE_MODE changes which samples survive loading (corrupt/missing are excluded vs raise),
    # so it is part of the assembled-benchmark identity -- otherwise a permissive cache would be
    # silently reused by a strict run, skipping the strict check entirely.
    if os.environ.get("OVERWRITE_MODE", "").strip().lower() not in ("", "0", "false", "no"):
        suffix += "__strict"
    return f"{params}__code-{code}__data-{data}{suffix}"


def cached_bench(benchmark: str, tag: str, **kwargs):
    """Load the assembled Benchmark from a content-keyed pickle cache (build on miss)."""
    path = CACHE_DIR / "benchmark" / f"{benchmark}__{tag}.pkl"
    if path.exists():
        try:
            return pickle.loads(path.read_bytes())
        except Exception:
            pass  # degraded/partial cache -> rebuild
    with perf.measure(f"bench.load/{benchmark}", tag=tag):
        bench = GI.get_input(benchmark, root=INPUT_ROOT, **kwargs)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = _atomic_tmp(path)
    try:
        tmp.write_bytes(pickle.dumps(bench))
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)
    return bench


def build_model(name: str, **kwargs) -> Any:
    """Instantiate a model, passing only accepted kwargs."""
    mod_path, cls_name = MODELS[name]
    cls = getattr(importlib.import_module(mod_path), cls_name)
    accepted = set(inspect.signature(cls).parameters)
    return cls(**{k: v for k, v in kwargs.items() if k in accepted})


def _model_source_files(model_name: str) -> list[Path]:
    """Wrapper source + the model-specific util modules it imports (e.g. galileoutil,
    agrifmutils), so a change to transitive model code also invalidates the embedding cache."""
    mod_src = REPO / "src" / (MODELS[model_name][0].replace(".", "/") + ".py")
    files = [mod_src]
    try:
        text = mod_src.read_text()
    except OSError:
        return files
    for name in re.findall(r"utils\.\w+\.([A-Za-z_]\w*?util[s]?)\b", text):
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


# Per-model checkpoint identity, folded into the embedding key so a weights change invalidates
# stale embeddings. It MUST be download-independent (identical before and after an HF lazy
# download), or a gen run (weights absent) and a later probe run (weights present) would key
# differently and never hit the cache -- which is why there is no legacy fallback.
#   * HF models  -> the pinned repo/filename/variant string. Changing the intended weights means
#                   editing the wrapper, already captured by the wrapper-source hash.
#   * local models -> size+mtime of the resolved file, so replacing the weights in place (the
#                   exact stale-reuse scenario) yields a new key.
# The HF pin includes the IMMUTABLE commit revision (must match the wrapper's *_HF_REVISION),
# so a moved branch / re-uploaded weights file changes the key. value: (kind, env_var, default_rel, hf_pin)
# The HF pin includes the weights revision AND (for pip-installed external model code: presto,
# olmoearth) the pinned package revision, so changing either the weights or the model CODE
# invalidates the embedding key. presto code commit must match sync.sh; olmoearth must match the
# [tool.uv.sources] olmoearth-pretrain rev in pyproject.toml. (galileo runs on local galileoutil,
# already covered by the model-source hash.)
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


@functools.lru_cache(maxsize=128)
def _hash_file_content(path_str: str, size: int, mtime_ns: int) -> str:
    """Whole-file sha (size + bytes), memoised by (path, size, mtime) so a multi-GB checkpoint is
    read at most once per process even though the fingerprint is requested many times (per
    benchmark, and again for the run signature)."""
    h = hashlib.sha256()
    h.update(str(size).encode())
    with open(path_str, "rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):  # 8 MiB blocks
            h.update(block)
    return h.hexdigest()[:16]


def _local_weight_id(path: Path) -> str:
    """Full-content id of a local checkpoint: fully immutable (any byte change -> new id)."""
    if path.is_dir():
        files = sorted(p for p in path.rglob("*") if p.is_file())
        return _hash_str("|".join(f"{f.relative_to(path)}:{_local_weight_id(f)}" for f in files))
    st = path.stat()
    return _hash_file_content(str(path), st.st_size, st.st_mtime_ns)


# Checkpoints live under the INPUT base (data/input/models/...), NOT under INPUT_ROOT, which
# points at data/input/benchmarks. Resolve them against the base so the default agrifm/tessera
# paths match the wrappers' own resolution (otherwise the fingerprint stats a nonexistent path
# and weight replacement never invalidates).
_INPUT_BASE = INPUT_ROOT.parent


def _checkpoint_fingerprint(model_name: str, weights_override: str | Path | None = None) -> str:
    """Download-independent identity of a model's checkpoint, or '' if it has none.

    ``weights_override`` is the EFFECTIVE weights path passed to the wrapper (e.g. via
    ``enc_kwargs['weights_path']``); when given it takes precedence over the spec's default/env
    resolution, so a custom checkpoint gets its own cache key instead of sharing another's.
    """
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
    base = f"n{bench.n_samples}_b{_hash_str(tag)}_e{_hash_files(*_model_source_files(model_name))}"
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
        for r in range(tiles_per_axis):
            for c in range(tiles_per_axis):
                tile_id = f"{patch.patch_id}_{r}_{c}"
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
    arr = arr.astype(EMB_DTYPE, copy=False)  # cache dtype; round BEFORE returning so a single-process
    tmp = _atomic_tmp(path)  # run probes on the exact same values a two-stage load would.
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
    """Encode a lazy spatial benchmark one tile at a time.

    Each tile is an independent cache entry, so a full PASTIS-R run never holds
    the release or its complete feature tensor in memory and resumes at tile
    granularity after interruption.
    """
    root = dense_embedding_cache_dir(bench, benchmark, model_name, tag, enc_kwargs.get("weights_path"))
    root.mkdir(parents=True, exist_ok=True)
    model = None
    for tile_id, fold, tile, labels in bench.iter_tiles():
        fold_dir = root / f"fold_{fold}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        feature_path = fold_dir / f"{tile_id}.npy"
        label_path = fold_dir / f"{tile_id}.labels.npy"
        if feature_path.exists() and label_path.exists() and not overwrite:
            continue
        if model is None:
            model = build_model(model_name, **enc_kwargs)
        with perf.measure(
            f"encode_dense/{model_name}",
            tile=tile_id,
            fold=fold,
            n_pixels=len(labels),
        ):
            if hasattr(model, "encode_dense"):
                features = model.encode_dense(tile)
            else:
                features = model.encode(tile.pixel_benchmark())
        if features.shape[0] != labels.shape[0]:
            raise ValueError(
                f"Dense model returned {features.shape[0]} rows for {labels.shape[0]} valid pixels"
            )
        for path, values, dtype in (
            (feature_path, features, EMB_DTYPE),
            (label_path, labels, "uint8"),
        ):
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
    """Sorted unique original 128x128 patch ids cached for ``folds`` (filename-only, no pixel load).

    Used by the dense target sweep to draw its 80/20 split over whole patches.
    """
    return sorted({int(p.name.split("_", 1)[0]) for p in _dense_label_paths(emb_dir, folds)})


def load_dense_samples(
    emb_dir: Path,
    folds: set[int],
    max_pixels: int,
    seed: int,
    patch_ids: set[int] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load a deterministic bounded pixel SAMPLE from cached dense tile features -- for TRAINING.

    Returns ``(features, labels, groups, tile_ids, patch_ids)``; ``patch_ids`` (per pixel) is the
    original 128x128 patch each pixel belongs to. ``patch_ids=`` restricts to those patches (used to
    build the target few-shot / oracle training pool). For EVALUATION use :func:`iter_dense_tiles`
    (every pixel, streamed) -- this capped sample is appropriate only for fitting the probe.
    """
    label_paths = _dense_label_paths(emb_dir, folds, patch_ids)
    if not label_paths:
        raise FileNotFoundError(f"No dense caches for folds {sorted(folds)} (patches={patch_ids}) under {emb_dir}")
    rng = np.random.default_rng(seed)
    per_tile = max(1, int(np.ceil(max_pixels / len(label_paths))))
    feature_parts: list[np.ndarray] = []
    label_parts: list[np.ndarray] = []
    group_parts: list[np.ndarray] = []
    tile_parts: list[np.ndarray] = []
    patch_parts: list[np.ndarray] = []
    for tile_idx, label_path in enumerate(label_paths):
        feature_path = label_path.with_name(label_path.name.replace(".labels.npy", ".npy"))
        labels = np.load(label_path, mmap_mode="r")
        features = np.load(feature_path, mmap_mode="r")
        take = rng.choice(len(labels), size=min(per_tile, len(labels)), replace=False)
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
    if len(y) > max_pixels:
        take = rng.choice(len(y), size=max_pixels, replace=False)
        x, y, groups, tile_ids, patch_ids_out = x[take], y[take], groups[take], tile_ids[take], patch_ids_out[take]
    return x, y, groups, tile_ids, patch_ids_out


def iter_dense_tiles(emb_dir: Path, folds: set[int], patch_ids: set[int] | None = None):
    """Yield ``(features (n, D) float32, labels (n,) int64)`` for EVERY cached tile -- full, no
    subsample -- optionally restricted to an original-patch set.

    This is the full-fold streaming evaluation source: scoring a probe tile-by-tile over every
    valid pixel gives the exact fold mIoU (and credible per-tile worst-case), instead of the noisy
    estimate a capped pixel sample would give. One tile (<=64x64 pixels) is in memory at a time.
    """
    for label_path in _dense_label_paths(emb_dir, folds, patch_ids):
        feature_path = label_path.with_name(label_path.name.replace(".labels.npy", ".npy"))
        labels = np.asarray(np.load(label_path), dtype=np.int64)
        features = np.asarray(np.load(feature_path), dtype=np.float32)
        yield features, labels