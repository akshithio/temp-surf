"""Cache-keyed benchmark assembly + model embedding extraction.

Every cache key includes a hash of the code that produced it (loader, model source),
so any code change self-invalidates the cache.  The caller never has to clear
``data/cache/`` by hand.
"""

from __future__ import annotations

import hashlib
import importlib
import inspect
import os
import pickle
from pathlib import Path
from typing import Any

import numpy as np

from dataio import get_input as GI
from evals.benchmarks.pastis_r import PastisBenchmark
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
    "pastis_r": "evals/benchmarks/pastis_r",
}

MODELS: dict[str, tuple[str, str]] = {
    "presto": ("models.presto", "PrestoModel"),
    "olmoearth": ("models.olmoearth", "OlmoEarthModel"),
    "galileo": ("models.galileo", "GalileoModel"),
    "agrifm": ("models.agrifm", "AgriFMModel"),
    "tessera": ("models.tessera", "TesseraModel"),
}

EMB_DTYPE = "float16"


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


def _input_fingerprint(bench_dir: Path) -> str:
    """Cheap (non-recursive) fingerprint of a dataset dir -- folds DATA re-staging
    into the key. Catches added/removed/renamed top-level entries and mtime bumps
    (what rsync/unzip do); a surgical in-place edit of one deep file may not bump it."""
    h = hashlib.sha256()
    try:
        for e in sorted(bench_dir.iterdir(), key=lambda x: x.name):
            st = e.stat()
            h.update(f"{e.name}:{st.st_size}:{int(st.st_mtime)}".encode())
    except OSError:
        h.update(b"<missing>")
    return h.hexdigest()[:10]


def bench_tag(benchmark: str, kwargs: dict) -> str:
    """Identity of an assembled benchmark: params + loader-code hash + input-data hash.

    Any change to the params, to get_input.py, to the benchmark's spec file, or to the
    staged input files yields a new tag -- so the bench pickle AND the embeddings derived
    from it self-invalidate.
    """
    params = "_".join(f"{k}-{kwargs[k]}" for k in sorted(kwargs)) or "default"
    spec_path = REPO / "src" / (BENCHMARK_MODULES[benchmark].replace(".", "/") + ".py")
    code = _hash_files(BENCH_SRC, spec_path)
    data = _input_fingerprint(INPUT_ROOT / benchmark)
    return f"{params}__code-{code}__data-{data}"


def cached_bench(benchmark: str, tag: str, **kwargs):
    """Load the assembled Benchmark from a content-keyed pickle cache (build on miss).

    Assembling a benchmark reads tens of thousands of small files -- the dominant CPU
    cost; re-runs load one pickle. ``tag`` (see ``bench_tag``) makes the cache
    self-invalidating on loader-code or input-data changes. Lives under ``data/cache``.
    """
    path = CACHE_DIR / "benchmark" / f"{benchmark}__{tag}.pkl"
    if path.exists():
        try:
            return pickle.loads(path.read_bytes())
        except Exception:
            pass  # degraded/partial cache -> rebuild
    with perf.measure(f"bench.load/{benchmark}", tag=tag):
        bench = GI.get_input(benchmark, root=INPUT_ROOT, **kwargs)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(pickle.dumps(bench))
    os.replace(tmp, path)
    return bench


def build_model(name: str, **kwargs) -> Any:
    """Instantiate a model, passing only the kwargs it actually accepts.

    (Lets us hand ``device`` to everything; models without that field silently ignore it.)
    """
    mod_path, cls_name = MODELS[name]
    cls = getattr(importlib.import_module(mod_path), cls_name)
    accepted = set(inspect.signature(cls).parameters)
    return cls(**{k: v for k, v in kwargs.items() if k in accepted})


def _model_source_files(model_name: str) -> list[Path]:
    mod_src = REPO / "src" / (MODELS[model_name][0].replace(".", "/") + ".py")
    return [mod_src]


def extract_and_cache(
    bench: Any, benchmark, model_name, tag, overwrite="skip", **enc_kwargs
) -> np.ndarray:
    """Cache and return the frozen-model embedding matrix for a benchmark."""
    sig = f"n{bench.n_samples}_b{_hash_str(tag)}_e{_hash_files(*_model_source_files(model_name))}"
    out = EMBEDDINGS_DIR / benchmark / model_name / sig
    out.mkdir(parents=True, exist_ok=True)
    path = out / "baseline.npy"
    if path.exists() and overwrite != "override":
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
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        np.save(f, arr.astype(EMB_DTYPE, copy=False))
    os.replace(tmp, path)
    return arr


def extract_dense_and_cache(
    bench: PastisBenchmark,
    benchmark: str,
    model_name: str,
    tag: str,
    overwrite: str = "skip",
    **enc_kwargs,
) -> Path:
    """Encode a lazy spatial benchmark one tile at a time.

    Each tile is an independent cache entry, so a full PASTIS-R run never holds
    the release or its complete feature tensor in memory and resumes at tile
    granularity after interruption.
    """
    sig = f"n{bench.n_samples}_b{_hash_str(tag)}_e{_hash_files(*_model_source_files(model_name))}"
    root = EMBEDDINGS_DIR / benchmark / model_name / sig / "baseline"
    root.mkdir(parents=True, exist_ok=True)
    model = None
    for tile_id, fold, tile, labels in bench.iter_tiles():
        fold_dir = root / f"fold_{fold}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        feature_path = fold_dir / f"{tile_id}.npy"
        label_path = fold_dir / f"{tile_id}.labels.npy"
        if feature_path.exists() and label_path.exists() and overwrite != "override":
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
            tmp = path.with_name(path.name + ".tmp")
            with open(tmp, "wb") as handle:
                np.save(handle, np.asarray(values, dtype=dtype))
            os.replace(tmp, path)
    return root


def load_dense_samples(
    emb_dir: Path,
    folds: set[int],
    max_pixels: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load a deterministic bounded pixel sample from cached dense tile features.

    Returns ``(features, labels, groups, tile_ids)`` where ``tile_ids`` is an
    integer index per pixel identifying which tile it came from (for per-tile
    metrics like per-tile mIoU).
    """
    label_paths = sorted(
        path
        for fold in sorted(folds)
        for path in (emb_dir / f"fold_{fold}").glob("*.labels.npy")
    )
    if not label_paths:
        raise FileNotFoundError(f"No dense caches for folds {sorted(folds)} under {emb_dir}")
    rng = np.random.default_rng(seed)
    per_tile = max(1, int(np.ceil(max_pixels / len(label_paths))))
    feature_parts: list[np.ndarray] = []
    label_parts: list[np.ndarray] = []
    group_parts: list[np.ndarray] = []
    tile_parts: list[np.ndarray] = []
    for tile_idx, label_path in enumerate(label_paths):
        feature_path = label_path.with_name(label_path.name.replace(".labels.npy", ".npy"))
        labels = np.load(label_path, mmap_mode="r")
        features = np.load(feature_path, mmap_mode="r")
        take = rng.choice(len(labels), size=min(per_tile, len(labels)), replace=False)
        fold = int(label_path.parent.name.removeprefix("fold_"))
        feature_parts.append(np.asarray(features[take], dtype=np.float32))
        label_parts.append(np.asarray(labels[take], dtype=np.int64))
        group_parts.append(np.full(len(take), fold, dtype=np.int64))
        tile_parts.append(np.full(len(take), tile_idx, dtype=np.int64))
    x = np.concatenate(feature_parts)
    y = np.concatenate(label_parts)
    groups = np.concatenate(group_parts)
    tile_ids = np.concatenate(tile_parts)
    if len(y) > max_pixels:
        take = rng.choice(len(y), size=max_pixels, replace=False)
        x, y, groups, tile_ids = x[take], y[take], groups[take], tile_ids[take]
    return x, y, groups, tile_ids
