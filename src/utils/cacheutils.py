"""Cache-keyed benchmark assembly + encoder embedding extraction.

Every cache key includes a hash of the code that produced it (loader, degrade,
encoder source), so any code change self-invalidates the cache.  The caller
never has to clear ``data/cache/`` by hand.
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
from utils import perfutils as perf

REPO = Path(__file__).resolve().parents[2]
SCRATCH = REPO / "data"
INPUT_ROOT = Path(os.environ.get("ROBUSTNESS_INPUT", REPO / "data" / "input")) / "benchmarks"
CACHE_DIR = SCRATCH / "cache"
OUTPUT_DIR = SCRATCH / "output"
GET_INPUT_SRC = REPO / "src" / "dataio" / "get_input.py"

ENCODERS: dict[str, tuple[str, str]] = {
    "presto": ("models.presto", "PrestoEncoder"),
    "olmoearth": ("models.olmoearth", "OlmoEarthEncoder"),
    "galileo": ("models.galileo", "GalileoEncoder"),
    "agrifm": ("models.agrifm", "AgriFMEncoder"),
    "tessera": ("models.tessera", "TesseraEncoder"),
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

    Any change to the params, to get_input.py (loader / degrade), or to the staged
    input files yields a new tag -- so the bench pickle AND the embeddings derived
    from it self-invalidate, and you never have to clear data/cache by hand.
    """
    params = "_".join(f"{k}-{kwargs[k]}" for k in sorted(kwargs)) or "default"
    code = _hash_files(GET_INPUT_SRC)
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


def build_encoder(name: str, **kwargs) -> Any:
    """Instantiate an encoder, passing only the kwargs it actually accepts.

    (Lets us hand ``device`` to everything; encoders without that field silently ignore it.)
    """
    mod_path, cls_name = ENCODERS[name]
    cls = getattr(importlib.import_module(mod_path), cls_name)
    accepted = set(inspect.signature(cls).parameters)
    return cls(**{k: v for k, v in kwargs.items() if k in accepted})


def _encoder_source_files(encoder_name: str) -> list[Path]:
    enc_src = REPO / "src" / (ENCODERS[encoder_name][0].replace(".", "/") + ".py")
    return [enc_src]


def extract_and_cache(
    bench, benchmark, encoder_name, tag, conditions, overwrite="skip", **enc_kwargs
) -> dict[str, np.ndarray]:
    sig = f"n{bench.n_samples}_b{_hash_str(tag)}_e{_hash_files(*_encoder_source_files(encoder_name))}"
    out = OUTPUT_DIR / "embeddings" / benchmark / encoder_name / sig
    out.mkdir(parents=True, exist_ok=True)
    emb: dict[str, np.ndarray] = {}
    encoder = None
    for name, sensor_off, tdrop in conditions:
        perf.set_identity({"encoder": encoder_name, "condition": name, "benchmark": benchmark})
        path = out / f"{name}.npy"
        if path.exists():
            if overwrite == "override":
                path.unlink()
            else:
                emb[name] = np.load(path).astype(np.float32, copy=False)
                continue
        if encoder is None:
            encoder = build_encoder(encoder_name, **enc_kwargs)
        print(f"  encoding {encoder_name}/{name} ...", flush=True)
        with perf.measure(f"encode/{encoder_name}/{name}", n_samples=bench.n_samples):
            with perf.measure(f"degrade/{name}"):
                degraded = GI.degrade(bench, sensor_off=sensor_off, temporal_drop=tdrop, seed=0)
            arr = encoder.encode(degraded)
        if not hasattr(encoder, "_macs"):
            try:
                encoder._macs = encoder.compute_macs()
            except Exception as exc:  # MACs are a diagnostic; never let profiling crash a run
                print(f"  (compute_macs failed for {encoder_name}: {type(exc).__name__}; recording 0)", flush=True)
                encoder._macs = 0
        perf.log_static(
            f"encode/{encoder_name}/{name}",
            macs=encoder._macs * bench.n_samples,
            n_samples=bench.n_samples,
            n_features=encoder.embedding_dim,
        )
        tmp = path.with_name(path.name + ".tmp")
        with open(tmp, "wb") as f:
            np.save(f, arr.astype(EMB_DTYPE, copy=False))
        os.replace(tmp, path)
        emb[name] = arr
    perf.set_identity(None)
    return emb


def extract_dense_and_cache(
    bench: GI.PastisBenchmark,
    benchmark: str,
    encoder_name: str,
    tag: str,
    conditions: list[tuple[str, str, float]],
    overwrite: str = "skip",
    **enc_kwargs,
) -> dict[str, Path]:
    """Encode a lazy spatial benchmark one tile at a time.

    Each tile is an independent cache entry, so a full PASTIS-R run never holds
    the release or its complete feature tensor in memory and resumes at tile
    granularity after interruption.
    """
    sig = f"n{bench.n_samples}_b{_hash_str(tag)}_e{_hash_files(*_encoder_source_files(encoder_name))}"
    root = OUTPUT_DIR / "embeddings" / benchmark / encoder_name / sig
    outputs: dict[str, Path] = {}
    encoder = None
    for condition, sensor_off, temporal_drop in conditions:
        condition_dir = root / condition
        condition_dir.mkdir(parents=True, exist_ok=True)
        outputs[condition] = condition_dir
        degraded = GI.degrade(bench, sensor_off=sensor_off, temporal_drop=temporal_drop, seed=0)
        for tile_id, fold, tile, labels in degraded.iter_tiles():
            fold_dir = condition_dir / f"fold_{fold}"
            fold_dir.mkdir(parents=True, exist_ok=True)
            feature_path = fold_dir / f"{tile_id}.npy"
            label_path = fold_dir / f"{tile_id}.labels.npy"
            if feature_path.exists() and label_path.exists() and overwrite != "override":
                continue
            if encoder is None:
                encoder = build_encoder(encoder_name, **enc_kwargs)
            with perf.measure(
                f"encode_dense/{encoder_name}/{condition}",
                tile=tile_id,
                fold=fold,
                n_pixels=len(labels),
            ):
                if hasattr(encoder, "encode_dense"):
                    features = encoder.encode_dense(tile)
                else:
                    features = encoder.encode(tile.pixel_benchmark())
            if features.shape[0] != labels.shape[0]:
                raise ValueError(
                    f"Dense encoder returned {features.shape[0]} rows for {labels.shape[0]} valid pixels"
                )
            for path, values, dtype in (
                (feature_path, features, EMB_DTYPE),
                (label_path, labels, "uint8"),
            ):
                tmp = path.with_name(path.name + ".tmp")
                with open(tmp, "wb") as handle:
                    np.save(handle, np.asarray(values, dtype=dtype))
                os.replace(tmp, path)
    return outputs


def load_dense_samples(
    condition_dir: Path,
    folds: set[int],
    max_pixels: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load a deterministic bounded pixel sample from cached dense tile features."""
    label_paths = sorted(
        path
        for fold in sorted(folds)
        for path in (condition_dir / f"fold_{fold}").glob("*.labels.npy")
    )
    if not label_paths:
        raise FileNotFoundError(f"No dense caches for folds {sorted(folds)} under {condition_dir}")
    rng = np.random.default_rng(seed)
    per_tile = max(1, int(np.ceil(max_pixels / len(label_paths))))
    feature_parts: list[np.ndarray] = []
    label_parts: list[np.ndarray] = []
    group_parts: list[np.ndarray] = []
    for label_path in label_paths:
        feature_path = label_path.with_name(label_path.name.replace(".labels.npy", ".npy"))
        labels = np.load(label_path, mmap_mode="r")
        features = np.load(feature_path, mmap_mode="r")
        take = rng.choice(len(labels), size=min(per_tile, len(labels)), replace=False)
        fold = int(label_path.parent.name.removeprefix("fold_"))
        feature_parts.append(np.asarray(features[take], dtype=np.float32))
        label_parts.append(np.asarray(labels[take], dtype=np.int64))
        group_parts.append(np.full(len(take), fold, dtype=np.int64))
    x = np.concatenate(feature_parts)
    y = np.concatenate(label_parts)
    groups = np.concatenate(group_parts)
    if len(y) > max_pixels:
        take = rng.choice(len(y), size=max_pixels, replace=False)
        x, y, groups = x[take], y[take], groups[take]
    return x, y, groups
