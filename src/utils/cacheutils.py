"""Cache-keyed benchmark assembly + encoder embedding extraction.

Every cache key includes a hash of the code that produced it (loader, corrupt,
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
from utils import perf

REPO = Path(__file__).resolve().parents[2]
SCRATCH = Path(os.environ.get("ROBUSTNESS_SCRATCH", REPO / "data"))
INPUT_ROOT = REPO / "data" / "input"
CACHE_DIR = SCRATCH / "cache"
OUTPUT_DIR = SCRATCH / "output"
GET_INPUT_SRC = REPO / "src" / "dataio" / "get_input.py"

ENCODERS: dict[str, tuple[str, str]] = {
    "presto": ("models.presto", "PrestoEncoder"),
    "olmoearth": ("models.olmoearth", "OlmoEarthEncoder"),
    "tessera": ("models.tessera", "TesseraEncoder"),
    "agrifm": ("models.agrifm", "AgriFMEncoder"),
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

    Any change to the params, to get_input.py (loader / corrupt), or to the staged
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
            pass  # corrupt/partial cache -> rebuild
    with perf.measure(f"bench.load/{benchmark}", tag=tag):
        bench = GI.get_input(benchmark, root=INPUT_ROOT, **kwargs)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(pickle.dumps(bench))
    os.replace(tmp, path)
    return bench


def build_encoder(name: str, **kwargs) -> Any:
    """Instantiate an encoder, passing only the kwargs it actually accepts.

    (Lets us hand ``device`` to everything; encoders without that field -- e.g.
    TesseraEncoder -- silently ignore it.)
    """
    mod_path, cls_name = ENCODERS[name]
    cls = getattr(importlib.import_module(mod_path), cls_name)
    accepted = set(inspect.signature(cls).parameters)
    return cls(**{k: v for k, v in kwargs.items() if k in accepted})


def extract_and_cache(
    bench, benchmark, encoder_name, tag, conditions, overwrite="skip", **enc_kwargs
) -> dict[str, np.ndarray]:
    enc_src = REPO / "src" / (ENCODERS[encoder_name][0].replace(".", "/") + ".py")
    sig = f"n{bench.n_samples}_b{_hash_str(tag)}_e{_hash_files(enc_src)}"
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
            with perf.measure(f"corrupt/{name}"):
                corrupted = GI.corrupt(bench, sensor_off=sensor_off, temporal_drop=tdrop, seed=0)
            arr = encoder.encode(corrupted)
        if not hasattr(encoder, "_macs"):
            encoder._macs = encoder.compute_macs()
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
