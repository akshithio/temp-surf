from __future__ import annotations

import contextlib
import fcntl
import hashlib
import importlib
import inspect
import json
import os
import pickle
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
OUTPUT_DIR = SCRATCH / "output"  # canonical results location (no override)
EMBEDDINGS_DIR = CACHE_DIR / "embeddings"

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
DENSE_LABEL_DTYPE = "uint8"  # dense label tiles are written as uint8 (class ids 0..IGNORE_INDEX)


class MissingEmbeddingCache(FileNotFoundError):
    pass


class ModelWrapper(Protocol):
    embedding_dim: int

    def encode(self, bench: Any) -> np.ndarray: ...


def _atomic_tmp(path: Path) -> Path:
    return path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")


@contextlib.contextmanager
def _cache_lock(path: Path):
    """Exclusive advisory flock serializing writers to one cache artifact (or tile)."""
    lock_path = path.with_name(f".{path.name}.lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with open(lock_path, "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


# The one production loader contract. Loader BEHAVIOR, not cache identity: the canonical pickle
# always reflects exactly this configuration, so it is never encoded in the filename.
_CANONICAL_LOADER = {"max_samples": None, "shuffle": True, "seed": 0}


def benchmark_cache_path(benchmark: str) -> Path:
    """The single canonical pickle path for a supported benchmark."""
    if benchmark not in BENCHMARK_MODULES:
        raise KeyError(f"Unknown benchmark {benchmark!r}. Known: {sorted(BENCHMARK_MODULES)}")
    return CACHE_DIR / "benchmark" / f"{benchmark}.pkl"


def cached_bench(benchmark: str):
    """Load-or-build the ONE canonical benchmark pickle at data/cache/benchmark/<benchmark>.pkl.

    The pickle is always built with the fixed production contract (max_samples=None, shuffle=True,
    seed=0), which is loader behavior -- not cache identity -- so it is not encoded in the filename.
    No loader kwargs are accepted and no per-parameter fallback file is ever created; utilities that
    need a subset must call ``dataio.get_input`` directly and must not write this cache. An
    unreadable pickle is reported and rebuilt in place under the writer lock.
    """
    path = benchmark_cache_path(benchmark)

    def _try_read(p: Path):
        try:
            with p.open("rb") as f:  # streaming unpickle -- no full-file bytes allocation
                return pickle.load(f), None
        except Exception as exc:  # unreadable / truncated / partial pickle
            return None, exc

    if path.exists():
        bench, exc = _try_read(path)
        if bench is not None:
            return bench
        print(f"  !! benchmark cache unreadable at {path}: {type(exc).__name__}: {exc} -- rebuilding", flush=True)

    with _cache_lock(path):
        if path.exists():  # a concurrent writer may have just (re)built it
            bench, exc = _try_read(path)
            if bench is not None:
                return bench
            print(f"  !! benchmark cache still unreadable at {path}: {type(exc).__name__}: {exc} -- rebuilding", flush=True)
        with perf.measure(f"bench.load/{benchmark}"):
            bench = GI.get_input(benchmark, root=INPUT_ROOT, **_CANONICAL_LOADER)
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


# Per-model checkpoint resolution -- used to LOCATE the actual checkpoint whose full SHA-256 is the
# frozen-run identity (see _resolve_checkpoint_path / checkpoint_sha256).
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


_INPUT_BASE = INPUT_ROOT.parent


# ============================================================================
# Frozen final-run embedding cache: fixed readable paths + identity manifests.
# The cryptic ``n..._b..._e..._w...`` fingerprint directory is gone; identity now lives in a
# readable ``<artifact>.manifest.json`` beside the artifact and is validated on every load.
# ============================================================================

# --- full checkpoint content digest (untruncated identity) -------------------

_CHECKPOINT_SHA_CACHE: dict[str, str] = {}


def _resolve_checkpoint_path(model_name: str, weights_override: str | Path | None = None) -> Path | None:
    """The actual on-disk checkpoint file/dir the wrapper will load, or None (e.g. raw)."""
    if weights_override:
        p = Path(weights_override).expanduser()
        return p if p.exists() else None
    spec = _CHECKPOINT_SPECS.get(model_name)
    if not spec:
        return None
    kind, env_name, default_rel, _hf_pin = spec
    if kind == "hf":
        local = _HF_DEFAULTS.get(model_name)
        if local:
            rel, required = local
            path = _INPUT_BASE / rel
            ok = all((path / r).exists() for r in required) if path.is_dir() else path.exists()
            if ok:
                return path
        return None
    if kind == "local":
        env_val = os.environ.get(env_name) if env_name else None
        path = Path(env_val).expanduser() if env_val else _INPUT_BASE / default_rel
        return path if path.exists() else None
    return None  # raw: no checkpoint


def _content_sha256(path: Path) -> str:
    """Full SHA-256 of a file's content, or a canonical aggregate over a directory (sorted relative
    POSIX paths + each file's SHA). Reuses ``artifacts.sha256_file`` -- no duplicate hash helper."""
    from utils import artifacts

    path = Path(path)
    if path.is_dir():
        h = hashlib.sha256()
        for rel in sorted(p.relative_to(path).as_posix() for p in path.rglob("*") if p.is_file()):
            h.update(rel.encode())
            h.update(b"\0")
            h.update((artifacts.sha256_file(path / rel) or "").encode())
            h.update(b"\n")
        return h.hexdigest()
    return artifacts.sha256_file(path) or ""


def checkpoint_sha256(model_name: str, weights_override: str | Path | None = None) -> str:
    """Full SHA-256 identity of the checkpoint a model will load (dir-aware, memoized per process).
    raw / no-checkpoint models hash a stable token so identity is always defined."""
    path = _resolve_checkpoint_path(model_name, weights_override)
    if path is None:
        token = f"raw:{os.environ.get('RAW_MODE', 'flatten')}" if model_name == "raw" else f"no-checkpoint:{model_name}"
        return hashlib.sha256(token.encode()).hexdigest()
    key = str(path.resolve())
    if key not in _CHECKPOINT_SHA_CACHE:
        _CHECKPOINT_SHA_CACHE[key] = _content_sha256(path)
    return _CHECKPOINT_SHA_CACHE[key]


def sample_ids_digest(sample_ids: Any) -> str:
    """Ordered digest over per-sample IDs -- pins the sample SET and its ROW ORDER."""
    if sample_ids is None:
        raise ValueError("sample_ids are required to identify a tabular embedding")
    h = hashlib.sha256()
    for sid in sample_ids:
        h.update(str(sid).encode())
        h.update(b"\0")
    return h.hexdigest()


def tile_set_digest(feature_rels: list[str], label_rels: list[str]) -> str:
    """Ordered digest over the expected dense tile identities -- the dense analogue of sample IDs."""
    h = hashlib.sha256()
    for rel in sorted(feature_rels) + sorted(label_rels):
        h.update(rel.encode())
        h.update(b"\0")
    return h.hexdigest()


# --- portable dataset content digest (written once by the preflight utility) -

DATASET_DIGEST_DIR = CACHE_DIR / "dataset_digests"


def dataset_digest(benchmark: str) -> str:
    """The frozen portable content digest for a benchmark's inputs.

    Produced ONCE by ``tools/preflight_dataset_digests`` (a CPU job on Gilbreth) and read here.
    Absent -> a hard error: the frozen run requires the preflight to have established and
    cross-checked dataset identity first; there is no weaker fallback.
    """
    path = DATASET_DIGEST_DIR / f"{benchmark}.txt"
    try:
        digest = path.read_text().strip()
    except OSError as exc:
        raise MissingEmbeddingCache(
            f"No frozen dataset digest for {benchmark!r} at {path}. Run "
            "tools/preflight_dataset_digests to hash and cross-check the inputs first."
        ) from exc
    if not digest:
        raise MissingEmbeddingCache(f"Empty dataset digest file for {benchmark!r} at {path}.")
    return digest


# --- frozen-run identity (recorded in the RUN manifest, NOT in embedding validity) -------

_FROZEN_IDENTITY: dict[str, Any] | None = None


def frozen_run_identity() -> dict[str, Any]:
    """Final commit + clean-tree state, for the run manifest. Memoized per process.

    The commit is read straight from the deployed ``.git`` checkout -- no env override. It gates
    the RUN, never the embedding (an embedding is identified by its content). A checkout with no
    ``.git`` yields ``final_commit=None``; the launch gate refuses to run without provenance.
    """
    global _FROZEN_IDENTITY
    if _FROZEN_IDENTITY is None:
        from utils import artifacts
        git = artifacts.git_identity(REPO)
        _FROZEN_IDENTITY = {
            "final_commit": git.get("commit"),
            "clean": (git.get("dirty") is False) if git.get("commit") else None,
            "tree_identity": git.get("tree_identity"),
        }
    return _FROZEN_IDENTITY


# --- fixed canonical paths ---------------------------------------------------


def artifact_name(s2_only: bool) -> str:
    """The one path component reflecting embedding-affecting inputs. Not a variant framework."""
    return "s2only" if s2_only else "baseline"


def _cell_dir(benchmark: str, model_name: str) -> Path:
    return EMBEDDINGS_DIR / benchmark / model_name


def embedding_cache_path(benchmark: str, model_name: str, artifact: str = "baseline") -> Path:
    return _cell_dir(benchmark, model_name) / f"{artifact}.npy"


def embedding_manifest_path(benchmark: str, model_name: str, artifact: str = "baseline") -> Path:
    return _cell_dir(benchmark, model_name) / f"{artifact}.manifest.json"


def dense_embedding_cache_dir(benchmark: str, model_name: str, artifact: str = "baseline") -> Path:
    return _cell_dir(benchmark, model_name) / artifact


def dense_manifest_path(benchmark: str, model_name: str, artifact: str = "baseline") -> Path:
    return _cell_dir(benchmark, model_name) / f"{artifact}.manifest.json"


# --- manifest + npy IO helpers ----------------------------------------------


def _read_manifest(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def _write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    tmp = _atomic_tmp(path)
    try:
        with open(tmp, "w") as f:
            json.dump(manifest, f, sort_keys=True, indent=2)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def _npy_shape_dtype(path: Path) -> tuple[tuple[int, ...], str]:
    """Inspect shape and dtype through NumPy's public mmap loader without reading the array body."""
    array = np.load(path, mmap_mode="r", allow_pickle=False)
    return array.shape, str(array.dtype)


def embedding_digest(benchmark: str, model_name: str, artifact: str = "baseline", *, dense: bool = False) -> str | None:
    """The embedding's recorded CONTENT digest -- artifact SHA (tabular) or tile-set digest (dense).
    This is what the run manifest references to bind results to a specific embedding."""
    man_path = dense_manifest_path(benchmark, model_name, artifact) if dense else embedding_manifest_path(benchmark, model_name, artifact)
    manifest = _read_manifest(man_path)
    if not manifest:
        return None
    return manifest.get("tile_set_digest") if dense else manifest.get("artifact_sha256")


# --- expected dense tile set (shared by build + load) ------------------------


def _dense_expected_rels(bench: PastisBenchmark) -> tuple[list[str], list[str]]:
    """Sorted relative POSIX identities of every non-void (feature, label) tile the descriptor
    expects. A void tile (all ignore_index) is skipped exactly as extraction skips it."""
    tiles_per_axis = 128 // bench.tile_size
    feature_rels: list[str] = []
    label_rels: list[str] = []
    for patch in bench.patches:
        target = None
        target_path = getattr(patch, "target_path", None)
        if target_path is not None:
            target = np.load(target_path, mmap_mode="r")[0]
        for r in range(tiles_per_axis):
            for c in range(tiles_per_axis):
                tile_id = f"{patch.patch_id}_{r}_{c}"
                if target is not None:
                    row, col = r * bench.tile_size, c * bench.tile_size
                    labels = target[row:row + bench.tile_size, col:col + bench.tile_size]
                    if not np.any(labels != bench.ignore_index):
                        continue
                feature_rels.append(f"fold_{patch.fold}/{tile_id}.npy")
                label_rels.append(f"fold_{patch.fold}/{tile_id}.labels.npy")
    return sorted(feature_rels), sorted(label_rels)


def _dense_tile_set_problems(root: Path, feat_rels: list[str], lab_rels: list[str]) -> list[str]:
    """Exact tile-set completeness of a dense cache dir against the descriptor's expected rel sets:
    missing (feature OR label) tiles, plus UNEXPECTED feature files AND UNEXPECTED label files.

    One helper shared by the runtime load path (``_dense_state``) and one-time adoption, so both
    refuse the identical set -- a foreign or stale tile of either kind can never pass unnoticed.
    """
    problems: list[str] = []
    missing = [rel for rel in feat_rels + lab_rels if not (root / rel).exists()]
    if missing:
        problems.append(f"{len(missing)} expected feature/label tile(s) missing (e.g. {missing[:3]})")
    if root.exists():
        on_disk = {p.relative_to(root).as_posix() for p in root.glob("fold_*/*.npy")}
        disk_labels = {rel for rel in on_disk if rel.endswith(".labels.npy")}
        extra_features = sorted((on_disk - disk_labels) - set(feat_rels))
        extra_labels = sorted(disk_labels - set(lab_rels))
        if extra_features:
            problems.append(f"{len(extra_features)} UNEXPECTED feature tile(s) not in the descriptor (e.g. {extra_features[:3]})")
        if extra_labels:
            problems.append(f"{len(extra_labels)} UNEXPECTED label tile(s) not in the descriptor (e.g. {extra_labels[:3]})")
    return problems


def _tabular_state(bench, benchmark, model_name, artifact, weights_override):
    """('absent'|'ok'|'mismatch', array_or_None, problems).

    absent   -- no manifest -> the cache was never completed (regenerate).
    ok       -- the sidecar certifies this array for THIS run (returns the loaded array).
    mismatch -- a sidecar exists but disagrees (refuse; the operator must delete the leaf).
    """
    art_path = embedding_cache_path(benchmark, model_name, artifact)
    man_path = embedding_manifest_path(benchmark, model_name, artifact)
    manifest = _read_manifest(man_path)
    if manifest is None:
        return "absent", None, []
    problems: list[str] = []
    expected = {
        "benchmark": benchmark, "model": model_name, "artifact": artifact,
        "checkpoint_sha256": checkpoint_sha256(model_name, weights_override),
        "dataset_digest": dataset_digest(benchmark),
        "sample_ids_digest": sample_ids_digest(bench.sample_ids),
    }
    for key, want in expected.items():
        if manifest.get(key) != want:
            problems.append(f"{key}: {manifest.get(key)!r} != {want!r}")
    if not art_path.exists():
        problems.append(f"array missing: {art_path}")
    else:
        try:
            shape, dtype = _npy_shape_dtype(art_path)
            if list(shape) != manifest.get("shape"):
                problems.append(f"shape {list(shape)} != manifest {manifest.get('shape')}")
            if dtype != manifest.get("dtype"):
                problems.append(f"dtype {dtype} != manifest {manifest.get('dtype')}")
        except Exception as exc:  # noqa: BLE001 -- a malformed header is itself a validity failure
            problems.append(f"unreadable .npy header: {exc}")
    if problems:
        return "mismatch", None, problems
    return "ok", np.load(art_path).astype(np.float32, copy=False), []


def load_cached_embeddings(bench: Any, benchmark: str, model_name: str, artifact: str = "baseline", weights_override=None) -> np.ndarray:
    """Load a frozen embedding matrix, validating its sidecar manifest.

    Raises MissingEmbeddingCache if the sidecar is absent (never built) or disagrees with THIS run
    (benchmark/model/artifact/checkpoint/dataset/sample-order/shape/dtype). The array bytes are not
    re-hashed on load.
    """
    state, arr, problems = _tabular_state(bench, benchmark, model_name, artifact, weights_override)
    if state == "ok":
        return arr
    if state == "absent":
        raise MissingEmbeddingCache(
            f"Embedding cache not built for {model_name}/{benchmark}/{artifact} (no manifest). "
            "Run RUN_STAGES including 'gen_embeddings'."
        )
    raise MissingEmbeddingCache(
        f"Embedding cache for {model_name}/{benchmark}/{artifact} does not match this run -- REFUSING:\n  - "
        + "\n  - ".join(problems) + "\nDelete the cache leaf to rebuild it deliberately."
    )


def _dense_state(bench, benchmark, model_name, artifact, weights_override):
    """('absent'|'ok'|'mismatch', root, problems) -- same three-state contract as tabular.

    Keeps the exact expected-tile-set completeness check (missing / unexpected) plus cheap identity
    (checkpoint, dataset, tile-set digest, counts, feature dim, dtype). Tiles are NOT re-hashed.
    """
    root = dense_embedding_cache_dir(benchmark, model_name, artifact)
    man_path = dense_manifest_path(benchmark, model_name, artifact)
    manifest = _read_manifest(man_path)
    if manifest is None:
        return "absent", root, []
    feat_rels, lab_rels = _dense_expected_rels(bench)
    problems: list[str] = []
    expected = {
        "benchmark": benchmark, "model": model_name, "artifact": artifact,
        "checkpoint_sha256": checkpoint_sha256(model_name, weights_override),
        "dataset_digest": dataset_digest(benchmark),
        "tile_set_digest": tile_set_digest(feat_rels, lab_rels),
        "feature_tile_count": len(feat_rels), "label_tile_count": len(lab_rels),
    }
    for key, want in expected.items():
        if manifest.get(key) != want:
            problems.append(f"{key}: {manifest.get(key)!r} != {want!r}")
    problems += _dense_tile_set_problems(root, feat_rels, lab_rels)
    if feat_rels and (root / feat_rels[0]).exists():  # cheap header spot-check (NOT every tile)
        try:
            shape, dtype = _npy_shape_dtype(root / feat_rels[0])
        except Exception as exc:  # noqa: BLE001 -- a malformed header is itself a validity failure
            problems.append(f"unreadable dense tile header {feat_rels[0]}: {exc}")
        else:
            if dtype != manifest.get("dtype"):
                problems.append(f"dtype {dtype} != manifest {manifest.get('dtype')}")
            if (int(shape[1]) if len(shape) > 1 else 0) != manifest.get("feature_dim"):
                problems.append(f"feature_dim {shape[1] if len(shape) > 1 else 0} != manifest {manifest.get('feature_dim')}")
    if problems:
        return "mismatch", root, problems
    return "ok", root, []


def require_dense_cache(bench: PastisBenchmark, benchmark: str, model_name: str, artifact: str = "baseline", weights_override=None) -> Path:
    """Return a certified dense tile cache root, or raise: absent (never built) or mismatch (refuse)."""
    state, root, problems = _dense_state(bench, benchmark, model_name, artifact, weights_override)
    if state == "ok":
        return root
    if state == "absent":
        raise MissingEmbeddingCache(
            f"Dense cache not built for {model_name}/{benchmark}/{artifact} (no manifest). "
            "Run RUN_STAGES including 'gen_embeddings'."
        )
    raise MissingEmbeddingCache(
        f"Dense cache for {model_name}/{benchmark}/{artifact} does not match this run -- REFUSING:\n  - "
        + "\n  - ".join(problems) + "\nDelete the cache dir to rebuild it deliberately."
    )


def extract_and_cache(bench: Any, benchmark, model_name, artifact: str = "baseline", **enc_kwargs) -> np.ndarray:
    """Encode and publish the frozen embedding matrix + its sidecar manifest.

    A matching sidecar is a hit; a MISMATCHED sidecar is refused (the operator must delete the leaf
    to rebuild -- this frozen run never auto-replaces a completed artifact). Publication under the
    existing writer lock: the array is written atomically, then the manifest is written LAST, so an
    absent manifest always means "incomplete".
    """
    from utils import artifacts

    weights_override = enc_kwargs.get("weights_path")
    art_path = embedding_cache_path(benchmark, model_name, artifact)
    man_path = embedding_manifest_path(benchmark, model_name, artifact)
    art_path.parent.mkdir(parents=True, exist_ok=True)

    with _cache_lock(art_path):
        state, arr, problems = _tabular_state(bench, benchmark, model_name, artifact, weights_override)
        if state == "ok":
            return arr
        if state == "mismatch":
            raise MissingEmbeddingCache(
                f"Refusing to overwrite a completed {model_name}/{benchmark}/{artifact} that does not "
                f"match this run:\n  - " + "\n  - ".join(problems) + "\nDelete the leaf to rebuild it."
            )
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
        perf.log_static(f"encode/{model_name}", macs=model._macs * bench.n_samples,
                        n_samples=bench.n_samples, n_features=model.embedding_dim)
        arr = np.ascontiguousarray(arr.astype(EMB_DTYPE, copy=False))
        tmp = _atomic_tmp(art_path)
        try:
            with open(tmp, "wb") as f:
                np.save(f, arr)
            os.replace(tmp, art_path)
        finally:
            tmp.unlink(missing_ok=True)
        _write_manifest(man_path, {  # manifest LAST -> its presence certifies a complete artifact
            "schema": 1, "benchmark": benchmark, "model": model_name, "artifact": artifact,
            "checkpoint_sha256": checkpoint_sha256(model_name, weights_override),
            "dataset_digest": dataset_digest(benchmark),
            "sample_ids_digest": sample_ids_digest(bench.sample_ids),
            "shape": [int(x) for x in arr.shape], "dtype": str(arr.dtype),
            "artifact_sha256": artifacts.sha256_file(art_path),
        })
    return arr.astype(np.float32, copy=False)


def _encode_dense_into(bench: PastisBenchmark, model_name: str, root: Path, enc_kwargs: dict) -> None:
    """Encode every non-void tile into ``root`` one tile at a time -- resumable (existing tiles are
    skipped) with per-tile atomic writes and per-tile locks."""
    root.mkdir(parents=True, exist_ok=True)
    model = None
    for tile_id, fold, tile, labels in bench.iter_tiles(cache_root=root, overwrite=False):
        if len(labels) == 0:
            continue
        fold_dir = root / f"fold_{fold}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        feature_path = fold_dir / f"{tile_id}.npy"
        label_path = fold_dir / f"{tile_id}.labels.npy"
        if feature_path.exists() and label_path.exists():
            continue
        with _cache_lock(feature_path):
            if feature_path.exists() and label_path.exists():
                continue
            if model is None:
                model = build_model(model_name, **enc_kwargs)
            with perf.measure(f"encode_dense/{model_name}", tile=tile_id, fold=fold, n_pixels=len(labels)):
                features = model.encode_dense(tile) if hasattr(model, "encode_dense") else model.encode(tile.pixel_benchmark())
            if features.shape[0] != labels.shape[0]:
                raise ValueError(f"Dense model returned {features.shape[0]} rows for {labels.shape[0]} valid pixels")
            for path, values, dtype in ((feature_path, features, EMB_DTYPE), (label_path, labels, DENSE_LABEL_DTYPE)):
                tmp = _atomic_tmp(path)
                try:
                    with open(tmp, "wb") as handle:
                        np.save(handle, np.asarray(values, dtype=dtype))
                    os.replace(tmp, path)
                finally:
                    tmp.unlink(missing_ok=True)


def _build_dense_manifest(bench, benchmark, model_name, artifact, root, weights_override) -> dict[str, Any]:
    """The slim dense sidecar: identity + tile counts + tile-set digest (no per-file arrays)."""
    feat_rels, lab_rels = _dense_expected_rels(bench)
    dtype, feature_dim = EMB_DTYPE, 0
    if feat_rels:
        shape, dtype = _npy_shape_dtype(root / feat_rels[0])
        feature_dim = int(shape[1]) if len(shape) > 1 else 0
    return {
        "schema": 1, "benchmark": benchmark, "model": model_name, "artifact": artifact,
        "checkpoint_sha256": checkpoint_sha256(model_name, weights_override),
        "dataset_digest": dataset_digest(benchmark),
        "feature_tile_count": len(feat_rels), "label_tile_count": len(lab_rels),
        "tile_set_digest": tile_set_digest(feat_rels, lab_rels),
        "feature_dim": feature_dim, "dtype": str(dtype),
    }


def extract_dense_and_cache(
    bench: PastisBenchmark, benchmark: str, model_name: str, artifact: str = "baseline", **enc_kwargs
) -> Path:
    """Encode the dense tile cache in place (resumable), verify completeness, then publish the
    manifest LAST. A matching sidecar is a hit; a mismatched one is refused (delete the dir to
    rebuild). The absent manifest during the build is exactly what marks it incomplete."""
    weights_override = enc_kwargs.get("weights_path")
    root = dense_embedding_cache_dir(benchmark, model_name, artifact)
    man_path = dense_manifest_path(benchmark, model_name, artifact)

    state, _root, problems = _dense_state(bench, benchmark, model_name, artifact, weights_override)
    if state == "ok":
        return root
    if state == "mismatch":
        raise MissingEmbeddingCache(
            f"Refusing to rebuild a completed dense {model_name}/{benchmark}/{artifact} that does not "
            f"match this run:\n  - " + "\n  - ".join(problems) + "\nDelete the cache dir to rebuild it."
        )

    _encode_dense_into(bench, model_name, root, enc_kwargs=enc_kwargs)  # per-tile locks; resumable
    man_path.parent.mkdir(parents=True, exist_ok=True)
    with _cache_lock(man_path):
        feat_rels, lab_rels = _dense_expected_rels(bench)
        missing = [rel for rel in feat_rels + lab_rels if not (root / rel).exists()]
        if missing:
            raise MissingEmbeddingCache(
                f"Dense build incomplete for {model_name}/{benchmark}/{artifact}: {len(missing)} "
                f"expected tile(s) missing (e.g. {missing[:3]})."
            )
        _write_manifest(man_path, _build_dense_manifest(bench, benchmark, model_name, artifact, root, weights_override))
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


def iter_dense_tiles(
    emb_dir: Path,
    folds: set[int],
    patch_ids: set[int] | None = None,
):
    for label_path in _dense_label_paths(emb_dir, folds, patch_ids):
        feature_path = label_path.with_name(label_path.name.replace(".labels.npy", ".npy"))
        labels = np.asarray(np.load(label_path), dtype=np.int64)
        features = np.asarray(np.load(feature_path), dtype=np.float32)
        yield features, labels
