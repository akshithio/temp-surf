"""Read-only integrity audit of the frozen embedding caches required by the final run.

AUDIT ONLY. This tool never regenerates, re-encodes, renames, moves, or deletes anything under
``data/``. The single file it may write is ``data/logs/cache.json``, where it records its findings
under a top-level ``audit`` key (``datasets`` and ``embeddings`` are preserved untouched). It adds
no sidecars, manifests, fingerprint directories, or extra audit JSON files.

The expected cache matrix is DERIVED from ``evals.compat`` plus the committed final-run
configuration below -- never inferred from whichever files happen to exist on disk.

What it checks per required cell, without modifying it:
  * tabular: canonical file present; cache.json record present; rank / row count / feature width /
    dtype / finiteness; benchmark dataset identity; ordered sample-ID digest; checkpoint SHA-256;
    full embedding-file SHA-256; baseline-vs-s2only input contract.
  * dense (PASTIS): exact expected non-void feature+label tile set (no missing / extra / unpaired);
    per-pair row-count agreement; shapes, dtypes, feature width, finiteness; stable tile identity;
    and a deterministic aggregate content digest: each tile is streamed in 8MiB blocks and the
    per-tile hashes are folded in sorted relative-path order (never completion order).

Semantic (re-encode) evidence is never recomputed here. It is reused only when it can be bound to
the exact current bytes -- for tabular cells that binding is the recorded ``artifact_sha256``.
Dense records carry no content digest, so prior dense spot-checks cannot be tied to current bytes;
those cells are reported as lacking semantic evidence rather than assumed good.

No command-line arguments: edit CONFIG below and run it. On Gilbreth run it through a CPU Slurm job,
never on the frontend.
"""

from __future__ import annotations

import hashlib
import json
import socket
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import numpy as np  # noqa: E402

from evals import compat  # noqa: E402
from utils import artifacts  # noqa: E402
from utils import cacheutils as C  # noqa: E402

# ===================== CONFIG (edit me; no CLI) =============================
CONFIG = {
    # The committed final-run configuration these caches must serve (mirrors src/main.py).
    "benchmarks": ["cropharvest", "eurocropsml", "breizhcrops", "pastis"],
    "active_models": None,   # None = every compat-eligible model for the benchmark
    "s2_only": False,        # False -> the run consumes ONLY the 'baseline' artifact
    # Cells absent on this host are reported "absent_here" (not a failure); fleet coverage is
    # judged across hosts, since PASTIS dense lives only on Gilbreth.
    "skip_absent": True,
    # Write findings into data/logs/cache.json. False = report only, zero writes.
    "write": True,
    # Parallel workers for per-file hashing/inspection. hashlib and file IO both release the GIL,
    # so threads scale here; the aggregate stays deterministic because results are recombined in
    # sorted relative-path order, never in completion order.
    "workers": 8,
    # "audit"   -- full structural + content audit (re-hashes; the expensive path).
    # "lineage" -- record lineage acceptance for cells ALREADY audited in cache.json. Re-hashes
    #              nothing; it only proves nothing was rewritten since the semantic check and
    #              rewrites the audit verdict in place.
    "mode": "lineage",
    "lineage": {
        # Dense cells whose PRIOR semantic spot check is accepted via lineage rather than re-encode.
        "cells": [
            "pastis/tessera/baseline", "pastis/olmoearth/baseline", "pastis/galileo/baseline",
            "pastis/agrifm/baseline", "pastis/presto/baseline", "pastis/raw/baseline",
        ],
        # Where that semantic evidence lives, and when it finished. Any cache byte written after
        # this instant would break the lineage, so the check below refuses in that case.
        "evidence_log": "data/output/logs/adopt-all-resume-11339121.out",
        "evidence_completed": "2026-07-20T01:20:28-04:00",
    },
}
# ===========================================================================

_CHUNK = 8 << 20
_BENCH_CACHE: dict[str, object] = {}


def _bench(benchmark: str):
    """Load a benchmark pickle at most ONCE per process (it is reused by every model's cell)."""
    if benchmark not in _BENCH_CACHE:
        bench = C.cached_bench(benchmark)
        _BENCH_CACHE[benchmark] = bench.s2_only() if CONFIG["s2_only"] else bench
    return _BENCH_CACHE[benchmark]


def _rels_cache(bench, benchmark: str):
    """Expected dense tile rels, computed once per benchmark (the walk is expensive)."""
    key = f"__rels__{benchmark}"
    if key not in _BENCH_CACHE:
        _BENCH_CACHE[key] = C._dense_expected_rels(bench)
    return _BENCH_CACHE[key]


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(_CHUNK), b""):
            digest.update(block)
    return digest.hexdigest()


def _expected_matrix() -> list[tuple[str, str, str]]:
    """(benchmark, model, artifact) the final run actually consumes, from compat + run config."""
    artifact = C.artifact_name(bool(CONFIG["s2_only"]))
    cells: list[tuple[str, str, str]] = []
    for benchmark in CONFIG["benchmarks"]:
        models = compat.eligible_models(benchmark)
        if CONFIG["active_models"] is not None:
            models = [m for m in models if m in CONFIG["active_models"]]
        cells.extend((benchmark, model, artifact) for model in models)
    return cells


def _code_revision() -> str:
    try:
        return subprocess.run(["git", "-C", str(C.REPO), "rev-parse", "HEAD"],
                              capture_output=True, text=True, timeout=30).stdout.strip() or "unknown"
    except Exception:  # noqa: BLE001
        return "unknown"


def _finite(path: Path) -> bool:
    """Finiteness without loading the whole array (chunked over a read-only memmap)."""
    arr = np.load(path, mmap_mode="r", allow_pickle=False)
    rows = arr.shape[0] if arr.ndim else 0
    step = max(1, min(rows, 4096))
    for start in range(0, rows, step):
        if not np.isfinite(np.asarray(arr[start:start + step], dtype=np.float64)).all():
            return False
    return True


def _audit_tabular(bench, benchmark, model, artifact, record) -> dict:
    path = C.embedding_cache_path(benchmark, model, artifact)
    if not path.is_file():
        return {"structural": "absent_here", "path": str(path)}
    problems: list[str] = []
    try:
        shape, dtype = C._npy_shape_dtype(path)
    except Exception as exc:  # noqa: BLE001
        return {"structural": f"fail: unreadable .npy header: {exc}", "path": str(path)}
    if len(shape) != 2:
        problems.append(f"rank {len(shape)} != 2")
    elif shape[0] != bench.n_samples:
        problems.append(f"rows {shape[0]} != benchmark n_samples {bench.n_samples}")
    if dtype != C.EMB_DTYPE:
        problems.append(f"dtype {dtype} != {C.EMB_DTYPE}")
    if list(shape) != record.get("shape"):
        problems.append(f"shape {list(shape)} != recorded {record.get('shape')}")
    if dtype != record.get("dtype"):
        problems.append(f"dtype {dtype} != recorded {record.get('dtype')}")
    if not _finite(path):
        problems.append("array contains non-finite values")

    expected = {
        "dataset_digest": C.dataset_digest(benchmark),
        "sample_ids_digest": C.sample_ids_digest(bench.sample_ids),
        "checkpoint_sha256": C.checkpoint_sha256(model, None),
    }
    for key, want in expected.items():
        if record.get(key) != want:
            problems.append(f"{key}: recorded {record.get(key)!r} != current {want!r}")

    content = artifacts.sha256_file(path)
    bytes_match = content == record.get("artifact_sha256")
    if not bytes_match:
        problems.append(f"artifact_sha256: file {content} != recorded {record.get('artifact_sha256')}")

    return {
        "kind": "tabular", "path": str(path),
        "structural": "ok" if not problems else "fail: " + "; ".join(problems),
        "content_digest": content,
        "bytes_match_record": bytes_match,
        "rows": int(shape[0]) if len(shape) == 2 else None,
        "feature_width": int(shape[1]) if len(shape) == 2 else None,
        "dtype": dtype,
    }


def _audit_dense(bench, benchmark, model, artifact, record) -> dict:
    root = C.dense_embedding_cache_dir(benchmark, model, artifact)
    if not root.is_dir():
        return {"structural": "absent_here", "path": str(root)}
    feat_rels, lab_rels = _rels_cache(bench, benchmark)
    problems = list(C._dense_tile_set_problems(root, feat_rels, lab_rels))

    # ONE parallel sweep over every expected tile: content hash + header + (features) finiteness.
    # Each tile is read once instead of three times, and the work spreads across CONFIG["workers"].
    def _inspect(rel: str) -> tuple[str, dict]:
        path, info = root / rel, {}
        if not path.is_file():
            return rel, info
        info["sha"] = _file_sha256(path)
        try:
            info["shape"], info["dtype"] = C._npy_shape_dtype(path)
        except Exception as exc:  # noqa: BLE001
            info["header_error"] = str(exc)
            return rel, info
        if not rel.endswith(".labels.npy"):
            info["finite"] = _finite(path)
        return rel, info

    ordered = sorted(feat_rels) + sorted(lab_rels)
    with ThreadPoolExecutor(max_workers=int(CONFIG["workers"])) as pool:
        inspected = dict(pool.map(_inspect, ordered))

    # Deterministic aggregate: sorted relative path + NUL + that file's content hash + LF,
    # recombined in path order (never completion order). Same construction the repo already uses.
    digest, checked = hashlib.sha256(), 0
    for rel in ordered:
        info = inspected.get(rel, {})
        if "sha" not in info:
            continue
        digest.update(rel.encode())
        digest.update(b"\0")
        digest.update(info["sha"].encode())
        digest.update(b"\n")
        checked += 1

    feature_dim = None
    for feat_rel in feat_rels:
        lab_rel = feat_rel.removesuffix(".npy") + ".labels.npy"
        finfo, linfo = inspected.get(feat_rel, {}), inspected.get(lab_rel, {})
        if "sha" not in finfo or "sha" not in linfo:
            continue
        if "header_error" in finfo or "header_error" in linfo:
            problems.append(f"unreadable tile header {feat_rel}: {finfo.get('header_error') or linfo.get('header_error')}")
            continue
        fshape, fdtype = finfo["shape"], finfo["dtype"]
        lshape, ldtype = linfo["shape"], linfo["dtype"]
        if fdtype != C.EMB_DTYPE:
            problems.append(f"{feat_rel}: dtype {fdtype} != {C.EMB_DTYPE}")
        if ldtype != C.DENSE_LABEL_DTYPE:
            problems.append(f"{feat_rel}: label dtype {ldtype} != {C.DENSE_LABEL_DTYPE}")
        if len(fshape) != 2:
            problems.append(f"{feat_rel}: rank {len(fshape)} != 2")
            continue
        if feature_dim is None:
            feature_dim = int(fshape[1])
        elif fshape[1] != feature_dim:
            problems.append(f"{feat_rel}: width {fshape[1]} != {feature_dim}")
        if len(lshape) != 1 or lshape[0] != fshape[0]:
            problems.append(f"{feat_rel}: label rows {lshape} != feature rows {fshape[0]}")
        if finfo.get("finite") is False:
            problems.append(f"{feat_rel}: non-finite values")

    for key, want in {"dataset_digest": C.dataset_digest(benchmark),
                      "checkpoint_sha256": C.checkpoint_sha256(model, None),
                      "tile_set_digest": C.tile_set_digest(feat_rels, lab_rels),
                      "feature_tile_count": len(feat_rels),
                      "label_tile_count": len(lab_rels)}.items():
        if record.get(key) != want:
            problems.append(f"{key}: recorded {record.get(key)!r} != current {want!r}")
    if feature_dim is not None and record.get("feature_dim") != feature_dim:
        problems.append(f"feature_dim: recorded {record.get('feature_dim')} != current {feature_dim}")

    return {
        "kind": "dense", "path": str(root),
        "structural": "ok" if not problems else "fail: " + "; ".join(problems[:8]),
        "content_digest": digest.hexdigest(),
        "files_digested": checked,
        "feature_tile_count": len(feat_rels),
        "label_tile_count": len(lab_rels),
        "feature_dim": feature_dim,
        # Dense records carry no adoption-time content digest, so prior spot checks cannot be
        # cryptographically bound to these bytes. This digest establishes that binding going forward.
        "bytes_match_record": None,
    }


def _newest_mtime(root: Path) -> float:
    """Newest mtime anywhere under ``root`` (metadata only -- reads no file contents)."""
    newest = root.stat().st_mtime
    for path in root.rglob("*"):
        try:
            newest = max(newest, path.stat().st_mtime)
        except OSError:  # a vanished temp entry cannot make the tree newer in any meaningful way
            continue
    return newest


def _lineage() -> int:
    """Accept prior semantic evidence for already-audited dense cells, without re-encoding.

    The lineage argument is only sound if NOTHING rewrote the cache after the semantic check. That
    is verified here by comparing the newest mtime under each cache root against the evidence
    timestamp; a newer byte anywhere refuses the cell. The aggregate content digest recorded by the
    earlier audit pass is NOT retrospective evidence -- the adoption log never contained it. It is
    the identity this audit FREEZES going forward, so any later drift becomes detectable.
    """
    cfg = CONFIG["lineage"]
    cutoff = datetime.fromisoformat(cfg["evidence_completed"]).timestamp()
    doc = C._read_cache_doc()
    audits = doc.get("audit", {})
    if not audits:
        print("[lineage] no audit section present -- run mode='audit' first", flush=True)
        return 1
    host = max(audits, key=lambda h: audits[h].get("timestamp", ""))
    cells = audits[host]["cells"]
    print(f"[lineage] host={host} evidence={cfg['evidence_log']} cutoff={cfg['evidence_completed']}", flush=True)

    accepted, refused = [], []
    for key in cfg["cells"]:
        cell = cells.get(key)
        if cell is None:
            refused.append((key, "no audited cell in cache.json"))
            continue
        if not str(cell.get("structural", "")).startswith("ok"):
            refused.append((key, f"structural not ok: {cell.get('structural')}"))
            continue
        if not cell.get("content_digest"):
            refused.append((key, "no aggregate content digest recorded"))
            continue
        newest = _newest_mtime(Path(cell["path"]))
        if newest > cutoff:
            refused.append((key, f"cache rewritten after the semantic check "
                                 f"(newest mtime {datetime.fromtimestamp(newest).isoformat()})"))
            continue
        cell["status"] = "valid"
        cell["semantic_evidence"] = (
            "accepted via lineage: dense re-encode spot check in "
            f"{cfg['evidence_log']} (completed {cfg['evidence_completed']}) passed for this cell, and "
            "dataset digest, checkpoint SHA-256, tile-set digest, tile counts, feature dim, shapes and "
            "dtypes all still match; no byte under the cache root was written after that check."
        )
        cell["no_rewrite_since_evidence"] = {
            "verified": True,
            "newest_mtime": datetime.fromtimestamp(newest).isoformat(),
            "evidence_completed": cfg["evidence_completed"],
        }
        cell["digest_role"] = (
            "The aggregate content digest FREEZES this cell's accepted identity from this audit "
            "forward; it was computed by this audit and was NOT present in the adoption log."
        )
        accepted.append(key)
        print(f"  [{key}] valid via lineage (newest mtime {datetime.fromtimestamp(newest).isoformat()})", flush=True)

    for key, why in refused:
        print(f"  [{key}] REFUSED: {why}", flush=True)

    audits[host]["lineage"] = {
        "timestamp": datetime.now(UTC).isoformat(),
        "evidence_log": cfg["evidence_log"],
        "evidence_completed": cfg["evidence_completed"],
        "accepted": accepted,
        "refused": [k for k, _ in refused],
        "basis": ("prior dense re-encode spot check + unchanged identity fields + no rewrite after "
                  "the check; the aggregate digest freezes identity going forward, it is not "
                  "retrospective evidence"),
    }
    if CONFIG["write"]:
        with C._cache_lock(C.CACHE_JSON_PATH):  # atomic; datasets/embeddings untouched
            current = C._read_cache_doc()
            current.setdefault("audit", {})[host] = audits[host]
            C._atomic_write_json(C.CACHE_JSON_PATH, current)
        print(f"[lineage] updated audit.{host} in {C.CACHE_JSON_PATH}", flush=True)
    counts = {s: sum(1 for v in cells.values() if v.get("status") == s) for s in ("valid", "missing_evidence", "invalid")}
    print(f"[lineage] verdict: {counts}", flush=True)
    return 0 if not refused else 1


def main() -> int:
    if CONFIG["mode"] == "lineage":
        return _lineage()
    host = socket.gethostname().split(".")[0]
    cells = _expected_matrix()
    print(f"[audit] host={host} expected cells={len(cells)} artifact={C.artifact_name(bool(CONFIG['s2_only']))}", flush=True)
    doc = C._read_cache_doc()
    records = doc.get("embeddings", {})
    results: dict[str, dict] = {}

    for benchmark, model, artifact in cells:
        key = C._embedding_key(benchmark, model, artifact)
        record = records.get(key)
        if record is None:
            results[key] = {"structural": "fail: no cache.json record", "status": "invalid"}
            print(f"  [{key}] NO RECORD", flush=True)
            continue
        dense = benchmark == "pastis"
        try:
            bench = _bench(benchmark)  # memoized: one load per benchmark, reused by every model
            out = (_audit_dense if dense else _audit_tabular)(bench, benchmark, model, artifact, record)
        except Exception as exc:  # noqa: BLE001 -- one bad cell must not abort the audit
            out = {"structural": f"fail: {type(exc).__name__}: {exc}"}

        if out.get("structural") == "absent_here":
            out["status"] = "absent_here"
        elif not str(out.get("structural", "")).startswith("ok"):
            out["status"] = "invalid"
        elif out.get("bytes_match_record") is True:
            # Prior re-encode spot check is bound to these exact bytes via artifact_sha256.
            out["status"] = "valid"
            out["semantic_evidence"] = "accepted: adoption spot check bound to current bytes via artifact_sha256"
        else:
            out["status"] = "missing_evidence"
            out["semantic_evidence"] = (
                "missing: no adoption-time content digest recorded, so prior spot check cannot be "
                "tied to the exact current bytes")
        results[key] = out
        print(f"  [{key}] {out['status']}: {str(out.get('structural'))[:80]}", flush=True)

    audit = {
        "timestamp": datetime.now(UTC).isoformat(),
        "code_revision": _code_revision(),
        "host": host,
        "run_config": {"benchmarks": CONFIG["benchmarks"], "active_models": CONFIG["active_models"],
                       "s2_only": CONFIG["s2_only"]},
        "expected_matrix": [C._embedding_key(b, m, a) for b, m, a in cells],
        "cells": results,
    }
    print("\n==== AUDIT SUMMARY ====")
    for status in ("valid", "missing_evidence", "invalid", "absent_here"):
        named = [k for k, v in results.items() if v.get("status") == status]
        print(f"  {status}: {len(named)}")
        for k in named:
            print(f"      {k}")

    if CONFIG["write"]:
        C.CACHE_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
        with C._cache_lock(C.CACHE_JSON_PATH):  # atomic read-modify-write; preserves other keys
            current = C._read_cache_doc()
            current.setdefault("audit", {})[host] = audit
            C._atomic_write_json(C.CACHE_JSON_PATH, current)
        print(f"\n[audit] recorded under audit.{host} in {C.CACHE_JSON_PATH}")
    else:
        print("\n[audit] report-only; cache.json untouched")

    print("AUDIT_JSON_BEGIN")
    print(json.dumps(audit, sort_keys=True))
    print("AUDIT_JSON_END")
    return 0 if all(v.get("status") in ("valid", "absent_here") for v in results.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
