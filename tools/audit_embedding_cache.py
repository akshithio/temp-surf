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
    and a deterministic aggregate content digest (see :func:`_dense_digest_order`).

DENSE DIGEST CONSTRUCTION -- FROZEN. The tile order is ``sorted(feature_rels)`` followed by
``sorted(label_rels)``; it is NOT one global sort of all paths (a global sort would interleave
``<tile>.labels.npy`` with ``<tile>.npy``). Each tile is streamed in 8MiB blocks, and per-tile
hashes are folded as ``rel NUL <hex> LF`` in that order. Every fleet verifier MUST use
:func:`_dense_digest_order` so it reproduces the already-published digests.

Semantic (re-encode) evidence is never recomputed here.
  * Tabular cells: the prior spot check is bound CRYPTOGRAPHICALLY to the current bytes, because
    adoption recorded ``artifact_sha256`` and the file still hashes to it.
  * Dense cells: no adoption-time content digest exists, so no such proof is possible. They are
    accepted only under an explicit OPERATIONAL-LINEAGE assumption (see :func:`_lineage`), which is
    weaker than proof and is recorded as such.

No command-line arguments: edit CONFIG below and run it. On Gilbreth run it through a CPU Slurm job,
never on the frontend.
"""

from __future__ import annotations

import hashlib
import json
import re
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
    # False (the setting for the canonical Gilbreth source audit): a required cell that is absent
    # here is a FAILURE ("absent_required") and exits nonzero. True is only for auditing a machine
    # that legitimately holds a subset, and then absence is reported without failing.
    "skip_absent": False,
    # Write findings into data/logs/cache.json. False = report only, zero writes.
    "write": True,
    # Parallel workers for per-file hashing/inspection. hashlib and file IO both release the GIL,
    # so threads scale here; the aggregate stays deterministic because results are recombined in
    # the frozen features-then-labels order (see _dense_digest_order), never in completion order.
    "workers": 8,
    # "audit"   -- full structural + content audit (re-hashes; the expensive path).
    # "lineage" -- record OPERATIONAL-lineage acceptance for cells ALREADY audited in cache.json.
    #              Re-hashes nothing. It validates the adoption evidence log, checks no newer mtime,
    #              and rewrites the audit verdict in place. This is not cryptographic proof.
    "mode": "lineage",
    "lineage": {
        # Dense cells whose PRIOR semantic spot check is accepted via lineage rather than re-encode.
        "cells": [
            "pastis/tessera/baseline", "pastis/olmoearth/baseline", "pastis/galileo/baseline",
            "pastis/agrifm/baseline", "pastis/presto/baseline", "pastis/raw/baseline",
        ],
        # Where that semantic evidence lives, and when it finished. The log is parsed and must show
        # a clean completed run that explicitly accepted every cell above; a cache byte with a newer
        # mtime additionally refuses the cell (corroborating, not proof).
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


def _dense_digest_order(feature_rels: list[str], label_rels: list[str]) -> list[str]:
    """THE frozen tile order for the dense aggregate digest: sorted features, then sorted labels.

    Deliberately NOT a single global sort of every path -- that would interleave each tile's
    ``.labels.npy`` with its ``.npy`` and produce a different digest. The six published PASTIS
    digests were computed with this order, so any fleet verifier must call this function.
    """
    return sorted(feature_rels) + sorted(label_rels)


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

    ordered = _dense_digest_order(feat_rels, lab_rels)  # FROZEN order: features, then labels
    with ThreadPoolExecutor(max_workers=int(CONFIG["workers"])) as pool:
        inspected = dict(pool.map(_inspect, ordered))

    # Aggregate: rel + NUL + that tile's content hash + LF, folded in the frozen order above
    # (never completion order), so the digest is independent of thread scheduling.
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


def _parse_evidence_log(path: Path, cells: list[str]) -> tuple[bool, list[str], dict[str, str]]:
    """Validate the adoption log actually supports lineage for EVERY configured cell.

    Returns ``(ok, problems, per_cell_line)``. The log must exist, must show a successful completed
    adoption run (a report with zero refusals), and must explicitly accept each configured cell.
    Anything absent, failed, truncated, or missing a cell refuses lineage outright.
    """
    problems: list[str] = []
    if not path.is_file():
        return False, [f"evidence log missing: {path}"], {}
    text = path.read_text(errors="replace")
    if "==== ADOPTION REPORT ====" not in text:
        problems.append("evidence log has no ADOPTION REPORT section (truncated or wrong log)")
    # "N accepted, M refused." -- any refusal, or a missing tally, invalidates the run.
    tally = re.search(r"(\d+)\s+accepted,\s+(\d+)\s+refused", text)
    if not tally:
        problems.append("evidence log has no 'N accepted, M refused' completion tally")
    elif int(tally.group(2)) != 0:
        problems.append(f"evidence log reports {tally.group(2)} refused cell(s) -- run was not clean")
    if "[job] complete" not in text:
        problems.append("evidence log has no '[job] complete' marker -- run did not finish")

    per_cell: dict[str, str] = {}
    for key in cells:
        # e.g. "  [adopted] pastis/raw/baseline: {...}"  or "[already-adopted] ..."
        match = re.search(rf"^\s*\[(adopted|already-adopted)\]\s+{re.escape(key)}\s*:(.*)$",
                          text, re.MULTILINE)
        if match:
            per_cell[key] = f"{match.group(1)}:{match.group(2).strip()}"
        else:
            problems.append(f"evidence log does not explicitly accept {key}")
    return (not problems), problems, per_cell


def _lineage() -> int:
    """Accept prior semantic evidence for already-audited dense cells, without re-encoding.

    IMPORTANT -- this is an OPERATIONAL argument, not a cryptographic proof. No adoption-time
    content digest exists for dense cells, so the current bytes CANNOT be shown mathematically
    identical to the bytes that passed the spot check. What is established instead:

      * the adoption log exists, completed cleanly, and explicitly accepted this cell;
      * dataset digest, checkpoint SHA-256, tile-set digest, counts, feature dim, shapes and dtypes
        all still match;
      * no file under the cache root carries an mtime newer than the adoption check.

    The mtime check is corroborating operational evidence only: copy tools (rsync -a, cp -p, tar -p)
    preserve timestamps, so an unchanged mtime does not by itself prove unchanged bytes. Acceptance
    is therefore recorded explicitly as an operational-lineage assumption.

    The aggregate content digest was computed by THIS audit and was never present in the adoption
    log. It is not retrospective evidence; it freezes the accepted identity going forward so any
    later drift is detectable.
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

    # GATE: the adoption log must exist, have completed cleanly, and explicitly accept every
    # configured cell. If it does not, NO cell gets lineage acceptance.
    evidence_path = C.REPO / cfg["evidence_log"]
    ok, log_problems, per_cell_evidence = _parse_evidence_log(evidence_path, list(cfg["cells"]))
    if not ok:
        for problem in log_problems:
            print(f"[lineage] EVIDENCE REJECTED: {problem}", flush=True)
        print(f"[lineage] refusing lineage for all {len(cfg['cells'])} cell(s)", flush=True)
        return 1
    print(f"[lineage] evidence log validated: {len(per_cell_evidence)}/{len(cfg['cells'])} cells explicitly accepted", flush=True)

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
            "accepted under an OPERATIONAL-LINEAGE assumption (not cryptographic proof): the dense "
            f"re-encode spot check in {cfg['evidence_log']} (completed {cfg['evidence_completed']}) "
            f"explicitly accepted this cell [{per_cell_evidence.get(key, 'n/a')}], and dataset digest, "
            "checkpoint SHA-256, tile-set digest, tile counts, feature dim, shapes and dtypes all "
            "still match. No adoption-time content digest exists for dense caches, so the current "
            "bytes CANNOT be shown mathematically identical to the bytes that passed that check."
        )
        cell["no_rewrite_since_evidence"] = {
            "newest_mtime": datetime.fromtimestamp(newest).isoformat(),
            "evidence_completed": cfg["evidence_completed"],
            "evidence_strength": "operational_only",
            "caveat": ("mtime is corroborating operational evidence, not proof: archive-mode copy "
                       "tools (rsync -a, cp -p, tar -p) preserve timestamps, so an unchanged mtime "
                       "does not by itself establish unchanged bytes."),
        }
        cell["digest_role"] = (
            "The aggregate content digest FREEZES this cell's accepted identity from this audit "
            "forward; it was computed by this audit and was NOT present in the adoption log. It is "
            "not retrospective evidence."
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
        "basis": ("OPERATIONAL LINEAGE, not cryptographic proof: validated adoption log explicitly "
                  "accepting each cell, plus unchanged identity fields, plus no newer mtime under the "
                  "cache root. mtime is corroborating only (archive-mode copies preserve timestamps). "
                  "The aggregate digest freezes identity going forward and is not retrospective "
                  "evidence."),
        "evidence_strength": "operational_only",
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
            # A required cell that is missing is only tolerable when this host is deliberately
            # auditing a subset; otherwise it is a hard failure, never a silent pass.
            out["status"] = "absent_here" if CONFIG["skip_absent"] else "absent_required"
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
    for status in ("valid", "missing_evidence", "invalid", "absent_required", "absent_here"):
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
    tolerated = {"valid"} | ({"absent_here"} if CONFIG["skip_absent"] else set())
    return 0 if all(v.get("status") in tolerated for v in results.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
