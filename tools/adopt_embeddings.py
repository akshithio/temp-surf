"""One-time adoption of already-computed embeddings into the frozen canonical cache.

TEMPORARY UTILITY -- part of the freeze-and-run migration, removed before the repository's
final-release cleanup. No command-line arguments: edit the CONFIG block below and run it. Report
mode NEVER writes; publish mode writes only accepted artifacts to the fixed canonical paths and
NEVER deletes a legacy artifact. On Gilbreth the spot checks re-encode samples, so publish runs as
a GPU Slurm job.

Each CONFIG["candidates"] entry declares a legacy leaf's intended identity; the tool VALIDATES that
claim before adopting:
  * tabular: count / dtype / feature width / array integrity, then a deterministic K-row re-encode
    spot check within a model-appropriate tolerance;
  * dense: exact expected tile-set completeness, then a deterministic K-tile pixel re-encode.
Anything that fails a check, cannot be reconstructed, or is ambiguous is REFUSED (regenerate it
fresh under the frozen commit instead).
"""

from __future__ import annotations

import os
import shutil
import sys
from dataclasses import replace
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import numpy as np  # noqa: E402

from utils import artifacts  # noqa: E402
from utils import cacheutils as C  # noqa: E402


def _spotcheck_indices(n: int, k: int, labels=None) -> list[int]:
    """Deterministic, well-distributed indices: boundaries + even spacing + one per class."""
    if n <= 0:
        return []
    if n <= k:
        return list(range(n))
    picks = {int(i) for i in np.linspace(0, n - 1, num=k, dtype=int).tolist()}
    if labels is not None:
        labels = np.asarray(labels)
        for c in np.unique(labels):
            picks.add(int(np.argmax(labels == c)))
    return sorted(picks)

# ===================== CONFIG (edit me; no CLI) =============================
CONFIG = {
    "mode": "report",          # "report" (never writes) | "publish" (writes accepted artifacts only)
    "spotcheck_k": 32,
    # Per-model absolute tolerance for the re-encode spot check (float32 encoders; justify per model).
    "tolerance": {"_default": 1e-4},
    # Explicit legacy leaves to adopt. `legacy` points at the old fingerprint dir's baseline.npy
    # (tabular) or the old baseline/ tile dir (dense). s2_only selects the artifact name.
    "candidates": [
        # {"benchmark": "cropharvest", "model": "galileo", "s2_only": False, "dense": False,
        #  "legacy": "/scratch/.../embeddings/cropharvest/galileo/n67692_..._e2b3b3b0f91_.../baseline.npy",
        #  "weights_path": None},
        # {"benchmark": "pastis", "model": "galileo", "s2_only": False, "dense": True,
        #  "legacy": "/scratch/.../embeddings/pastis/galileo/n..._.../baseline", "weights_path": None},
    ],
}
# ===========================================================================


def _tol(model: str) -> float:
    return CONFIG["tolerance"].get(model, CONFIG["tolerance"]["_default"])


def _identical(a: Path, b: Path) -> bool:
    """Byte-identity of two files via full content SHA-256 (reuses artifacts.sha256_file)."""
    return artifacts.sha256_file(a) == artifacts.sha256_file(b)


def _subset_bench(bench, idx: list[int]):
    """A copy of a tabular benchmark restricted (in order) to the given sample indices -- so the
    spot check re-encodes only K rows rather than the whole matrix."""
    def sub_ms(ms):
        return replace(ms, values=[ms.values[i] for i in idx], months=[ms.months[i] for i in idx],
                       doy=[ms.doy[i] for i in idx], years=[ms.years[i] for i in idx])
    native = replace(bench.native, s2=sub_ms(bench.native.s2), s1=sub_ms(bench.native.s1), climate=sub_ms(bench.native.climate))
    kw = dict(native=native, labels=np.asarray(bench.labels)[idx], groups=np.asarray(bench.groups)[idx],
              latlon=np.asarray(bench.latlon)[idx])
    if bench.years is not None:
        kw["years"] = np.asarray(bench.years)[idx]
    if bench.sample_ids is not None:
        kw["sample_ids"] = np.asarray(bench.sample_ids)[idx]
    return replace(bench, **kw)


def _load_bench(benchmark: str, s2_only: bool):
    from evals import evals as EV
    bench_mod = EV.load_benchmark(benchmark)
    bench = C.cached_bench(bench_mod.BENCHMARK)
    return (bench.s2_only() if s2_only else bench), bench_mod.BENCHMARK


def _adopt_tabular(cand, enc_kwargs) -> dict:
    benchmark, model = cand["benchmark"], cand["model"]
    artifact = C.artifact_name(cand.get("s2_only", False))
    legacy = Path(cand["legacy"])
    weights = cand.get("weights_path")
    report = {"candidate": f"{benchmark}/{model}/{artifact}", "legacy": str(legacy)}

    bench, bench_key = _load_bench(benchmark, cand.get("s2_only", False))
    if not legacy.exists():
        return {**report, "status": "refused", "reason": "legacy artifact missing"}
    try:
        arr = np.load(legacy)
    except Exception as exc:  # noqa: BLE001 -- a malformed legacy array is a refusal, not a crash
        return {**report, "status": "refused", "reason": f"unreadable legacy array: {exc}"}
    problems = []
    if arr.shape[0] != bench.n_samples:
        problems.append(f"n_samples {arr.shape[0]} != {bench.n_samples}")
    if str(arr.dtype) != C.EMB_DTYPE:
        problems.append(f"dtype {arr.dtype} != {C.EMB_DTYPE}")
    model_obj = C.build_model(model, **enc_kwargs)
    if arr.shape[1] != model_obj.embedding_dim:
        problems.append(f"feature width {arr.shape[1]} != {model_obj.embedding_dim}")
    if problems:
        return {**report, "status": "refused", "reason": "; ".join(problems)}

    idx = _spotcheck_indices(bench.n_samples, CONFIG["spotcheck_k"], labels=getattr(bench, "labels", None))
    re_enc = model_obj.encode(_subset_bench(bench, idx)).astype(np.float32)
    max_err = float(np.max(np.abs(re_enc - arr[idx]))) if len(idx) else 0.0
    tol = _tol(model)
    spot = {"k": len(idx), "max_abs_err": max_err, "tol": tol}
    if max_err > tol:
        return {**report, "status": "refused", "reason": f"spot check max_abs_err {max_err:.2e} > tol {tol:.0e}", "spot_check": spot}

    art_path = C.embedding_cache_path(bench_key, model, artifact)
    man_path = C.embedding_manifest_path(bench_key, model, artifact)
    if man_path.exists():
        return {**report, "status": "refused", "reason": "canonical manifest already present (delete it to re-adopt)", "spot_check": spot}
    if CONFIG["mode"] != "publish":
        return {**report, "status": "would-adopt", "spot_check": spot}

    # Deliberate operator action: write the canonical baseline.npy atomically, then the sidecar LAST.
    art_path.parent.mkdir(parents=True, exist_ok=True)
    with C._cache_lock(art_path):
        tmp = C._atomic_tmp(art_path)
        try:
            with open(tmp, "wb") as f:
                np.save(f, np.ascontiguousarray(arr))
            os.replace(tmp, art_path)
        finally:
            tmp.unlink(missing_ok=True)
        C._write_manifest(man_path, {
            "schema": 1, "benchmark": bench_key, "model": model, "artifact": artifact,
            "checkpoint_sha256": C.checkpoint_sha256(model, weights), "dataset_digest": C.dataset_digest(bench_key),
            "sample_ids_digest": C.sample_ids_digest(bench.sample_ids),
            "shape": [int(x) for x in arr.shape], "dtype": str(arr.dtype),
            "artifact_sha256": artifacts.sha256_file(art_path),
            "adoption": {"source_leaf": str(legacy), "method": "spot-check-migrated",
                         "spot_check": {"k": len(idx), "indices": idx, "max_abs_err": max_err, "tol": tol}},
        })
    return {**report, "status": "adopted", "spot_check": spot}


def _adopt_dense(cand, enc_kwargs) -> dict:
    """Adopt a legacy PASTIS dense tile cache WITHOUT regenerating it.

    Validates the legacy tile dir against the descriptor's EXACT expected (feature, label) tile set
    -- completeness (missing / unexpected), dtype, feature width, and cheap identity (checkpoint,
    dataset digest, artifact selection) -- then re-encodes K expected tiles' pixels as a correctness
    spot check. On publish it copies only accepted tiles into the fixed canonical ``baseline/`` dir
    with per-file atomic writes under the existing writer locks, runs the exact missing/extra check,
    and writes ``baseline.manifest.json`` LAST. The legacy dir is never touched.
    """
    benchmark, model = cand["benchmark"], cand["model"]
    s2_only = cand.get("s2_only", False)
    artifact = C.artifact_name(s2_only)
    legacy = Path(cand["legacy"])
    weights = cand.get("weights_path")
    report = {"candidate": f"{benchmark}/{model}/{artifact}", "legacy": str(legacy)}

    bench, bench_key = _load_bench(benchmark, s2_only)
    if not legacy.is_dir():
        return {**report, "status": "refused", "reason": "legacy dense cache dir missing"}

    # The dataset digest must exist (preflight established it); hard-fail early with a clear message.
    # Checkpoint identity is validated by the K-tile re-encode below and recorded in the manifest.
    C.dataset_digest(bench_key)

    feat_rels, lab_rels = C._dense_expected_rels(bench)
    problems = C._dense_tile_set_problems(legacy, feat_rels, lab_rels)  # missing + extra feat/label

    model_obj = C.build_model(model, **enc_kwargs)
    # One-time adoption validates EVERY expected tile's header (not just the first): a malformed,
    # wrong-dtype, wrong-rank, or wrong-width feature tile, or a mis-shaped/mis-typed label tile, is a
    # refusal that names the offending tile. Tiles already flagged missing are skipped here.
    for frel in feat_rels:
        fpath = legacy / frel
        lrel = frel.removesuffix(".npy") + ".labels.npy"
        lpath = legacy / lrel
        if not fpath.exists() or not lpath.exists():
            continue  # already reported by the completeness check
        try:
            fshape, fdtype = C._npy_shape_dtype(fpath)
        except Exception as exc:  # noqa: BLE001 -- a malformed feature tile is a refusal, not a crash
            problems.append(f"unreadable feature tile {frel}: {exc}")
            continue
        if fdtype != C.EMB_DTYPE:
            problems.append(f"{frel}: dtype {fdtype} != {C.EMB_DTYPE}")
        if len(fshape) != 2:
            problems.append(f"{frel}: rank {len(fshape)} != 2")
        elif fshape[1] != model_obj.embedding_dim:
            problems.append(f"{frel}: feature width {fshape[1]} != {model_obj.embedding_dim}")
        try:
            lshape, ldtype = C._npy_shape_dtype(lpath)
        except Exception as exc:  # noqa: BLE001 -- a malformed label tile is a refusal, not a crash
            problems.append(f"unreadable label tile {lrel}: {exc}")
            continue
        if ldtype != C.DENSE_LABEL_DTYPE:
            problems.append(f"{lrel}: dtype {ldtype} != {C.DENSE_LABEL_DTYPE}")
        if len(lshape) != 1:
            problems.append(f"{lrel}: rank {len(lshape)} != 1")
        elif len(fshape) == 2 and lshape[0] != fshape[0]:
            problems.append(f"{lrel}: {lshape[0]} rows != feature rows {fshape[0]}")
    if problems:
        shown = "; ".join(problems[:6]) + (f"; (+{len(problems) - 6} more)" if len(problems) > 6 else "")
        return {**report, "status": "refused", "reason": shown}

    # Deterministic K-tile pixel re-encode: recompute the selected tiles and compare to the legacy
    # feature tiles within tolerance. A tile-id set mismatch (checked==0) is itself a refusal.
    picks = {feat_rels[i] for i in _spotcheck_indices(len(feat_rels), CONFIG["spotcheck_k"])}
    max_err, checked = 0.0, 0
    for tile_id, fold, tile, _labels in bench.iter_tiles():
        rel = f"fold_{fold}/{tile_id}.npy"
        if rel not in picks:
            continue
        features = model_obj.encode_dense(tile) if hasattr(model_obj, "encode_dense") else model_obj.encode(tile.pixel_benchmark())
        features = np.asarray(features, dtype=np.float32)
        legacy_feat = np.load(legacy / rel).astype(np.float32)
        if features.shape != legacy_feat.shape:
            return {**report, "status": "refused",
                    "reason": f"spot check tile {rel}: shape {features.shape} != legacy {legacy_feat.shape}"}
        max_err = max(max_err, float(np.max(np.abs(features - legacy_feat))) if features.size else 0.0)
        checked += 1
    tol = _tol(model)
    spot = {"tiles": checked, "max_abs_err": max_err, "tol": tol}
    if checked == 0:
        return {**report, "status": "refused", "reason": "spot check re-encoded no tiles (descriptor/legacy tile-id mismatch)"}
    if max_err > tol:
        return {**report, "status": "refused", "reason": f"spot check max_abs_err {max_err:.2e} > tol {tol:.0e}", "spot_check": spot}

    root = C.dense_embedding_cache_dir(bench_key, model, artifact)
    man_path = C.dense_manifest_path(bench_key, model, artifact)
    if man_path.exists():
        return {**report, "status": "refused", "reason": "canonical manifest already present (delete it to re-adopt)", "spot_check": spot}
    if CONFIG["mode"] != "publish":
        return {**report, "status": "would-adopt", "spot_check": spot}

    # Per-file atomic publication of EXACTLY the expected tiles. Never trust an existing destination
    # just because it exists: keep it only when it is byte-identical to the selected legacy source,
    # otherwise atomically REPLACE it -- a foreign or partial tile from another candidate cannot be
    # certified. This is the safe-resume contract (an identical tile is a no-op).
    root.mkdir(parents=True, exist_ok=True)
    for rel in feat_rels + lab_rels:
        dst, src = root / rel, legacy / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        with C._cache_lock(dst):
            if dst.exists() and _identical(src, dst):
                continue  # already published identically -> resume without rewriting
            tmp = C._atomic_tmp(dst)
            try:
                shutil.copyfile(src, tmp)
                os.replace(tmp, dst)  # atomic create-or-replace of a differing/foreign tile
            finally:
                tmp.unlink(missing_ok=True)

    # Exact missing / unexpected-feature / unexpected-label check on the canonical dir, THEN the
    # manifest LAST (shared helper -- identical rule to the runtime load path).
    with C._cache_lock(man_path):
        pub_problems = C._dense_tile_set_problems(root, feat_rels, lab_rels)
        if pub_problems:
            return {**report, "status": "refused",
                    "reason": "post-copy completeness failed: " + "; ".join(pub_problems), "spot_check": spot}
        C._write_manifest(man_path, {
            **C._build_dense_manifest(bench, bench_key, model, artifact, root, weights),
            "adoption": {"source_leaf": str(legacy), "method": "spot-check-migrated",
                         "spot_check": {**spot, "picks": sorted(picks)}},
        })
    return {**report, "status": "adopted", "spot_check": spot}


def main() -> int:
    from utils import gputils
    enc_kwargs = {"device": gputils.device()}
    print(f"[adopt] mode={CONFIG['mode']}  candidates={len(CONFIG['candidates'])}")
    reports = []
    for cand in CONFIG["candidates"]:
        adopt = _adopt_dense if cand.get("dense") else _adopt_tabular
        try:
            reports.append(adopt(cand, enc_kwargs))
        except Exception as exc:  # noqa: BLE001 -- report per-candidate, never abort the whole run
            reports.append({"candidate": f"{cand['benchmark']}/{cand['model']}", "status": "error", "reason": repr(exc)})

    print("\n==== ADOPTION REPORT ====")
    for r in reports:
        print(f"  [{r['status']}] {r.get('candidate')}: {r.get('reason', r.get('spot_check', ''))}")
    refused = [r for r in reports if r["status"] in ("refused", "error")]
    print(f"\n{len(reports) - len(refused)} adoptable, {len(refused)} refused. Legacy artifacts left untouched.")
    return 1 if (refused and CONFIG["mode"] == "publish") else 0


if __name__ == "__main__":
    raise SystemExit(main())
