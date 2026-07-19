"""Validate and move legacy embeddings into the fixed canonical cache.

This temporary migration utility never regenerates embeddings and never copies their data. Edit the
CONFIG block and run it without command-line arguments. Report mode performs every validation but
does not write. Apply mode validates every candidate before moving any candidate, then atomically
renames each accepted cell on its filesystem and records it in cache.json last.

Each cell is independently resumable. If execution stops after a rename but before its cache.json
record is written, rerunning validates the canonical artifact and finishes the record. A later cell
failing does not roll back cells already completed successfully.
"""

from __future__ import annotations

import os
import sys
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import NamedTuple

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import numpy as np  # noqa: E402

from utils import artifacts  # noqa: E402
from utils import cacheutils as C  # noqa: E402

# ===================== CONFIG (edit me; no CLI) =============================
CONFIG = {
    "mode": "report",  # "report" (never writes) | "apply" (rename accepted artifacts)
    "spotcheck_k": 32,
    "tolerance": {"_default": 1e-4},
    "candidates": [
        # {"benchmark": "cropharvest", "model": "galileo", "s2_only": False,
        #  "dense": False, "legacy": "/scratch/.../fingerprint/baseline.npy",
        #  "weights_path": None},
        # {"benchmark": "pastis", "model": "galileo", "s2_only": False,
        #  "dense": True, "legacy": "/scratch/.../fingerprint/baseline",
        #  "weights_path": None},
    ],
}
# ===========================================================================


class _Plan(NamedTuple):
    report: dict
    apply: Callable[[], dict] | None = None


def _spotcheck_indices(n: int, k: int, labels=None) -> list[int]:
    """Return deterministic, well-distributed indices with at least one per class."""
    if n <= 0:
        return []
    if n <= k:
        return list(range(n))
    picks = {int(i) for i in np.linspace(0, n - 1, num=k, dtype=int).tolist()}
    if labels is not None:
        labels = np.asarray(labels)
        for value in np.unique(labels):
            picks.add(int(np.argmax(labels == value)))
    return sorted(picks)


def _tol(model: str) -> float:
    return CONFIG["tolerance"].get(model, CONFIG["tolerance"]["_default"])


def _subset_bench(bench, idx: list[int]):
    """Return a tabular benchmark restricted to the selected sample indices."""

    def sub_ms(ms):
        return replace(
            ms,
            values=[ms.values[i] for i in idx],
            months=[ms.months[i] for i in idx],
            doy=[ms.doy[i] for i in idx],
            years=[ms.years[i] for i in idx],
        )

    native = replace(
        bench.native,
        s2=sub_ms(bench.native.s2),
        s1=sub_ms(bench.native.s1),
        climate=sub_ms(bench.native.climate),
    )
    values = {
        "native": native,
        "labels": np.asarray(bench.labels)[idx],
        "groups": np.asarray(bench.groups)[idx],
        "latlon": np.asarray(bench.latlon)[idx],
    }
    if bench.years is not None:
        values["years"] = np.asarray(bench.years)[idx]
    if bench.sample_ids is not None:
        values["sample_ids"] = np.asarray(bench.sample_ids)[idx]
    return replace(bench, **values)


def _load_bench(benchmark: str, s2_only: bool):
    from evals import evals as EV

    bench_mod = EV.load_benchmark(benchmark)
    bench = C.cached_bench(bench_mod.BENCHMARK)
    return (bench.s2_only() if s2_only else bench), bench_mod.BENCHMARK


def _refused(report: dict, reason: str, spot_check=None) -> _Plan:
    result = {**report, "status": "refused", "reason": reason}
    if spot_check is not None:
        result["spot_check"] = spot_check
    return _Plan(result)


def _existing_ancestor(path: Path) -> Path:
    current = path
    while not current.exists():
        parent = current.parent
        if parent == current:
            raise FileNotFoundError(f"No existing ancestor for {path}")
        current = parent
    return current


def _device_id(path: Path) -> int:
    return path.stat().st_dev


def _cross_device_problem(source: Path, destination: Path) -> str | None:
    destination_fs = _existing_ancestor(destination.parent)
    if _device_id(source) == _device_id(destination_fs):
        return None
    return (
        f"cross-filesystem move refused: {source} and destination parent "
        f"{destination_fs} are on different devices"
    )


def _remove_empty_fingerprint_parent(source: Path) -> None:
    """Remove only the now-empty fingerprint directory that directly contained the source."""
    try:
        source.parent.rmdir()
    except OSError:
        pass


def _canonical_tabular_problem(bench, benchmark, model, artifact, weights) -> tuple[str, list[str]]:
    state, _arr, problems = C._tabular_state(bench, benchmark, model, artifact, weights)
    return state, problems


def _prepare_tabular(cand, enc_kwargs) -> _Plan:
    benchmark, model = cand["benchmark"], cand["model"]
    artifact = C.artifact_name(cand.get("s2_only", False))
    legacy = Path(cand["legacy"])
    weights = cand.get("weights_path")
    report = {"candidate": f"{benchmark}/{model}/{artifact}", "legacy": str(legacy)}
    bench, bench_key = _load_bench(benchmark, cand.get("s2_only", False))
    destination = C.embedding_cache_path(bench_key, model, artifact)

    state, canonical_problems = _canonical_tabular_problem(
        bench, bench_key, model, artifact, weights
    )
    if state == "mismatch":
        return _refused(report, "invalid canonical cache: " + "; ".join(canonical_problems))
    already_adopted = state == "ok"

    recovering = not already_adopted and destination.exists() and not legacy.exists()
    if not already_adopted and destination.exists() and legacy.exists() and destination != legacy:
        return _refused(report, f"canonical destination already exists without a cache.json record: {destination}")
    source = destination if already_adopted or recovering or destination == legacy else legacy
    if not source.is_file():
        return _refused(report, "legacy artifact missing")

    try:
        arr = np.load(source, mmap_mode="r", allow_pickle=False)
    except Exception as exc:  # noqa: BLE001
        return _refused(report, f"unreadable embedding array: {exc}")
    problems = []
    if arr.ndim != 2:
        problems.append(f"rank {arr.ndim} != 2")
    elif arr.shape[0] != bench.n_samples:
        problems.append(f"n_samples {arr.shape[0]} != {bench.n_samples}")
    if str(arr.dtype) != C.EMB_DTYPE:
        problems.append(f"dtype {arr.dtype} != {C.EMB_DTYPE}")
    if problems:
        return _refused(report, "; ".join(problems))

    model_obj = C.build_model(model, **enc_kwargs)
    idx = _spotcheck_indices(
        bench.n_samples, CONFIG["spotcheck_k"], labels=getattr(bench, "labels", None)
    )
    re_encoded = np.asarray(model_obj.encode(_subset_bench(bench, idx)), dtype=np.float32)
    if re_encoded.ndim == 2 and re_encoded.shape[0] == len(idx) and re_encoded.shape[1] != arr.shape[1]:
        return _refused(
            report,
            f"feature width {arr.shape[1]} != encoded {re_encoded.shape[1]}",
        )
    if re_encoded.ndim != 2 or re_encoded.shape != (len(idx), arr.shape[1]):
        return _refused(
            report,
            f"spot check shape {re_encoded.shape} != expected {(len(idx), arr.shape[1])}",
        )
    max_err = float(np.max(np.abs(re_encoded - arr[idx]))) if idx else 0.0
    tolerance = _tol(model)
    spot = {"k": len(idx), "max_abs_err": max_err, "tol": tolerance}
    if max_err > tolerance:
        return _refused(
            report,
            f"spot check max_abs_err {max_err:.2e} > tol {tolerance:.0e}",
            spot,
        )
    source_digest = artifacts.sha256_file(source)
    if already_adopted:
        record = C._cache_record(bench_key, model, artifact)
        if record.get("artifact_sha256") != source_digest:
            return _refused(report, "canonical array content does not match its cache.json record", spot)
        return _Plan({**report, "status": "already-adopted", "spot_check": spot})
    if not recovering:
        cross_device = _cross_device_problem(source, destination)
        if cross_device:
            return _refused(report, cross_device, spot)

    record = {
        "checkpoint_sha256": C.checkpoint_sha256(model, weights),
        "dataset_digest": C.dataset_digest(bench_key),
        "sample_ids_digest": C.sample_ids_digest(bench.sample_ids),
        "shape": [int(value) for value in arr.shape],
        "dtype": str(arr.dtype),
        "artifact_sha256": source_digest,
    }

    def apply() -> dict:
        destination.parent.mkdir(parents=True, exist_ok=True)
        with C._cache_lock(destination):
            current_state, current_problems = _canonical_tabular_problem(
                bench, bench_key, model, artifact, weights
            )
            if current_state == "ok":
                return {**report, "status": "already-adopted", "spot_check": spot}
            if current_state == "mismatch":
                raise RuntimeError(
                    f"canonical cache changed after validation: {'; '.join(current_problems)}"
                )
            if recovering:
                if not destination.is_file():
                    raise RuntimeError(f"canonical recovery artifact disappeared: {destination}")
            else:
                if destination.exists():
                    raise RuntimeError(f"canonical destination appeared after validation: {destination}")
                if not source.is_file():
                    raise RuntimeError(f"legacy source disappeared after validation: {source}")
                cross_device = _cross_device_problem(source, destination)
                if cross_device:
                    raise RuntimeError(cross_device)
                os.replace(source, destination)
            C.update_cache(embeddings={C._embedding_key(bench_key, model, artifact): record})
        if legacy != destination:
            _remove_empty_fingerprint_parent(legacy)
        return {**report, "status": "adopted", "spot_check": spot}

    return _Plan({**report, "status": "ready", "spot_check": spot}, apply)


def _prepare_dense(cand, enc_kwargs) -> _Plan:
    benchmark, model = cand["benchmark"], cand["model"]
    s2_only = cand.get("s2_only", False)
    artifact = C.artifact_name(s2_only)
    legacy = Path(cand["legacy"])
    weights = cand.get("weights_path")
    report = {"candidate": f"{benchmark}/{model}/{artifact}", "legacy": str(legacy)}
    bench, bench_key = _load_bench(benchmark, s2_only)
    destination = C.dense_embedding_cache_dir(bench_key, model, artifact)

    state, _root, canonical_problems = C._dense_state(
        bench, bench_key, model, artifact, weights
    )
    if state == "mismatch":
        return _refused(report, "invalid canonical cache: " + "; ".join(canonical_problems))
    already_adopted = state == "ok"

    recovering = not already_adopted and destination.exists() and not legacy.exists()
    if not already_adopted and destination.exists() and legacy.exists() and destination != legacy:
        return _refused(report, f"canonical destination already exists without a cache.json record: {destination}")
    source = destination if already_adopted or recovering or destination == legacy else legacy
    if not source.is_dir():
        return _refused(report, "legacy dense cache directory missing")

    C.dataset_digest(bench_key)
    feature_rels, label_rels = C._dense_expected_rels(bench)
    problems = C._dense_tile_set_problems(source, feature_rels, label_rels)
    feature_dim = None
    for rel in feature_rels:
        path = source / rel
        if not path.exists():
            continue
        try:
            shape, _dtype = C._npy_shape_dtype(path)
        except Exception:  # noqa: BLE001
            continue
        if len(shape) == 2:
            feature_dim = int(shape[1])
            break

    for feature_rel in feature_rels:
        feature_path = source / feature_rel
        label_rel = feature_rel.removesuffix(".npy") + ".labels.npy"
        label_path = source / label_rel
        if not feature_path.exists() or not label_path.exists():
            continue
        try:
            feature_shape, feature_dtype = C._npy_shape_dtype(feature_path)
        except Exception as exc:  # noqa: BLE001
            problems.append(f"unreadable feature tile {feature_rel}: {exc}")
            continue
        if feature_dtype != C.EMB_DTYPE:
            problems.append(f"{feature_rel}: dtype {feature_dtype} != {C.EMB_DTYPE}")
        if len(feature_shape) != 2:
            problems.append(f"{feature_rel}: rank {len(feature_shape)} != 2")
        elif feature_dim is not None and feature_shape[1] != feature_dim:
            problems.append(
                f"{feature_rel}: feature width {feature_shape[1]} != {feature_dim} "
                "(inconsistent across tiles)"
            )
        try:
            label_shape, label_dtype = C._npy_shape_dtype(label_path)
        except Exception as exc:  # noqa: BLE001
            problems.append(f"unreadable label tile {label_rel}: {exc}")
            continue
        if label_dtype != C.DENSE_LABEL_DTYPE:
            problems.append(f"{label_rel}: dtype {label_dtype} != {C.DENSE_LABEL_DTYPE}")
        if len(label_shape) != 1:
            problems.append(f"{label_rel}: rank {len(label_shape)} != 1")
        elif len(feature_shape) == 2 and label_shape[0] != feature_shape[0]:
            problems.append(
                f"{label_rel}: {label_shape[0]} rows != feature rows {feature_shape[0]}"
            )
    if problems:
        shown = "; ".join(problems[:6])
        if len(problems) > 6:
            shown += f"; (+{len(problems) - 6} more)"
        return _refused(report, shown)

    model_obj = C.build_model(model, **enc_kwargs)
    picks = {
        feature_rels[index]
        for index in _spotcheck_indices(len(feature_rels), CONFIG["spotcheck_k"])
    }
    max_err, checked = 0.0, 0
    for tile_id, fold, tile, _labels in bench.iter_tiles():
        rel = f"fold_{fold}/{tile_id}.npy"
        if rel not in picks:
            continue
        if hasattr(model_obj, "encode_dense"):
            features = model_obj.encode_dense(tile)
        else:
            features = model_obj.encode(tile.pixel_benchmark())
        features = np.asarray(features, dtype=np.float32)
        legacy_features = np.load(source / rel, allow_pickle=False).astype(np.float32)
        if features.shape != legacy_features.shape:
            return _refused(
                report,
                f"spot check tile {rel}: shape {features.shape} != legacy {legacy_features.shape}",
            )
        if features.size:
            max_err = max(max_err, float(np.max(np.abs(features - legacy_features))))
        checked += 1
    tolerance = _tol(model)
    spot = {"tiles": checked, "max_abs_err": max_err, "tol": tolerance}
    if checked == 0:
        return _refused(
            report, "spot check re-encoded no tiles (descriptor/legacy tile-id mismatch)"
        )
    if max_err > tolerance:
        return _refused(
            report,
            f"spot check max_abs_err {max_err:.2e} > tol {tolerance:.0e}",
            spot,
        )
    if already_adopted:
        return _Plan({**report, "status": "already-adopted", "spot_check": spot})
    if not recovering:
        cross_device = _cross_device_problem(source, destination)
        if cross_device:
            return _refused(report, cross_device, spot)

    record = C._dense_record(bench, bench_key, model, source, weights)

    def apply() -> dict:
        destination.parent.mkdir(parents=True, exist_ok=True)
        with C._cache_lock(destination):
            current_state, _current_root, current_problems = C._dense_state(
                bench, bench_key, model, artifact, weights
            )
            if current_state == "ok":
                return {**report, "status": "already-adopted", "spot_check": spot}
            if current_state == "mismatch":
                raise RuntimeError(
                    f"canonical cache changed after validation: {'; '.join(current_problems)}"
                )
            if recovering:
                if not destination.is_dir():
                    raise RuntimeError(f"canonical recovery directory disappeared: {destination}")
            else:
                if destination.exists():
                    raise RuntimeError(f"canonical destination appeared after validation: {destination}")
                if not source.is_dir():
                    raise RuntimeError(f"legacy source disappeared after validation: {source}")
                cross_device = _cross_device_problem(source, destination)
                if cross_device:
                    raise RuntimeError(cross_device)
                os.replace(source, destination)
            final_problems = C._dense_tile_set_problems(
                destination, feature_rels, label_rels
            )
            if final_problems:
                raise RuntimeError("canonical completeness failed: " + "; ".join(final_problems))
            C.update_cache(embeddings={C._embedding_key(bench_key, model, artifact): record})
        if legacy != destination:
            _remove_empty_fingerprint_parent(legacy)
        return {**report, "status": "adopted", "spot_check": spot}

    return _Plan({**report, "status": "ready", "spot_check": spot}, apply)


def _prepare_candidate(cand, enc_kwargs) -> _Plan:
    prepare = _prepare_dense if cand.get("dense") else _prepare_tabular
    try:
        return prepare(cand, enc_kwargs)
    except Exception as exc:  # noqa: BLE001
        return _Plan(
            {
                "candidate": f"{cand['benchmark']}/{cand['model']}",
                "status": "error",
                "reason": repr(exc),
            }
        )


def _adopt_tabular(cand, enc_kwargs) -> dict:
    plan = _prepare_tabular(cand, enc_kwargs)
    if CONFIG["mode"] == "apply" and plan.apply is not None:
        return plan.apply()
    return plan.report


def _adopt_dense(cand, enc_kwargs) -> dict:
    plan = _prepare_dense(cand, enc_kwargs)
    if CONFIG["mode"] == "apply" and plan.apply is not None:
        return plan.apply()
    return plan.report


def _summarize(reports: list[dict]) -> list[dict]:
    print("\n==== ADOPTION REPORT ====")
    for report in reports:
        detail = report.get("reason", report.get("spot_check", ""))
        print(f"  [{report['status']}] {report.get('candidate')}: {detail}")
    failures = [report for report in reports if report["status"] in {"refused", "error"}]
    print(f"\n{len(reports) - len(failures)} accepted, {len(failures)} refused.")
    return failures


def main() -> int:
    from utils import gputils

    if CONFIG["mode"] not in {"report", "apply"}:
        raise ValueError("CONFIG['mode'] must be 'report' or 'apply'")
    enc_kwargs = {"device": gputils.device()}
    candidates = CONFIG["candidates"]
    print(f"[adopt] mode={CONFIG['mode']}  candidates={len(candidates)}")

    plans = [_prepare_candidate(candidate, enc_kwargs) for candidate in candidates]
    failures = _summarize([plan.report for plan in plans])
    if failures or CONFIG["mode"] == "report":
        return 1 if failures else 0

    results = []
    for plan in plans:
        if plan.apply is None:
            results.append(plan.report)
            continue
        try:
            results.append(plan.apply())
        except Exception as exc:  # noqa: BLE001
            results.append({**plan.report, "status": "error", "reason": repr(exc)})
    return 1 if _summarize(results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
