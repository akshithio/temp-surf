from __future__ import annotations

import hashlib
import json
import os
import threading
from pathlib import Path
from typing import Any

import numpy as np
from joblib import Parallel, delayed

from utils import artifacts, cacheutils
from utils import ioutils as IOU
from utils import perfutils as perf

VALID_RUN_STAGES = {"gen_embeddings", "probing"}


def validate_run_stages(run_stages: list[str]) -> set[str]:
    stages = set(run_stages)
    unknown = stages - VALID_RUN_STAGES
    if unknown:
        raise ValueError(f"Unknown RUN_STAGES entries: {sorted(unknown)}. Valid entries: {sorted(VALID_RUN_STAGES)}")
    if not stages:
        raise ValueError("RUN_STAGES must include at least one stage.")
    return stages


def build_run_manifest(
    model_name: str,
    benchmark: str,
    artifact: str,
    embedding_digest: str | None,
    split_regimes,
    seeds,
    enc_kwargs,
    *,
    active_probes,
    budget_regimes,
    max_dense_pixels,
    write_predictions: bool = True,
) -> dict:
    """A readable, exact-match final-run manifest. Records every result-affecting knob plus the
    final commit, uv.lock digest, numerical-core deps, and the embedding's recorded content digest
    (artifact SHA for tabular, tile-set digest for dense). Splits are consumed from the frozen
    data/splits/ CSVs (discovered + checksum-verified via data/logs/splits.json), not bound here."""
    from evals import probes as _probes
    from evals import split_artifacts as _SA

    fi = cacheutils.frozen_run_identity()
    env = artifacts.capture_environment()
    enc = {k: v for k, v in sorted(enc_kwargs.items()) if k != "device"}
    manifest = {
        "schema": artifacts.SCHEMA_VERSION,
        "final_commit": fi["final_commit"],
        "clean_tree": fi["clean"],
        "tree_identity": fi["tree_identity"],
        "uv_lock_digest": artifacts.sha256_file(cacheutils.REPO / "uv.lock"),
        "deps": env.get("numerical_core", {}),
        "python": env.get("python"),
        "benchmark": benchmark,
        "model": model_name,
        "embedding": {"artifact": artifact, "digest": embedding_digest},
        "seeds": sorted(int(s) for s in seeds),
        "regimes": sorted(split_regimes),
        "budgets": list(budget_regimes),
        "probes": list(active_probes),
        "probe_cap": perf.PROBE_CAP or 0,          # the committed PROBE_CAP main.py pushed into perfutils
        "probe_tuning": bool(_probes.PROBE_TUNING),  # the committed PROBE_TUNING main.py pushed into probes
        "max_dense_pixels": max_dense_pixels,
        # Readable label-access contract: enabled ONLY when geographic_ood is requested; the unit is the
        # benchmark's (patches for dense PASTIS, else samples). The canonical counts/routes/splits/unit are
        # still recorded when disabled so the manifest stays self-describing.
        "label_access": _SA.label_access_contract(
            enabled=_SA.LABEL_ACCESS_REGIME in set(split_regimes), benchmark=benchmark
        ),
        # NEVER claim predictions were enabled when none will be written. Tabular writes per-sample
        # predictions for every regime; dense (PASTIS) writes them ONLY for the geographic_ood label-access
        # suite -- so a dense run without geographic_ood records write_predictions=False.
        "write_predictions": bool(write_predictions) and (
            _SA.LABEL_ACCESS_REGIME in set(split_regimes)
            if benchmark in _SA.DENSE_LABEL_ACCESS_BENCHMARKS else True
        ),
        "enc": enc,
    }
    return manifest


def run_manifest_digest(manifest: dict) -> str:
    """Canonical SHA-256 of a run manifest -- the stable id recorded in the completion marker."""
    return hashlib.sha256(json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _manifest_diffs(old: dict, new: dict) -> list[str]:
    """Exact field-by-field differences, reported as ``key: old != new`` (recurses one level)."""
    diffs: list[str] = []
    for key in sorted(set(old) | set(new)):
        a, b = old.get(key), new.get(key)
        if a == b:
            continue
        if isinstance(a, dict) and isinstance(b, dict):
            for sub in sorted(set(a) | set(b)):
                if a.get(sub) != b.get(sub):
                    diffs.append(f"{key}.{sub}: {a.get(sub)!r} != {b.get(sub)!r}")
        else:
            diffs.append(f"{key}: {a!r} != {b!r}")
    return diffs


def check_run_manifest(results_dir, manifest: dict, *, overwrite_mode: bool) -> None:
    """Resume only under an EXACT final-run manifest match; otherwise refuse with the exact diffs."""
    if overwrite_mode:
        return
    results_dir = Path(results_dir)
    man_path = results_dir / artifacts.RUN_MANIFEST_FILE
    rows_path = results_dir / "probe_results.jsonl"
    if man_path.exists():
        try:
            existing = json.loads(man_path.read_text())
        except (OSError, ValueError) as exc:
            raise RuntimeError(f"Refusing to resume {results_dir}: {artifacts.RUN_MANIFEST_FILE} is unreadable ({exc}).") from exc
        diffs = _manifest_diffs(existing, manifest)
        if diffs:
            raise RuntimeError(
                f"Refusing to resume {results_dir}: final-run manifest differs from this run:\n  - "
                + "\n  - ".join(diffs)
                + "\nSet OVERWRITE_MODE=True or use a fresh results dir."
            )
    elif rows_path.exists() and rows_path.stat().st_size > 0:
        raise RuntimeError(
            f"Refusing to resume {results_dir}: it has results but NO {artifacts.RUN_MANIFEST_FILE} "
            "(a pre-guard or foreign run). Verify they match this config, or use OVERWRITE_MODE=True."
        )


def publish_run_manifest(results_dir, manifest: dict) -> None:
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    IOU.write_json(results_dir / artifacts.RUN_MANIFEST_FILE, manifest)


def budget_row_key(row):
    return (
        row.get("seed"),
        row.get("split_regime"),
        row.get("holdout"),
        row.get("method"),
        row.get("probe_family"),
        row.get("budget_type"),
        row.get("label_budget"),
        row.get("evaluation_split"),
        row.get("label_access_route", ""),
    )


def prune_partial_budgets(rows, rows_path, preds_path, rerun_keys):
    if not rerun_keys:
        return rows
    kept = [r for r in rows if budget_row_key(r) not in rerun_keys]
    if len(kept) != len(rows):
        tmp_rows = cacheutils._atomic_tmp(rows_path)
        tmp_rows.unlink(missing_ok=True)
        if kept:
            IOU.append_jsonl(tmp_rows, kept)
        else:
            tmp_rows.touch()
        os.replace(tmp_rows, rows_path)
    if preds_path is not None:
        # Stream-filter (bounded memory): multiclass predictions.jsonl can be tens of GB, so we
        # must never materialize the whole file here (would OOM on crash-resume).
        IOU.rewrite_jsonl_dropping(preds_path, lambda p: budget_row_key(p) in rerun_keys)
    return kept


def _probe_cell(
    probe_fn,
    emb,
    train,
    val,
    test,
    y,
    groups,
    meta,
    seed,
    family="logistic",
    budgets=None,
    source_val=None,
    source_test=None,
    write_predictions: bool = True,
) -> tuple[list[dict], list[dict]]:
    x_tr, x_cond_te = emb[train], emb[test]
    y_tr, y_te, g_tr = y[train], y[test], groups[train]
    x_val = emb[val] if len(val) else None
    y_val = y[val] if len(val) else None
    extra_evals = {}
    for label, idx in (("source_validation", source_val), ("source_test", source_test)):
        if idx is not None and len(idx):
            extra_evals[label] = (emb[idx], y[idx], np.asarray(idx), np.asarray(groups)[idx])
    mname = meta.get("method", "?")
    identity = {k: meta[k] for k in ("seed", "holdout", "method") if k in meta}
    rows: list[dict] = []
    perf.set_identity(identity)
    with perf.measure(
        f"probe.run/{meta.get('benchmark', '?')}/{mname}",
        n_samples_train=len(train),
        n_samples_test=len(test),
        n_features=x_tr.shape[1],
    ):
        preds: list[dict] = []
        probe_fn(
            rows,
            x_tr,
            x_cond_te,
            y_tr,
            y_te,
            seed,
            meta=meta,
            groups_train=g_tr,
            predictions=preds if write_predictions else None,
            sample_ids_test=np.asarray(test),
            groups_test=np.asarray(groups)[test],
            x_val=x_val,
            y_val=y_val,
            extra_evals=extra_evals or None,
            family=family,
            **({} if budgets is None else {"budgets": budgets}),
        )
    perf.set_identity(None)
    return rows, preds if write_predictions else []


def _probe_cell_target(
    probe_fn,
    emb,
    train,
    val,
    target_pool,
    target_test,
    y,
    groups,
    meta,
    seed,
    family="logistic",
    budgets=None,
    write_predictions: bool = True,
) -> tuple[list[dict], list[dict]]:
    # schema v2: few-shot label budgets draw ONLY from the frozen target_label_pool and every budget
    # is scored on the frozen target_test. The whole target region (pool ++ test) is assembled for the
    # budget-0 "full" deployment anchor; pool_idx/test_idx mark the fixed split so labels are never
    # drawn from target_test.
    target_pool = np.asarray(target_pool, dtype=np.int64)
    target_test = np.asarray(target_test, dtype=np.int64)
    full = np.concatenate([target_pool, target_test])
    x_source_tr = emb[train]
    x_target_full, y_target_full = emb[full], y[full]
    g_source_tr = groups[train]
    x_val = emb[val] if len(val) else None
    y_val = y[val] if len(val) else None
    pool_idx = np.arange(len(target_pool))
    test_idx = np.arange(len(target_pool), len(full))
    mname = meta.get("method", "?")
    identity = {k: meta[k] for k in ("seed", "holdout", "method") if k in meta}
    rows: list[dict] = []
    perf.set_identity(identity)
    with perf.measure(
        f"probe.target/{meta.get('benchmark', '?')}/{mname}",
        n_samples_source=len(train),
        n_samples_target=len(full),
        n_features=x_source_tr.shape[1],
    ):
        preds: list[dict] = []
        probe_fn(
            rows,
            x_source_tr,
            x_target_full,
            y[train],
            y_target_full,
            seed,
            meta=meta,
            groups_source=g_source_tr,
            predictions=preds if write_predictions else None,
            sample_ids_target=full,
            groups_target=np.asarray(groups)[full],
            x_val=x_val,
            y_val=y_val,
            family=family,
            pool_idx=pool_idx,
            test_idx=test_idx,
            **({} if budgets is None else {"budgets": budgets}),
        )
    perf.set_identity(None)
    return rows, preds if write_predictions else []


def _probe_cell_label_access(
    probe_fn,
    emb,
    train,
    val,
    target_pool,
    target_test,
    matched_source_ranked_idx,
    fixed_source_removal_ranked_idx,
    target_ranked_idx,
    y,
    groups,
    meta,
    seed,
    family="logistic",
    write_predictions: bool = True,
) -> tuple[list[dict], list[dict]]:
    """One geographic_ood label-access cell: all 13 routes + the source_only complete-target diagnostic.
    Maps the frozen label-access order (CURRENT row indices, already validated against the split at load)
    to EXACT positions within the source-train / target-pool arrays, hard-failing if any index is not in
    its partition. Runs the whole suite in ONE call so source_only is fit once and its scorer reused."""
    train = np.asarray(train, dtype=np.int64)
    target_pool = np.asarray(target_pool, dtype=np.int64)
    target_test = np.asarray(target_test, dtype=np.int64)

    def _to_positions(current_idx, partition, name):
        pos = {int(v): p for p, v in enumerate(partition.tolist())}
        out = np.empty(len(current_idx), dtype=np.int64)
        for i, v in enumerate(np.asarray(current_idx, dtype=np.int64).tolist()):
            if v not in pos:
                raise ValueError(
                    f"label-access {name}: current index {v} is not in its partition (size {len(partition)})"
                )
            out[i] = pos[v]
        return out

    matched = _to_positions(matched_source_ranked_idx, train, "matched_source")
    fixed = _to_positions(fixed_source_removal_ranked_idx, train, "fixed_source_removal")
    target = _to_positions(target_ranked_idx, target_pool, "target")

    g = np.asarray(groups) if groups is not None else None
    x_val = emb[val] if len(val) else None
    y_val = y[val] if len(val) else None
    mname = meta.get("method", "?")
    perf.set_identity({k: meta[k] for k in ("seed", "holdout", "method") if k in meta})
    rows: list[dict] = []
    preds: list[dict] = []
    with perf.measure(
        f"probe.label_access/{meta.get('benchmark', '?')}/{mname}",
        n_samples_source=len(train), n_samples_pool=len(target_pool), n_samples_test=len(target_test),
    ):
        probe_fn(
            rows, emb[train], emb[target_pool], emb[target_test], y[train], y[target_pool], y[target_test], seed,
            matched_source_order=matched, fixed_removal_order=fixed, target_order=target,
            meta=meta,
            groups_source=g[train] if g is not None else None,
            groups_pool=g[target_pool] if g is not None else None,
            groups_test=g[target_test] if g is not None else None,
            predictions=preds if write_predictions else None,
            sample_ids_test=target_test,
            sample_ids_full=np.concatenate([target_pool, target_test]),
            x_val=x_val, y_val=y_val, family=family,
        )
    perf.set_identity(None)
    return rows, preds if write_predictions else []


def _run_segmentation_cell(
    bench_mod,
    emb_dir,
    cfg,
    seed,
    family,
    source_budgets,
    target_budgets,
    max_dense_pixels,
    meta,
    *,
    all_folds,
    label_access=None,
    patch_domain=None,
    predictions_sink=None,
) -> list[dict]:
    """Run one dense probe cell against a schema-v2 :class:`DenseSourceTargetSplit`.

    Allocation is patch-level: each partition is a set of patch IDs, streamed across every fold
    directory (``all_folds``) and filtered by patch id -- folds are a cache-layout detail, never a
    split unit. Routing follows the route capability:
      * random_id (has_target=False): evaluate IN DISTRIBUTION on source_test;
      * official (has_target=True, supports_target_labels=False): fit source_train, calibrate on
        source_val, evaluate ZERO-SHOT on target_test -- no target-label access, no target sweep;
      * geographic/spatial (supports_target_labels=True): the source budgets ALSO evaluate zero-shot
        on target_test, AND the target budgets draw few-shot patches ONLY from target_label_pool and
        are scored on the SAME target_test.
    """
    del bench_mod
    from evals import evals as EV

    def sample_dense(patch_ids, sample_seed):
        return cacheutils.load_dense_samples(emb_dir, all_folds, max_dense_pixels, sample_seed, patch_ids=set(patch_ids))

    def stream_dense(patch_ids):
        return cacheutils.iter_dense_tiles(emb_dir, all_folds, patch_ids=set(patch_ids))

    train_patches = set(cfg.source_train_patches)
    val_patches = set(cfg.source_val_patches)
    # official/geographic (has_target) evaluate zero-shot on target_test; random_id on source_test.
    test_patches = set(cfg.target_test_patches if cfg.has_target else cfg.source_test_patches)
    # x_train is the SOURCE-SWEEP training set (globally subsampled). The label-access suite does NOT use
    # it -- it loads its own per-patch-deterministic pixels via load_dense_patch_pixels (patch-first).
    x_train, y_train, groups_train, _, _ = cacheutils.load_dense_samples(
        emb_dir, all_folds, max_dense_pixels, seed, patch_ids=train_patches
    )
    x_val, y_val, _, _, _ = sample_dense(val_patches, seed + 10_000)
    eval_streams = {
        "validation": lambda vp=val_patches: stream_dense(vp),
        "test": lambda tp=test_patches: stream_dense(tp),
    }
    # 80/10/10 within-source reference: source_test is streamed as an UNTOUCHED extra eval scope
    # (never trained or tuned on) when the split carries one -- geographic/spatial.
    source_test_patches = {int(p) for p in cfg.source_test_patches}
    if cfg.has_target and source_test_patches:
        eval_streams["source_test"] = lambda sp=source_test_patches: stream_dense(sp)
    rows: list[dict] = []
    EV.run_probes_segmentation(
        rows, x_train, x_val, y_train, y_val, seed,
        eval_streams=eval_streams, budgets=source_budgets, meta=meta,
        groups_train=groups_train, family=family,
    )
    if cfg.has_target and cfg.supports_target_labels:
        # few-shot patches drawn ONLY from target_label_pool, scored on the SAME target_test patches
        # (never labels from target_test).
        pool_patches = {int(p) for p in cfg.target_label_pool_patches}
        target_test_patches = {int(p) for p in cfg.target_test_patches}
        if label_access is not None:
            # geographic_ood headline: the 13-route patch-level label-access suite REPLACES the legacy
            # target-budget sweep (no duplicate legacy target rows). Other target-label regimes
            # (spatial_cluster_ood) keep the legacy sweep -- label_access is None for them.
            #
            # PATCH-FIRST loader boundary: the suite resolves its patch sets first, then assembles pixels
            # via load_pixels, which subsamples EACH patch deterministically (run seed + patch id) to a
            # per-patch cap sized so the largest route (all source base + whole pool) stays within
            # MAX_DENSE_PIXELS. It does NOT reuse the globally-subsampled x_train above (that stays the
            # source-sweep training set). cap_patches threads the effective PROBE_CAP; patch_domain
            # balances the base pool across source domains.
            n_units = len(cfg.source_train_patches) + len(cfg.target_label_pool_patches)
            per_patch_cap = max(1, int(max_dense_pixels) // max(1, n_units))

            def _load_pixels(patch_ids, _cap=per_patch_cap):
                return cacheutils.load_dense_patch_pixels(
                    emb_dir, all_folds, {int(p) for p in patch_ids}, run_seed=seed, per_patch_cap=_cap
                )

            def _stream_eval(patch_ids):
                return cacheutils.iter_dense_tiles_with_ids(emb_dir, all_folds, patch_ids={int(p) for p in patch_ids})

            EV.run_probes_segmentation_label_access(
                rows, seed,
                source_patches=frozenset(int(p) for p in cfg.source_train_patches),
                pool_patches=frozenset(pool_patches),
                target_test_patches=frozenset(target_test_patches),
                matched_source_order=label_access.matched_source_ranked_patches,
                fixed_removal_order=label_access.fixed_source_removal_ranked_patches,
                target_order=label_access.target_ranked_patches,
                load_pixels=_load_pixels, stream_eval=_stream_eval,
                x_val=x_val, y_val=y_val, meta={**meta, "budget_type": "label_access"},
                family=family, cap_patches=perf.PROBE_CAP, patch_domain=patch_domain,
                predictions_sink=predictions_sink,
            )
        else:
            EV.run_probes_segmentation_target(
                rows, x_train, y_train, seed,
                target_patches=pool_patches | target_test_patches,
                pool_patches=pool_patches, target_test_patches=target_test_patches,
                sample_target=lambda pids, sd: sample_dense(pids, sd),
                stream_target=lambda pids: stream_dense(pids),
                x_val=x_val, y_val=y_val, budgets=target_budgets, meta={**meta, "budget_type": "target"},
                family=family, groups_source=groups_train,
            )
    return rows


def _run_segmentation_pair(
    benchmark_name,
    model_name,
    seeds,
    max_dense_pixels,
    split_regimes,
    run_stages,
    active_probes,
    budget_regimes,
    s2_only,
    overwrite_mode,
    strict_mode,
    write_predictions,
    enc_kwargs,
) -> None:
    from evals import compat, evals as EV, split_artifacts  # noqa: I001
    from evals.regimes import base as regime_base

    stages = validate_run_stages(run_stages)
    gen_embeddings, probing = "gen_embeddings" in stages, "probing" in stages
    source_budgets, target_budgets = EV._budget_lists(budget_regimes)
    bench_mod = EV.load_benchmark(benchmark_name)
    perf.reset()
    bench = cacheutils.cached_bench(bench_mod.BENCHMARK)
    # S2-only common-input mode (mirrors _run_tabular_pair): restrict every model to the shared
    # Sentinel-2 (+ temporal) input so cross-model differences can't be a modality effect. The
    # __s2only suffix isolates the dense cache and results dir from the native run.
    suffix = "__s2only" if s2_only else ""
    artifact = cacheutils.artifact_name(s2_only)
    results_dir = cacheutils.OUTPUT_DIR / "results" / model_name / (benchmark_name + suffix)

    # Regime compatibility, then FAIL-FAST SPLIT VALIDATION -- both moved ABOVE embeddings so an
    # unsupported regime or an invalid/missing patch split refuses the pair BEFORE any dense
    # extraction, cache require, embedding digest, or results-directory mutation (byte-for-byte, even
    # under OVERWRITE). Embedding-only runs (probing disabled) never require split artifacts.
    supported = getattr(bench_mod, "SPLIT_REGIMES", ["random_id"])
    unsupported = [r for r in split_regimes if r not in supported]
    if unsupported:
        raise ValueError(f"Unknown/unsupported split regimes for {benchmark_name}: {unsupported}. Supported: {supported}")
    regimes = [r for r in supported if r in split_regimes]

    splits_by_seed: dict[int, list[Any]] = {}
    patch_fold: dict[int, int] = {}
    patch_tile: dict[int, str | None] = {}
    dense_la_by_cell: dict[tuple[int, str], Any] = {}
    if probing:
        # patch_fold / patch_tile are the CURRENT benchmark patch->fold and patch->tile mappings.
        # patch_fold's keys are the eligible patch universe; its values (folds) are the cache-layout
        # dirs a patch set is streamed from. patch_tile drives the geographic_ood structural check.
        splits_root = cacheutils.SCRATCH / "splits"
        patch_fold = {int(p.patch_id): int(p.fold) for p in getattr(bench, "patches", [])}
        patch_tile = {int(k): v for k, v in getattr(bench, "patch_tiles", {}).items()}  # @property dict
        splits_by_seed = split_artifacts.load_dense_splits(
            splits_root, bench_mod.BENCHMARK, patch_fold, patch_tile, regimes, seeds
        )
        # Fail-fast: load + validate the frozen dense label_access.csv for every geographic_ood headline
        # target BEFORE any probing, so a missing/stale patch order refuses the pair up front.
        for _s in seeds:
            for _ld in splits_by_seed[_s]:
                if _ld.regime == split_artifacts.LABEL_ACCESS_REGIME and _ld.split.supports_target_labels:
                    dense_la_by_cell[(_s, _ld.split.label)] = split_artifacts.load_dense_label_access(
                        splits_root, bench_mod.BENCHMARK, _s, _ld.split
                    )

    bench_for_emb = bench.s2_only() if s2_only else bench
    if gen_embeddings:
        cacheutils.extract_dense_and_cache(
            bench_for_emb,
            bench_mod.BENCHMARK,
            model_name,
            artifact,
            **enc_kwargs,
        )
    emb_dir = (
        cacheutils.require_dense_cache(bench_for_emb, bench_mod.BENCHMARK, model_name, artifact, enc_kwargs.get("weights_path"))
        if probing
        else cacheutils.dense_embedding_cache_dir(bench_mod.BENCHMARK, model_name, artifact)
    )
    if not probing:
        n_events = perf.write_log(results_dir / "perf.jsonl")
        print(f"  embedding stage complete; perf: {n_events} events logged", flush=True)
        return
    emb_digest = cacheutils.embedding_digest(bench_mod.BENCHMARK, model_name, artifact, dense=True)
    rows_path = results_dir / "probe_results.jsonl"

    # Build/check the run manifest only AFTER split loading fixed the consumed path set (validate
    # before mutation: nothing above this point wrote into results_dir).
    manifest = build_run_manifest(
        model_name, benchmark_name, artifact, emb_digest, split_regimes, seeds, enc_kwargs,
        active_probes=active_probes, budget_regimes=budget_regimes, max_dense_pixels=max_dense_pixels,
        write_predictions=write_predictions,
    )
    signature = run_manifest_digest(manifest)
    preds_path = results_dir / "predictions.jsonl"
    check_run_manifest(results_dir, manifest, overwrite_mode=overwrite_mode)
    if overwrite_mode:
        for path in (
            rows_path,
            preds_path,
            results_dir / "probe_results.csv",
            results_dir / "summary.csv",
            results_dir / "deltas.csv",
            results_dir / "label_access_contrasts.csv",          # Stage-5, rewritten fresh each finalize
            results_dir / "label_access_contrasts_summary.csv",
            results_dir / "data_quality.json",
            results_dir / "split_ref.json",          # legacy per-model artifact -- retired, remove on resume
            results_dir / "split_manifest.json",     # legacy per-model artifact -- retired, remove on resume
            results_dir / artifacts.RUN_MANIFEST_FILE,
            results_dir / artifacts.ENVIRONMENT_FILE,
            results_dir / artifacts.RUN_COMPLETE_FILE,
        ):
            if path.exists():
                path.unlink()
    # ORDER MATTERS -- see main.py: every refusal (split loading, then check_run_manifest) ran above;
    # only now do we mutate.
    artifacts.write_environment(results_dir, overwrite_mode=overwrite_mode)
    publish_run_manifest(results_dir, manifest)
    artifacts.invalidate_run_complete(results_dir)
    # Bounded-memory streaming prediction sink for the dense label-access suite: workers append per-patch
    # batches under a lock (joblib threads share this process). Enabled ONLY when write_predictions AND a
    # geographic_ood headline target exists -- so the manifest's write_predictions can never over-claim.
    _preds_active = write_predictions and any(
        ld.regime == split_artifacts.LABEL_ACCESS_REGIME and ld.split.supports_target_labels
        for s in seeds for ld in splits_by_seed[s]
    )
    _preds_lock = threading.Lock()

    def _predictions_sink(records):
        with _preds_lock:
            IOU.append_jsonl(preds_path, records)
    predictions_sink = _predictions_sink if _preds_active else None
    # Attribute regime problems / skipped cells to THIS pair (both accumulators are shard-global).
    regime_problems_before = len(regime_base.REGIME_PROBLEMS)
    cell_failures_before = len(perf.CELL_FAILURES)
    if data_quality := getattr(bench, "data_quality", None):
        IOU.write_json(results_dir / "data_quality.json", data_quality)
    rows = IOU.read_jsonl(rows_path)
    fam_fields = ("seed", "method", "split_regime", "holdout", "probe_family")

    def _fam_key(r):
        return tuple(r.get(k) for k in fam_fields)

    present_by_family: dict[tuple, set] = {}
    for r in rows:
        # Include the label_access_route (empty for non-label-access) so two same-budget routes
        # (e.g. source_plus_target(25) vs fixed_total_mixed(25)) are never confused on resume.
        present_by_family.setdefault(_fam_key(r), set()).add(
            (r.get("budget_type"), r.get("label_budget"), r.get("evaluation_split"), r.get("label_access_route", ""))
        )
    # The 80/10/10 within-source reference (source_test) is evaluated as an extra scope whenever the
    # split HAS a target eval AND carries a source_test partition -- geographic/spatial. random_id's
    # source_test IS its primary eval; official's 90/10 pool has no source_test.
    has_source_diag = {
        (seed, ld.regime, ld.split.label): bool(ld.split.has_target and ld.split.source_test_patches)
        for seed in seeds for ld in splits_by_seed[seed]
    }
    # Target-budget rows are expected ONLY when the regime supports target labels. official
    # (has_target=True, supports_target_labels=False) is zero-shot: source budgets on target_test, no
    # target sweep -- so it must NOT be gated by "regime != random_id".
    supports_target = {
        (seed, ld.regime, ld.split.label): bool(ld.split.supports_target_labels)
        for seed in seeds for ld in splits_by_seed[seed]
    }
    # Every scope tuple is (budget_type, label_budget, evaluation_split, label_access_route). The legacy
    # target sweep and the source sweep carry route "" ; the label-access suite carries a real route.
    expected_target = {("target", b, "held_out", "") for b in target_budgets}
    if any(float(b) == 0.0 for b in target_budgets):
        expected_target.add(("target", 0, "full", ""))
    label_access_expected = {
        ("label_access", b, es, route) for (route, b, es) in split_artifacts.label_access_expected_rows()
    }

    def _expected(key):
        # dense already scores "validation" (the source_val calibration set); the source diagnostic
        # adds ONLY the untouched within-source reference "source_test".
        splits = ("validation", "test")
        if has_source_diag.get((key[0], key[2], key[3]), False):
            splits = (*splits, "source_test")
        exp = {("source", b, s, "") for b in source_budgets for s in splits}
        if supports_target.get((key[0], key[2], key[3]), False):
            # geographic_ood headline runs the label-access suite; other target-label regimes
            # (spatial_cluster_ood) keep the legacy target-budget sweep.
            exp |= label_access_expected if key[2] == split_artifacts.LABEL_ACCESS_REGIME else expected_target
        return exp

    done_families = {
        k for k, seen in present_by_family.items() if _expected(k).issubset(seen)
    }
    incomplete = set(present_by_family) - done_families
    if incomplete:
        rows = [r for r in rows if _fam_key(r) not in incomplete]
        tmp_rows = cacheutils._atomic_tmp(rows_path)
        tmp_rows.unlink(missing_ok=True)
        if rows:
            IOU.append_jsonl(tmp_rows, rows)
        else:
            tmp_rows.touch()
        os.replace(tmp_rows, rows_path)
    # TRANSACTIONAL predictions: keep ONLY the predictions of families that are fully done (whose result
    # rows are all present, so they will NOT rerun). This drops the predictions of every family scheduled
    # to rerun -- including a family that appended predictions but CRASHED before its result rows were
    # published (it has predictions but no rows, so it is not 'done'). Without this, that family's rerun
    # would duplicate its predictions. Runs whenever a predictions file exists, not only on row pruning.
    # (rewrite_jsonl_dropping repairs a torn tail and hard-fails on a corrupt interior row.)
    if write_predictions and preds_path.exists():
        IOU.rewrite_jsonl_dropping(preds_path, lambda p: _fam_key(p) not in done_families)

    # PHASE B: the per-model split_manifest.json is retired -- the frozen patch-level assignments.csv
    # leaves under data/splits/ (discovered + checksum-verified via data/logs/splits.json) are the
    # source of truth; no model-specific split definition is written here.
    # Every cell this pair is supposed to produce, from the config and the REALIZED fold configs.
    # `_expected` already encodes the per-family scope rules (source budgets x {validation,test}
    # plus source diagnostics, and the target sweep only on regimes that support target labels);
    # reuse it so the completeness check and the resume logic cannot disagree about a complete pair.
    expected_keys: set[tuple] = set()
    jobs = []
    # ERM only -- see main.py. The literal keeps method="erm" on every dense row.
    method_name = EV.ERM_METHOD
    method_meta = EV.erm_metadata()
    all_folds = set(patch_fold.values())  # cache-layout fold dirs a patch set is streamed across
    for seed in seeds:
        for loaded_dense in splits_by_seed[seed]:
            split_regime, cfg = loaded_dense.regime, loaded_dense.split
            holdout = cfg.label
            for f in active_probes:
                for bt, lb, es, route in _expected((seed, method_name, split_regime, holdout, f)):
                    # artifacts.CELL_KEY_FIELDS order; the 9th field is the label-access route ("" for the
                    # source sweep and the legacy target sweep; a real route for the label-access suite).
                    expected_keys.add((seed, split_regime, holdout, method_name, f, bt, lb, es, route))
            families_to_run = [
                f for f in active_probes
                if (seed, method_name, split_regime, holdout, f) not in done_families
            ]
            for family in families_to_run:
                seg_meta = {
                    "model": model_name,
                    "benchmark": bench_mod.BENCHMARK,
                    "method": method_name,
                    **method_meta,
                    "split_regime": split_regime,
                    "domain_basis": cfg.group_kind,
                    "holdout": holdout,
                    "target_role": cfg.target_role,
                    "probe_family": family,
                    "label_access_route": "",  # non-label-access default; the sweep overrides per route
                }
                jobs.append(
                    delayed(_run_segmentation_cell)(
                        bench_mod,
                        emb_dir,
                        cfg,
                        seed,
                        family,
                        source_budgets,
                        target_budgets,
                        max_dense_pixels,
                        seg_meta,
                        all_folds=all_folds,
                        label_access=dense_la_by_cell.get((seed, holdout)),
                        patch_domain=patch_tile,     # source patch -> Sentinel tile, for balanced base selection
                        predictions_sink=predictions_sink,
                    )
                )
    if jobs:
        n_features_guess = int(os.environ.get("PASTIS_PROBE_FEATURE_GUESS", "4096"))
        n_jobs = perf._effective_n_jobs(job_bytes=max_dense_pixels * n_features_guess * 4)
        print(f"  segmentation probe jobs={len(jobs)} n_jobs={n_jobs}", flush=True)
        for cell_rows in Parallel(n_jobs=n_jobs, return_as="generator", prefer="threads")(jobs):
            IOU.append_jsonl(rows_path, cell_rows)
            rows.extend(cell_rows)
    IOU.write_csv(results_dir / "probe_results.csv", rows)
    summary = IOU.summarize_rows(
        rows,
        keys=[
            "model",
            "method",
            "probe_family",
            "split_regime",
            "holdout",
            "evaluation_split",
            "budget_type",
            "label_budget",
            "label_access_route",
        ],
        metrics=EV.METRICS_SEGMENTATION,
        # Dense label-access supervision counts are PATCH counts; aggregate them (never key on them) and
        # preserve the "patches" unit -- identical treatment to the tabular summary.
        count_aggregates=["n_source_labels", "n_target_labels", "n_total_labels"],
        passthrough=["label_budget_unit"],
    )
    IOU.write_csv(results_dir / "summary.csv", summary)
    IOU.write_json(results_dir / "metric_roles.json", EV.METRIC_ROLES["segmentation"])
    deltas = IOU.compute_deltas(
        rows,
        EV.METRICS_SEGMENTATION,
        id_source_budget=EV._id_source_budget(source_budgets),
        ood_target_budget=0,
        target_id_budget=EV.TARGET_ID_UPPER_BOUND if EV.TARGET_ID_UPPER_BOUND in target_budgets else None,
    )
    IOU.write_csv(results_dir / "deltas.csv", deltas)
    # Stage 5: paired label-access contrasts (pure post-processing). No-op unless geographic_ood ran;
    # hard-fails on a missing/duplicate operand or unresolvable anchor. write_run_complete re-validates.
    from evals import contrasts
    contrasts.compute_and_write(results_dir, rows)
    declared, available = set(compat.input_modalities(model_name)), {"s2", "s1", "time"}
    IOU.write_json(results_dir / "model_inputs.json", {
        "model": model_name, "benchmark": benchmark_name, "s2_only_mode": s2_only,
        "compatibility_rank": compat.rank(benchmark_name, model_name),
        "adaptation_severity": compat.adaptation_severity(benchmark_name, model_name),
        "declared_modalities": sorted(declared), "available_modalities": sorted(available),
        "effective_modalities": sorted(declared & available),
    })
    perf.write_log(results_dir / "perf.jsonl")

    # LAST write of the dense pair -- deltas.csv above is the final derived artifact. The dense
    # path finalizes separately from the tabular one, so it needs its own marker or PASTIS would
    # be the only benchmark whose directories can never be validated.
    artifacts.write_run_complete(
        results_dir,
        run_manifest_sha256=signature,
        expected_keys=expected_keys,
        # Absent file -> [] so write_run_complete reports it as a missing REQUIRED
        # artifact, rather than the caller dying here on FileNotFoundError.
        rows=artifacts.parse_jsonl_rows(rows_path) if rows_path.exists() else [],
        regime_problems=regime_base.REGIME_PROBLEMS[regime_problems_before:],
        cell_failures=perf.CELL_FAILURES[cell_failures_before:],
        extra={"model": model_name, "benchmark": benchmark_name, "dense": True},
    )
