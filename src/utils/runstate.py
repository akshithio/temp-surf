from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

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
    (artifact SHA for tabular, tile-set digest for dense)."""
    from evals import probes as _probes

    fi = cacheutils.frozen_run_identity()
    env = artifacts.capture_environment()
    enc = {k: v for k, v in sorted(enc_kwargs.items()) if k != "device"}
    return {
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
        "write_predictions": bool(write_predictions),
        "enc": enc,
    }


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
    test,
    y,
    groups,
    meta,
    seed,
    family="logistic",
    budgets=None,
    write_predictions: bool = True,
) -> tuple[list[dict], list[dict]]:
    x_source_tr, x_target_full = emb[train], emb[test]
    y_source_tr, y_target_full = y[train], y[test]
    g_source_tr = groups[train]
    x_val = emb[val] if len(val) else None
    y_val = y[val] if len(val) else None
    mname = meta.get("method", "?")
    identity = {k: meta[k] for k in ("seed", "holdout", "method") if k in meta}
    rows: list[dict] = []
    perf.set_identity(identity)
    with perf.measure(
        f"probe.target/{meta.get('benchmark', '?')}/{mname}",
        n_samples_source=len(train),
        n_samples_target=len(test),
        n_features=x_source_tr.shape[1],
    ):
        preds: list[dict] = []
        probe_fn(
            rows,
            x_source_tr,
            x_target_full,
            y_source_tr,
            y_target_full,
            seed,
            meta=meta,
            groups_source=g_source_tr,
            predictions=preds if write_predictions else None,
            sample_ids_target=np.asarray(test),
            groups_target=np.asarray(groups)[test],
            x_val=x_val,
            y_val=y_val,
            family=family,
            **({} if budgets is None else {"budgets": budgets}),
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
) -> list[dict]:
    from evals import evals as EV

    def sample_dense(folds, sample_seed, patch_ids=None):
        return cacheutils.load_dense_samples(
            emb_dir,
            folds,
            max_dense_pixels,
            sample_seed,
            patch_ids=patch_ids,
        )

    def stream_dense(folds, patch_ids=None):
        return cacheutils.iter_dense_tiles(emb_dir, folds, patch_ids=patch_ids)

    x_train, y_train, groups_train, _, _ = cacheutils.load_dense_samples(
        emb_dir,
        cfg.train_folds,
        max_dense_pixels,
        seed,
        patch_ids=cfg.train_patches,
    )
    x_val, y_val, _, _, _ = sample_dense(cfg.val_folds, seed + 10_000, cfg.val_patches)
    eval_streams = {
        "validation": lambda vf=cfg.val_folds, vp=cfg.val_patches: stream_dense(vf, vp),
        "test": lambda tf=cfg.test_folds, tp=cfg.test_patches: stream_dense(tf, tp),
    }
    if cfg.source_val_patches and cfg.source_test_patches:
        eval_streams["source_validation"] = lambda tf=cfg.train_folds, p=cfg.source_val_patches: stream_dense(tf, p)
        eval_streams["source_test"] = lambda tf=cfg.train_folds, p=cfg.source_test_patches: stream_dense(tf, p)
    rows: list[dict] = []
    EV.run_probes_segmentation(
        rows,
        x_train,
        x_val,
        y_train,
        y_val,
        seed,
        eval_streams=eval_streams,
        budgets=source_budgets,
        meta=meta,
        groups_train=groups_train,
        family=family,
    )
    if cfg.has_target:
        target_patch_ids = cfg.test_patches or set(cacheutils.dense_fold_patches(emb_dir, cfg.test_folds))
        EV.run_probes_segmentation_target(
            rows,
            x_train,
            y_train,
            seed,
            target_patches=target_patch_ids,
            sample_target=lambda pids, sd, tf=cfg.test_folds: sample_dense(tf, sd, pids),
            stream_target=lambda pids, tf=cfg.test_folds: stream_dense(tf, pids),
            x_val=x_val,
            y_val=y_val,
            budgets=target_budgets,
            meta=meta,
            family=family,
            groups_source=groups_train,
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
    from evals import compat, evals as EV  # noqa: I001
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
    results_dir = cacheutils.OUTPUT_DIR / "results" / model_name / (benchmark_name + suffix)
    if not probing:
        n_events = perf.write_log(results_dir / "perf.jsonl")
        print(f"  embedding stage complete; perf: {n_events} events logged", flush=True)
        return
    emb_digest = cacheutils.embedding_digest(bench_mod.BENCHMARK, model_name, artifact, dense=True)
    manifest = build_run_manifest(
        model_name,
        benchmark_name,
        artifact,
        emb_digest,
        split_regimes,
        seeds,
        enc_kwargs,
        active_probes=active_probes,
        budget_regimes=budget_regimes,
        max_dense_pixels=max_dense_pixels,
        write_predictions=write_predictions,
    )
    signature = run_manifest_digest(manifest)
    check_run_manifest(results_dir, manifest, overwrite_mode=overwrite_mode)
    rows_path = results_dir / "probe_results.jsonl"
    if overwrite_mode:
        for path in (
            rows_path,
            results_dir / "probe_results.csv",
            results_dir / "summary.csv",
            results_dir / "deltas.csv",
            results_dir / "data_quality.json",
            results_dir / "split_manifest.json",
            results_dir / artifacts.RUN_MANIFEST_FILE,
            results_dir / artifacts.ENVIRONMENT_FILE,
            results_dir / artifacts.RUN_COMPLETE_FILE,
        ):
            if path.exists():
                path.unlink()
    # ORDER MATTERS -- see main.py: every check that can REFUSE this resume runs before anything
    # is mutated, so a refused resume leaves an existing valid completion marker untouched.
    artifacts.write_environment(results_dir, overwrite_mode=overwrite_mode)
    publish_run_manifest(results_dir, manifest)
    # See main.py: a resume is about to make this directory incomplete again.
    artifacts.invalidate_run_complete(results_dir)
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
        present_by_family.setdefault(_fam_key(r), set()).add(
            (r.get("budget_type"), r.get("label_budget"), r.get("evaluation_split"))
        )
    supported = getattr(bench_mod, "SPLIT_REGIMES", ["random_id"])
    unsupported = [r for r in split_regimes if r not in supported]
    if unsupported:
        raise ValueError(f"Unknown/unsupported split regimes for {benchmark_name}: {unsupported}. Supported: {supported}")
    regimes = [r for r in supported if r in split_regimes]
    fold_configs_by_seed = {
        seed: list(
            regime_base.segmentation_fold_configs(
                bench_mod, regimes, seed=seed, emb_dir=emb_dir, strict_mode=strict_mode, bench=bench
            )
        )
        for seed in seeds
    }
    has_source_diag = {
        (seed, regime, cfg.label): bool(cfg.source_val_patches and cfg.source_test_patches)
        for seed in seeds for regime, cfg in fold_configs_by_seed[seed]
    }
    expected_target = {("target", b, "held_out") for b in target_budgets}
    if any(float(b) == 0.0 for b in target_budgets):
        expected_target.add(("target", 0, "full"))

    def _expected(key):
        splits = ("validation", "test")
        if has_source_diag.get((key[0], key[2], key[3]), False):
            splits = (*splits, "source_validation", "source_test")
        exp = {("source", b, s) for b in source_budgets for s in splits}
        if key[regime_idx] != "random_id":
            exp |= expected_target
        return exp

    regime_idx = fam_fields.index("split_regime")
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

    EV._write_split_manifest(
        results_dir,
        [
            EV._segmentation_split_manifest_entry(
                model_name=model_name,
                benchmark_name=benchmark_name,
                seed=seed,
                split_regime=regime,
                holdout=cfg.label,
                train_folds=cfg.train_folds,
                val_folds=cfg.val_folds,
                test_folds=cfg.test_folds,
                train_patches=cfg.train_patches,
                val_patches=cfg.val_patches,
                test_patches=cfg.test_patches,
                emb_dir=emb_dir,
            )
            for seed in seeds
            for regime, cfg in fold_configs_by_seed[seed]
        ],
    )
    # Every cell this pair is supposed to produce, from the config and the REALIZED fold configs.
    # `_expected` already encodes the per-family scope rules (source budgets x {validation,test}
    # plus source diagnostics, and the target sweep on non-random_id regimes); reuse it so the
    # completeness check and the resume logic cannot disagree about what a complete pair is.
    expected_keys: set[tuple] = set()
    jobs = []
    # ERM only -- see main.py. The literal keeps method="erm" on every dense row.
    method_name = EV.ERM_METHOD
    method_meta = EV.erm_metadata()
    for seed in seeds:
        for split_regime, cfg in fold_configs_by_seed[seed]:
            holdout = cfg.label
            for f in active_probes:
                for bt, lb, es in _expected((seed, method_name, split_regime, holdout, f)):
                    # artifacts.CELL_KEY_FIELDS order
                    expected_keys.add((seed, split_regime, holdout, method_name, f, bt, lb, es))
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
                    "probe_family": family,
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
        ],
        metrics=EV.METRICS_SEGMENTATION,
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
