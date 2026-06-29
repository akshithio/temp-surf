from __future__ import annotations

import os

import numpy as np
from joblib import Parallel, delayed

from utils import cacheutils
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


def run_signature(
    model_name: str,
    tag: str,
    split_regimes,
    seeds,
    enc_kwargs,
    *,
    active_probes,
    budget_regimes,
    max_samples,
    max_dense_pixels,
    write_predictions: bool = True,
) -> str:
    src = cacheutils.REPO / "src"
    code = cacheutils._hash_files(
        src / "main.py",
        *(p for p in [src / "evals" / "probes.py"] if p.exists()),
        *sorted((src / "evals" / "probes").glob("*.py")),
        src / "evals" / "evals.py",
        src / "evals" / "confounds.py",
        src / "evals" / "regimes" / "base.py",
        src / "utils" / "ioutils.py",
        src / "utils" / "cacheutils.py",
        src / "utils" / "perfutils.py",
        src / "utils" / "runstate.py",
        *cacheutils._model_source_files(model_name),
        *sorted((src / "evals" / "regimes").glob("*.py")),
    )
    enc = {k: v for k, v in sorted(enc_kwargs.items()) if k != "device"}
    parts = [
        f"tag={tag}",
        f"ckpt={cacheutils._checkpoint_fingerprint(model_name, enc_kwargs.get('weights_path'))}",
        f"probes={active_probes}",
        f"budgets={budget_regimes}",
        f"seeds={list(seeds)}",
        f"regimes={sorted(split_regimes)}",
        f"max_samples={max_samples}",
        f"max_dense_pixels={max_dense_pixels}",
        f"write_predictions={write_predictions}",
        f"enc={enc}",
        f"code={code}",
    ]
    return cacheutils._hash_str("|".join(map(str, parts)))


def check_run_signature(results_dir, signature: str, *, overwrite_mode: bool) -> None:
    if overwrite_mode:
        return
    sig_path = results_dir / "run_signature.txt"
    rows_path = results_dir / "probe_results.jsonl"
    if sig_path.exists():
        existing = sig_path.read_text().strip()
        if existing != signature:
            raise RuntimeError(
                f"Refusing to resume {results_dir}: signature {existing[:10]!r} != {signature[:10]!r} "
                "(different experiment config, or a corrupt/partial signature). Set OVERWRITE_MODE=True "
                "or remove the directory."
            )
    elif rows_path.exists() and rows_path.stat().st_size > 0:
        raise RuntimeError(
            f"Refusing to resume {results_dir}: it has results but NO run_signature.txt (a pre-guard "
            "or foreign run). Verify they match this config and write the signature, or use "
            "OVERWRITE_MODE=True."
        )


def publish_run_signature(results_dir, signature: str) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    sig_path = results_dir / "run_signature.txt"
    tmp = cacheutils._atomic_tmp(sig_path)
    try:
        tmp.write_text(signature)
        os.replace(tmp, sig_path)
    finally:
        tmp.unlink(missing_ok=True)


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
        preds = IOU.read_jsonl(preds_path)
        kept_preds = [p for p in preds if budget_row_key(p) not in rerun_keys]
        if len(kept_preds) != len(preds):
            tmp_preds = cacheutils._atomic_tmp(preds_path)
            tmp_preds.unlink(missing_ok=True)
            if kept_preds:
                IOU.append_jsonl(tmp_preds, kept_preds)
            else:
                tmp_preds.touch()
            os.replace(tmp_preds, preds_path)
    return kept


def _probe_cell(
    probe_fn,
    emb,
    train,
    val,
    test,
    y,
    groups,
    cls,
    kwargs,
    uses_target,
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
    transform = None
    if cls is not None:
        x_paired = x_cond_te if uses_target else None
        transform = cls(**kwargs)
        with perf.measure(f"method.fit/{mname}", identity=identity, n_samples=len(train), n_features=x_tr.shape[1]):
            transform.fit(x_tr, y_tr, g_tr, x_paired=x_paired)
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
            transform=transform,
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
    cls,
    kwargs,
    uses_target,
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
    transform = None
    if cls is not None:
        x_paired = x_target_full if uses_target else None
        transform = cls(**kwargs)
        with perf.measure(
            f"method.fit/{mname}", identity=identity, n_samples=len(train), n_features=x_source_tr.shape[1]
        ):
            transform.fit(x_source_tr, y_source_tr, g_source_tr, x_paired=x_paired)
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
            transform=transform,
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
    method_name,
    cls,
    kwargs,
    family,
    source_budgets,
    target_budgets,
    max_dense_pixels,
    meta,
) -> list[dict]:
    from evals import evals as EV

    x_train, y_train, groups_train, _, _ = cacheutils.load_dense_samples(
        emb_dir, cfg.train_folds, max_dense_pixels, seed, patch_ids=cfg.train_patches
    )
    x_val, y_val, _, _, _ = cacheutils.load_dense_samples(
        emb_dir, cfg.val_folds, max_dense_pixels, seed + 10_000, patch_ids=cfg.val_patches
    )
    transform = None
    if cls is not None:
        transform = cls(**kwargs)
        identity = {k: meta[k] for k in ("seed", "holdout", "method") if k in meta}
        with perf.measure(
            f"method.fit/{method_name}", identity=identity, n_samples=len(y_train), n_features=x_train.shape[1]
        ):
            transform.fit(x_train, y_train, groups_train, x_paired=None)
    eval_streams = {
        "validation": lambda vf=cfg.val_folds, vp=cfg.val_patches: cacheutils.iter_dense_tiles(
            emb_dir, vf, patch_ids=vp
        ),
        "test": lambda tf=cfg.test_folds, tp=cfg.test_patches: cacheutils.iter_dense_tiles(
            emb_dir, tf, patch_ids=tp
        ),
    }
    if cfg.source_val_patches and cfg.source_test_patches:
        eval_streams["source_validation"] = lambda tf=cfg.train_folds, p=cfg.source_val_patches: cacheutils.iter_dense_tiles(
            emb_dir, tf, patch_ids=p
        )
        eval_streams["source_test"] = lambda tf=cfg.train_folds, p=cfg.source_test_patches: cacheutils.iter_dense_tiles(
            emb_dir, tf, patch_ids=p
        )
    rows: list[dict] = []
    EV.run_probes_segmentation(
        rows,
        x_train,
        x_val,
        y_train,
        y_val,
        seed,
        eval_streams=eval_streams,
        transform=transform,
        budgets=source_budgets,
        meta=meta,
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
            sample_target=lambda pids, sd, tf=cfg.test_folds: cacheutils.load_dense_samples(
                emb_dir, tf, max_dense_pixels, sd, patch_ids=pids
            ),
            stream_target=lambda pids, tf=cfg.test_folds: cacheutils.iter_dense_tiles(
                emb_dir, tf, patch_ids=pids
            ),
            x_val=x_val,
            y_val=y_val,
            transform=transform,
            budgets=target_budgets,
            meta=meta,
            family=family,
        )
    return rows


def _run_segmentation_pair(
    benchmark_name,
    model_name,
    seeds,
    max_samples,
    max_dense_pixels,
    split_regimes,
    run_stages,
    active_probes,
    budget_regimes,
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
    bench_kwargs = dict(max_samples=max_samples, shuffle=True, seed=0)
    tag = cacheutils.bench_tag(bench_mod.BENCHMARK, bench_kwargs)
    perf.reset()
    bench = cacheutils.cached_bench(bench_mod.BENCHMARK, tag, **bench_kwargs)
    if gen_embeddings:
        cacheutils.extract_dense_and_cache(
            bench,
            bench_mod.BENCHMARK,
            model_name,
            tag,
            overwrite=overwrite_mode,
            **enc_kwargs,
        )
    emb_dir = (
        cacheutils.require_dense_cache(bench, bench_mod.BENCHMARK, model_name, tag, enc_kwargs.get("weights_path"))
        if probing
        else cacheutils.dense_embedding_cache_dir(bench, bench_mod.BENCHMARK, model_name, tag, enc_kwargs.get("weights_path"))
    )
    results_dir = cacheutils.OUTPUT_DIR / "results" / model_name / benchmark_name
    if not probing:
        n_events = perf.write_log(results_dir / "perf.jsonl")
        print(f"  embedding stage complete; perf: {n_events} events logged", flush=True)
        return
    signature = run_signature(
        model_name,
        tag,
        split_regimes,
        seeds,
        enc_kwargs,
        active_probes=active_probes,
        budget_regimes=budget_regimes,
        max_samples=max_samples,
        max_dense_pixels=max_dense_pixels,
        write_predictions=write_predictions,
    )
    check_run_signature(results_dir, signature, overwrite_mode=overwrite_mode)
    rows_path = results_dir / "probe_results.jsonl"
    if overwrite_mode:
        for path in (
            rows_path,
            results_dir / "probe_results.csv",
            results_dir / "summary.csv",
            results_dir / "deltas.csv",
            results_dir / "data_quality.json",
            results_dir / "split_manifest.json",
            results_dir / "run_signature.txt",
        ):
            if path.exists():
                path.unlink()
    publish_run_signature(results_dir, signature)
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
    jobs = []
    for seed in seeds:
        for method_name, (cls, kwargs) in EV.build_methods(bench_mod.LABEL_KIND, seed).items():
            for split_regime, cfg in fold_configs_by_seed[seed]:
                holdout = cfg.label
                families_to_run = [
                    f for f in active_probes
                    if (seed, method_name, split_regime, holdout, f) not in done_families
                ]
                for family in families_to_run:
                    seg_meta = {
                        "model": model_name,
                        "benchmark": bench_mod.BENCHMARK,
                        "method": method_name,
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
                            method_name,
                            cls,
                            kwargs,
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
        "model": model_name, "benchmark": benchmark_name, "s2_only_mode": False,
        "compatibility_rank": compat.rank(benchmark_name, model_name),
        "adaptation_severity": compat.adaptation_severity(benchmark_name, model_name),
        "declared_modalities": sorted(declared), "available_modalities": sorted(available),
        "effective_modalities": sorted(declared & available),
    })
    perf.write_log(results_dir / "perf.jsonl")
