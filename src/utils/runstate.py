"""Run-stage validation, result signatures, and resume-state helpers."""

from __future__ import annotations

import os

import numpy as np

from utils import cacheutils
from utils import ioutils as IOU
from utils import perfutils as perf

VALID_RUN_STAGES = {"gen_embeddings", "probing"}

def validate_run_stages(run_stages: list[str]) -> set[str]:
    """Validate the configured run stages."""
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
) -> str:
    """Fingerprint the result-defining experiment inputs and source files."""
    src = cacheutils.REPO / "src"
    code = cacheutils._hash_files(
        src / "main.py",
        src / "evals" / "probes.py",
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
        f"enc={enc}",
        f"code={code}",
    ]
    return cacheutils._hash_str("|".join(map(str, parts)))


def check_run_signature(results_dir, signature: str, *, overwrite_mode: bool) -> None:
    """Refuse to resume a result directory from a different experiment."""
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
    """Write the run signature atomically."""
    results_dir.mkdir(parents=True, exist_ok=True)
    sig_path = results_dir / "run_signature.txt"
    tmp = cacheutils._atomic_tmp(sig_path)
    try:
        tmp.write_text(signature)
        os.replace(tmp, sig_path)
    finally:
        tmp.unlink(missing_ok=True)


def budget_row_key(row):
    """Budget-level identity of a result or prediction row."""
    return (
        row.get("seed"),
        row.get("split_regime"),
        row.get("holdout"),
        row.get("method"),
        row.get("probe_family"),
        row.get("budget_type"),
        row.get("label_budget"),
    )


def prune_partial_budgets(rows, rows_path, preds_path, rerun_keys):
    """Remove existing rows/predictions for budgets that are about to be regenerated."""
    if not rerun_keys:
        return rows
    kept = [r for r in rows if budget_row_key(r) not in rerun_keys]
    if len(kept) != len(rows):
        rows_path.unlink(missing_ok=True)
        IOU.append_jsonl(rows_path, kept)
    preds = IOU.read_jsonl(preds_path)
    kept_preds = [p for p in preds if budget_row_key(p) not in rerun_keys]
    if len(kept_preds) != len(preds):
        preds_path.unlink(missing_ok=True)
        IOU.append_jsonl(preds_path, kept_preds)
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
) -> tuple[list[dict], list[dict]]:
    """Fit the optional method transform and run the source-budget probe."""
    x_tr, x_cond_te = emb[train], emb[test]
    y_tr, y_te, g_tr = y[train], y[test], groups[train]
    x_val = emb[val] if len(val) else None
    y_val = y[val] if len(val) else None
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
            predictions=preds,
            sample_ids_test=np.asarray(test),
            groups_test=np.asarray(groups)[test],
            x_val=x_val,
            y_val=y_val,
            family=family,
            **({} if budgets is None else {"budgets": budgets}),
        )
    perf.set_identity(None)
    return rows, preds


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
) -> tuple[list[dict], list[dict]]:
    """Target-budget variant using the full target pool."""
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
            predictions=preds,
            sample_ids_target=np.asarray(test),
            groups_target=np.asarray(groups)[test],
            x_val=x_val,
            y_val=y_val,
            family=family,
            **({} if budgets is None else {"budgets": budgets}),
        )
    perf.set_identity(None)
    return rows, preds


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
    enc_kwargs,
) -> None:
    """Run dense PASTIS-R execution over fold-based regimes."""
    from evals import compat
    from evals import evals as EV
    from evals.regimes import base as regime_base

    stages = validate_run_stages(run_stages)
    gen_embeddings = "gen_embeddings" in stages
    probing = "probing" in stages
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
    )
    check_run_signature(results_dir, signature, overwrite_mode=overwrite_mode)
    rows_path = results_dir / "probe_results.jsonl"
    if overwrite_mode:
        for path in (
            rows_path,
            results_dir / "probe_results.csv",
            results_dir / "summary.csv",
            results_dir / "deltas.csv",
            results_dir / "split_manifest.json",
            results_dir / "run_signature.txt",
        ):
            if path.exists():
                path.unlink()
    publish_run_signature(results_dir, signature)
    rows = IOU.read_jsonl(rows_path)
    fam_fields = ("seed", "method", "split_regime", "holdout", "probe_family")

    def _fam_key(r):
        return tuple(r.get(k) for k in fam_fields)

    present_by_family: dict[tuple, set] = {}
    for r in rows:
        present_by_family.setdefault(_fam_key(r), set()).add(
            (r.get("budget_type"), r.get("label_budget"), r.get("evaluation_split"))
        )
    expected_source = {("source", b, s) for b in source_budgets for s in ("validation", "test")}
    expected_target = {("target", b, "held_out") for b in target_budgets}
    if any(float(b) == 0.0 for b in target_budgets):
        expected_target.add(("target", 0, "full"))

    def _expected(regime):
        exp = set(expected_source)
        if regime == "geographic_ood":
            exp |= expected_target
        return exp

    regime_idx = fam_fields.index("split_regime")
    done_families = {
        k for k, seen in present_by_family.items() if _expected(k[regime_idx]).issubset(seen)
    }
    incomplete = set(present_by_family) - done_families
    if incomplete:
        rows = [r for r in rows if _fam_key(r) not in incomplete]
        rows_path.unlink(missing_ok=True)
        IOU.append_jsonl(rows_path, rows)

    supported = getattr(bench_mod, "SPLIT_REGIMES", ["random_id"])
    regimes = [r for r in supported if r in split_regimes]
    fold_configs = list(regime_base.segmentation_fold_configs(bench_mod, regimes, overwrite_mode=overwrite_mode))
    EV._write_split_manifest(
        results_dir,
        [
            EV._segmentation_split_manifest_entry(
                model_name=model_name,
                benchmark_name=benchmark_name,
                seed=seed,
                split_regime=split_regime,
                holdout=holdout,
                train_folds=set(train_folds),
                val_folds=set(val_folds),
                test_folds=set(test_folds),
                emb_dir=emb_dir,
            )
            for seed in seeds
            for split_regime, holdout, train_folds, val_folds, test_folds in fold_configs
        ],
    )
    for seed in seeds:
        for method_name, (cls, kwargs) in EV.build_methods(bench_mod.LABEL_KIND, seed).items():
            for split_regime, holdout, train_folds, val_folds, test_folds in fold_configs:
                families_to_run = [
                    f for f in active_probes
                    if (seed, method_name, split_regime, holdout, f) not in done_families
                ]
                if not families_to_run:
                    continue
                x_train, y_train, groups_train, _, _ = cacheutils.load_dense_samples(
                    emb_dir, train_folds, max_dense_pixels, seed
                )
                x_val, y_val, _, _, _ = cacheutils.load_dense_samples(
                    emb_dir, val_folds, max_dense_pixels, seed + 10_000
                )
                transform = None
                if cls is not None:
                    transform = cls(**kwargs)
                    transform.fit(x_train, y_train, groups_train, x_paired=None)
                for family in families_to_run:
                    cell_rows: list[dict] = []
                    seg_meta = {
                        "model": model_name,
                        "benchmark": bench_mod.BENCHMARK,
                        "method": method_name,
                        "split_regime": split_regime,
                        "domain_basis": "geography",
                        "holdout": holdout,
                        "probe_family": family,
                    }
                    EV.run_probes_segmentation(
                        cell_rows,
                        x_train,
                        x_val,
                        y_train,
                        y_val,
                        seed,
                        eval_streams={
                            "validation": lambda vf=val_folds: cacheutils.iter_dense_tiles(emb_dir, vf),
                            "test": lambda tf=test_folds: cacheutils.iter_dense_tiles(emb_dir, tf),
                        },
                        transform=transform,
                        budgets=source_budgets,
                        meta=seg_meta,
                        family=family,
                    )
                    if split_regime == "geographic_ood":
                        EV.run_probes_segmentation_target(
                            cell_rows,
                            x_train,
                            y_train,
                            seed,
                            target_patches=cacheutils.dense_fold_patches(emb_dir, test_folds),
                            sample_target=lambda pids, sd, tf=test_folds: cacheutils.load_dense_samples(
                                emb_dir, tf, max_dense_pixels, sd, patch_ids=pids
                            ),
                            stream_target=lambda pids, tf=test_folds: cacheutils.iter_dense_tiles(
                                emb_dir, tf, patch_ids=pids
                            ),
                            x_val=x_val,
                            y_val=y_val,
                            transform=transform,
                            budgets=target_budgets,
                            meta=seg_meta,
                            family=family,
                        )
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
    declared = set(compat.input_modalities(model_name))
    available = {"s2", "s1", "time"}
    IOU.write_json(results_dir / "model_inputs.json", {
        "model": model_name, "benchmark": benchmark_name, "s2_only_mode": False,
        "declared_modalities": sorted(declared),
        "available_modalities": sorted(available),
        "effective_modalities": sorted(declared & available),
    })
    perf.write_log(results_dir / "perf.jsonl")
