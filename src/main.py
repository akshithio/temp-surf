"""Orchestrator for the frozen-embedding robustness pipeline.

Edit the config block below, then run:

    cd src && python main.py
    cd src && python utils/gputils.py
"""

from __future__ import annotations

import os
import sys
from typing import Any

import numpy as np
from joblib import Parallel, delayed

from evals import compat
from evals import evals as EV
from evals.regimes import base as regime_base
from utils import cacheutils, gputils, runstate
from utils import ioutils as IOU
from utils import perfutils as perf

# === Configuration ===========================================================
BENCHMARKS = ["cropharvest", "eurocropsml", "breizhcrops", "pastis"]
RUN_STAGES = ["gen_embeddings", "probing"]
SPLIT_REGIMES = ["random_id", "official", "geographic_ood", "spatial_cluster_ood"]
ACTIVE_PROBES = ["logistic"]
BUDGET_REGIMES = {
    "source": [0.05, 0.10, 0.25, 1.0],
    "target": [0, 5, 10, 25, 50, EV.TARGET_ID_UPPER_BOUND],
}
MAX_SAMPLES = None
MAX_DENSE_PIXELS = 50_000  # sampled pixels per PASTIS fold partition
OVERWRITE_MODE = False
SEEDS = [0]
# =============================================================================

# Downstream loaders use this to decide whether partial/corrupt inputs warn or fail.
os.environ["OVERWRITE_MODE"] = "1" if OVERWRITE_MODE else ""


def _run_tabular_pair(
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
    stages = runstate.validate_run_stages(run_stages)
    gen_embeddings = "gen_embeddings" in stages
    probing = "probing" in stages
    source_budgets, target_budgets = EV._budget_lists(budget_regimes)
    bench_mod = EV.load_benchmark(benchmark_name)
    probe_fn_src, metrics = {
        "binary": (EV.run_probes, EV.METRICS_BINARY),
        "multiclass": (EV.run_probes_multiclass, EV.METRICS_MULTICLASS),
    }[bench_mod.LABEL_KIND]
    probe_fn_tgt, _ = {
        "binary": (EV.run_probes_target, EV.METRICS_BINARY),
        "multiclass": (EV.run_probes_multiclass_target, EV.METRICS_MULTICLASS),
    }[bench_mod.LABEL_KIND]
    supported = getattr(bench_mod, "SPLIT_REGIMES", split_regimes)
    split_regimes = [r for r in split_regimes if r in supported]

    bench_kwargs = dict(max_samples=max_samples, shuffle=True, seed=0)
    tag = cacheutils.bench_tag(bench_mod.BENCHMARK, bench_kwargs)
    perf.reset()
    bench = cacheutils.cached_bench(bench_mod.BENCHMARK, tag, **bench_kwargs)
    s2_only = os.environ.get("RB_S2_ONLY", "").strip().lower() not in ("", "0", "false", "no")
    suffix = "__s2only" if s2_only else ""
    if s2_only:
        bench = bench.s2_only()
    emb_tag = tag + suffix
    y, _native_groups = bench_mod.make_targets(bench)
    if gen_embeddings:
        emb = cacheutils.extract_and_cache(
            bench, bench_mod.BENCHMARK, model_name, emb_tag, overwrite=overwrite_mode, **enc_kwargs
        )
    else:
        emb = cacheutils.load_cached_embeddings(
            bench, bench_mod.BENCHMARK, model_name, emb_tag, enc_kwargs.get("weights_path")
        )

    results_dir = cacheutils.OUTPUT_DIR / "results" / model_name / (benchmark_name + suffix)
    if not probing:
        n_events = perf.write_log(results_dir / "perf.jsonl")
        print(f"  embedding stage complete; perf: {n_events} events logged", flush=True)
        return
    signature = runstate.run_signature(
        model_name,
        emb_tag,
        split_regimes,
        seeds,
        enc_kwargs,
        active_probes=active_probes,
        budget_regimes=budget_regimes,
        max_samples=max_samples,
        max_dense_pixels=max_dense_pixels,
    )
    runstate.check_run_signature(results_dir, signature, overwrite_mode=overwrite_mode)
    rows_path = results_dir / "probe_results.jsonl"
    preds_path = results_dir / "predictions.jsonl"

    if overwrite_mode:
        for p in [
            rows_path,
            preds_path,
            results_dir / "probe_results.csv",
            results_dir / "summary.csv",
            results_dir / "deltas.csv",
            results_dir / "split_manifest.json",
            results_dir / "run_signature.txt",
        ]:
            if p.exists():
                p.unlink()
    runstate.publish_run_signature(results_dir, signature)

    rows = IOU.read_jsonl(rows_path)
    done = {
        (
            r.get("seed"),
            r.get("split_regime"),
            r.get("holdout"),
            r.get("method"),
            r.get("probe_family"),
            r.get("budget_type"),
            r.get("label_budget"),
            r.get("evaluation_split"),
        )
        for r in rows
    }

    jobs = []

    def uses_target_flag(cls):
        return getattr(cls, "USES_TARGET", False)

    def _scopes(budget_type, b):
        if budget_type == "target":
            return ("full", "held_out") if b == 0 else ("held_out",)
        return ("test",)

    def _missing(base, budget_type, expected):
        return [b for b in expected if not all((*base, budget_type, b, sc) in done for sc in _scopes(budget_type, b))]

    rerun_keys: set = set()
    split_specs: list[tuple] = []
    split_manifest: list[dict[str, Any]] = []

    for seed in seeds:
        for split_regime in split_regimes:
            holdouts = regime_base.holdouts_for(bench_mod, split_regime)
            for split_label, train, val, test, groups, has_target, domain_basis in regime_base.iter_splits(
                split_regime,
                bench,
                y,
                holdouts,
                seed,
                overwrite_mode=overwrite_mode,
                val_group=regime_base.val_group_for(bench_mod, split_regime),
            ):
                split_specs.append(
                    (seed, split_regime, split_label, train, val, test, groups, has_target, domain_basis)
                )
                split_manifest.append(
                    EV._split_manifest_entry(
                        model_name=model_name,
                        benchmark_name=benchmark_name,
                        seed=seed,
                        split_regime=split_regime,
                        domain_basis=domain_basis,
                        holdout=split_label,
                        train=train,
                        val=val,
                        test=test,
                        domains=groups,
                        labels=y,
                    )
                )
    EV._write_split_manifest(results_dir, split_manifest)

    for seed in seeds:
        seed_split_specs = [spec for spec in split_specs if spec[0] == seed]
        for mname, (cls, kwargs) in EV.build_methods(bench_mod.LABEL_KIND, seed).items():
            for _, split_regime, split_label, train, val, test, groups, has_target, domain_basis in seed_split_specs:
                for family in active_probes:
                    meta = {
                        "model": model_name,
                        "benchmark": bench_mod.BENCHMARK,
                        "method": mname,
                        "split_regime": split_regime,
                        "domain_basis": domain_basis,
                        "holdout": split_label,
                        "probe_family": family,
                    }
                    base = (seed, split_regime, split_label, mname, family)
                    if has_target:
                        todo = _missing(base, "target", target_budgets)
                        if todo:
                            rerun_keys.update((*base, "target", b) for b in todo)
                            jobs.append(
                                delayed(runstate._probe_cell_target)(
                                    probe_fn_tgt,
                                    emb,
                                    train,
                                    val,
                                    test,
                                    y,
                                    groups,
                                    cls,
                                    kwargs,
                                    uses_target_flag(cls),
                                    {**meta, "budget_type": "target"},
                                    seed,
                                    family,
                                    todo,
                                )
                            )
                    todo_src = _missing(base, "source", source_budgets)
                    if todo_src:
                        rerun_keys.update((*base, "source", b) for b in todo_src)
                        jobs.append(
                            delayed(runstate._probe_cell)(
                                probe_fn_src,
                                emb,
                                train,
                                val,
                                test,
                                y,
                                groups,
                                cls,
                                kwargs,
                                uses_target_flag(cls),
                                {**meta, "budget_type": "source"},
                                seed,
                                family,
                                todo_src,
                            )
                        )

    rows = runstate.prune_partial_budgets(rows, rows_path, preds_path, rerun_keys)

    if jobs:
        n_jobs = perf._effective_n_jobs(emb)
        print(f"  probe jobs={len(jobs)} n_jobs={n_jobs}", flush=True)
        for cell_rows, cell_preds in Parallel(n_jobs=n_jobs, return_as="generator", prefer="threads")(jobs):
            if cell_preds:
                IOU.append_jsonl(preds_path, cell_preds)
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
            "domain_basis",
            "budget_type",
            "label_budget",
            "evaluation_split",
        ],
        metrics=metrics,
    )
    IOU.write_csv(results_dir / "summary.csv", summary)
    IOU.write_json(results_dir / "metric_roles.json", EV.METRIC_ROLES[bench_mod.LABEL_KIND])

    deltas = IOU.compute_deltas(
        rows,
        metrics,
        predictions=IOU.read_jsonl(preds_path),
        id_source_budget=EV._id_source_budget(source_budgets),
        ood_target_budget=0,
        target_id_budget=EV.TARGET_ID_UPPER_BOUND if EV.TARGET_ID_UPPER_BOUND in target_budgets else None,
    )
    IOU.write_csv(results_dir / "deltas.csv", deltas)

    from evals import confounds

    axes = {"geography": np.asarray(bench.groups), "class": np.asarray(y)}
    if getattr(bench, "years", None) is not None:
        axes["year"] = np.asarray(bench.years)
    IOU.write_json(results_dir / "domain_confounds.json", confounds.domain_confound_report(axes))

    declared = set(compat.input_modalities(model_name))
    available = bench.available_modalities()
    IOU.write_json(
        results_dir / "model_inputs.json",
        {
            "model": model_name,
            "benchmark": benchmark_name,
            "s2_only_mode": s2_only,
            "declared_modalities": sorted(declared),
            "available_modalities": sorted(available),
            "effective_modalities": sorted(declared & available),
        },
    )

    perf_path = results_dir / "perf.jsonl"
    n_events = perf.write_log(perf_path)
    print(f"  perf: {n_events} events logged to {perf_path}", flush=True)


def run_pair(
    *,
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
    """Run one configured model/benchmark pair."""
    bench_mod = EV.load_benchmark(benchmark_name)
    if bench_mod.LABEL_KIND == "segmentation":
        runstate._run_segmentation_pair(
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
        )
        return
    _run_tabular_pair(
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
    )


def main() -> int:
    regime_base.clear_regime_problems()
    enc_kwargs = {"device": gputils.device()}

    all_pairs = [(mod, bm) for bm in BENCHMARKS for mod in compat.eligible_models(bm)]
    work = gputils.take_shard(all_pairs)
    shard, nshards = gputils.shard_indices()
    failures: list[tuple[str, str, str]] = []

    for mod, bm in work:
        print(f"\n========== [shard {shard}/{nshards}] {mod} / {bm} ==========", flush=True)
        print(f"  split_regimes={SPLIT_REGIMES}", flush=True)
        print(f"  run_stages={RUN_STAGES}", flush=True)
        try:
            run_pair(
                benchmark_name=bm,
                model_name=mod,
                seeds=SEEDS,
                max_samples=MAX_SAMPLES,
                max_dense_pixels=MAX_DENSE_PIXELS,
                split_regimes=SPLIT_REGIMES,
                run_stages=RUN_STAGES,
                active_probes=ACTIVE_PROBES,
                budget_regimes=BUDGET_REGIMES,
                overwrite_mode=OVERWRITE_MODE,
                enc_kwargs=enc_kwargs,
            )
        except NotImplementedError as exc:
            print(f"   [shard {shard}] {mod}/{bm} skipped (not implemented): {exc}", flush=True)
        except cacheutils.MissingEmbeddingCache:
            raise
        except Exception as exc:
            import traceback

            failures.append((mod, bm, f"{type(exc).__name__}: {exc}"))
            print(
                f"!! [shard {shard}] {mod}/{bm} FAILED: {type(exc).__name__}: {exc} (continuing; re-run to resume)",
                flush=True,
            )
            traceback.print_exc()

    regime_base.report_regime_problems()

    if failures:
        bar = "!" * 78
        print(f"\n{bar}\n[shard {shard}/{nshards}] {len(failures)} (model, benchmark) pair(s) FAILED:", flush=True)
        for mod, bm, reason in failures:
            print(f"  - {mod}/{bm}: {reason}", flush=True)
        print(f"{bar}", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
