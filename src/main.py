"""Orchestrator for the frozen-embedding robustness pipeline.

Edit the config block below, then run:

    cd src && python main.py
"""

from __future__ import annotations

import os
import sys
from typing import Any

for _thread_var in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"]:
    os.environ.setdefault(_thread_var, "1")

import numpy as np  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402

from evals import compat  # noqa: E402
from evals import evals as EV  # noqa: E402
from evals.regimes import base as regime_base  # noqa: E402
from utils import artifacts, cacheutils, gputils, runstate  # noqa: E402
from utils import ioutils as IOU  # noqa: E402
from utils import perfutils as perf  # noqa: E402

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
STRICT_MODE = False
WRITE_PREDICTIONS = True
LAUNCH_GPU_SHARDS = True
GPU_SHARDS = None
SEEDS = [0, 1, 2]
# =============================================================================

# --- Per-machine benchmark placement (the only value that differs across
# machines; everything else above is uniform and synced identically). ---
_rb_benchmarks = os.environ.get("RB_BENCHMARKS", "").strip()
if _rb_benchmarks:
    _requested = [b.strip() for b in _rb_benchmarks.split(",") if b.strip()]
    _unknown = [b for b in _requested if b not in BENCHMARKS]
    if _unknown:
        raise ValueError(
            f"RB_BENCHMARKS has unknown benchmark(s) {_unknown}; valid: {BENCHMARKS}"
        )
    BENCHMARKS = _requested

# --- Per-machine model restriction. Empty/unset = all eligible models for the
# selected benchmarks (the normal case). Used only when splitting one
# benchmark's models across machines (e.g. pastis presto/raw on one box, the
# heavy encoders on another). Validated against eligibility in main(). ---
_rb_models = os.environ.get("RB_MODELS", "").strip()
RB_MODELS = [m.strip() for m in _rb_models.split(",") if m.strip()] if _rb_models else None

# --- Per-machine seed restriction. Empty/unset = all seeds (the normal case).
# Used only when splitting one (model, benchmark) cell's seeds across machines
# to parallelize a slow probe; results merge cleanly at collation since the
# embedding cache is content-addressed and rows are keyed by seed. ---
_rb_seeds = os.environ.get("RB_SEEDS", "").strip()
if _rb_seeds:
    _requested_seeds = [int(s.strip()) for s in _rb_seeds.split(",") if s.strip()]
    _unknown_seeds = [s for s in _requested_seeds if s not in SEEDS]
    if _unknown_seeds:
        raise ValueError(
            f"RB_SEEDS has seed(s) {_unknown_seeds} not in {SEEDS}"
        )
    SEEDS = _requested_seeds

# --- Probe-family override. Unset = ACTIVE_PROBES above (logistic, the headline
# probe). Used for the probe-capacity robustness check: re-probe cached
# embeddings with the MLP family into a fresh RB_OUTPUT_DIR, then compare the
# decomposition to the logistic run. Validated against the known families. ---
_rb_probes = os.environ.get("RB_ACTIVE_PROBES", "").strip()
if _rb_probes:
    _requested_probes = [p.strip() for p in _rb_probes.split(",") if p.strip()]
    _unknown_probes = [p for p in _requested_probes if p not in ("logistic", "mlp", "knn")]
    if _unknown_probes:
        raise ValueError(
            f"RB_ACTIVE_PROBES has unknown probe(s) {_unknown_probes}; known: logistic, mlp, knn"
        )
    ACTIVE_PROBES = _requested_probes

# --- Split-regime restriction. Unset = SPLIT_REGIMES above (all regimes, the
# normal case). Used for the probe-capacity cap-sensitivity check, which targets
# a single regime (geographic_ood) so the decomposition verdict can be compared
# across caps without recomputing every regime. Per-benchmark unsupported regimes
# are still filtered downstream; this only narrows the global set. ---
_rb_regimes = os.environ.get("RB_SPLIT_REGIMES", "").strip()
if _rb_regimes:
    _requested_regimes = [r.strip() for r in _rb_regimes.split(",") if r.strip()]
    _unknown_regimes = [r for r in _requested_regimes if r not in SPLIT_REGIMES]
    if _unknown_regimes:
        raise ValueError(
            f"RB_SPLIT_REGIMES has unknown regime(s) {_unknown_regimes}; valid: {SPLIT_REGIMES}"
        )
    SPLIT_REGIMES = _requested_regimes

os.environ["STRICT_MODE"] = "1" if STRICT_MODE else ""


class _Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, text):
        for stream in self.streams:
            stream.write(text)
        return len(text)

    def flush(self):
        for stream in self.streams:
            stream.flush()


_LOG_HANDLE = None


def _tee_stdout_to_log() -> None:
    global _LOG_HANDLE
    if os.environ.get("RB_PARENT_LOG_CAPTURE"):
        return
    shard, nshards = gputils.shard_indices()
    name = f"main_shard_{shard}.log" if nshards > 1 else "main.log"
    log_dir = cacheutils.OUTPUT_DIR / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    _LOG_HANDLE = open(log_dir / name, "w", buffering=1)
    sys.stdout = _Tee(sys.stdout, _LOG_HANDLE)
    sys.stderr = sys.stdout
    print(f"[main] stdout/stderr -> {_LOG_HANDLE.name}", flush=True)


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
    strict_mode,
    enc_kwargs,
    write_predictions=True,
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
    unsupported = [r for r in split_regimes if r not in supported]
    if unsupported:
        raise ValueError(f"Unknown/unsupported split regimes for {benchmark_name}: {unsupported}. Supported: {supported}")
    split_regimes = [r for r in split_regimes if r in supported]

    bench_kwargs = dict(max_samples=max_samples, shuffle=True, seed=0)
    tag = cacheutils.bench_tag(bench_mod.BENCHMARK, bench_kwargs)
    perf.reset()
    bench = cacheutils.cached_bench(bench_mod.BENCHMARK, tag, **bench_kwargs)
    s2_only = os.environ.get("RB_S2_ONLY", "").strip().lower() not in ("", "0", "false", "no")
    suffix = "__s2only" if s2_only else ""
    emb_tag = tag + suffix
    y, _native_groups = bench_mod.make_targets(bench)
    bench_for_emb = bench.s2_only() if s2_only else bench
    if gen_embeddings:
        emb = cacheutils.extract_and_cache(
            bench_for_emb, bench_mod.BENCHMARK, model_name, emb_tag, overwrite=overwrite_mode, **enc_kwargs
        )
    else:
        emb = cacheutils.load_cached_embeddings(
            bench_for_emb, bench_mod.BENCHMARK, model_name, emb_tag, enc_kwargs.get("weights_path")
        )

    results_dir = cacheutils.OUTPUT_DIR / "results" / model_name / (benchmark_name + suffix)
    data_quality = getattr(bench, "data_quality", None)
    if not probing:
        if data_quality:
            IOU.write_json(results_dir / "data_quality.json", data_quality)
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
        write_predictions=write_predictions,
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
            results_dir / "data_quality.json",
            results_dir / "split_manifest.json",
            results_dir / "run_signature.txt",
            results_dir / artifacts.ENVIRONMENT_FILE,
            results_dir / artifacts.RUN_COMPLETE_FILE,
        ]:
            if p.exists():
                p.unlink()
    # ORDER MATTERS. Every check that can REFUSE this resume runs before anything is mutated:
    # check_run_signature above, then the environment gate here. Only once the resume is known to
    # be allowed do we invalidate the completion marker -- otherwise a refused resume would
    # destroy the valid marker of the finished run it just declined to touch.
    artifacts.write_environment(results_dir, overwrite_mode=overwrite_mode)
    runstate.publish_run_signature(results_dir, signature)
    # This pair is about to be made incomplete again, so any completion marker from a previous
    # run must not survive it: a stale marker asserts a finished state that is being undone.
    artifacts.invalidate_run_complete(results_dir)
    if data_quality:
        IOU.write_json(results_dir / "data_quality.json", data_quality)

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

    def _scopes(budget_type, b, source_diag=False):
        if budget_type == "target":
            return ("full", "held_out") if b == 0 else ("held_out",)
        return ("test", "source_validation", "source_test") if source_diag else ("test",)

    def _missing(base, budget_type, expected, source_diag=False):
        return [
            b for b in expected
            if not all((*base, budget_type, b, sc) in done for sc in _scopes(budget_type, b, source_diag))
        ]

    rerun_keys: set = set()
    split_specs: list[tuple] = []
    split_manifest: list[dict[str, Any]] = []
    # DOMAIN_CENSUS is a process-global accumulator; reset it per (model, benchmark) pair so a
    # pair's census artifact describes only that pair's data.
    regime_base.clear_domain_census()
    # REGIME_PROBLEMS accumulates across every pair in the shard, so slice from here to attribute
    # problems to THIS pair -- a pair must not be blocked by another pair's dropped regime, nor
    # excused by having been the first.
    regime_problems_before = len(regime_base.REGIME_PROBLEMS)
    cell_failures_before = len(perf.CELL_FAILURES)

    for seed in seeds:
        for split_regime in split_regimes:
            holdouts = regime_base.holdouts_for(bench_mod, split_regime)
            for split_label, train, val, test, groups, has_target, domain_basis, source_val, source_test in regime_base.iter_splits(
                split_regime,
                bench,
                y,
                holdouts,
                seed,
                strict_mode=strict_mode,
                val_group=regime_base.val_group_for(bench_mod, split_regime),
            ):
                split_specs.append(
                    (
                        seed, split_regime, split_label, train, val, test, groups, has_target,
                        domain_basis, source_val, source_test,
                    )
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
    EV._write_domain_census(results_dir, benchmark_name)

    # The complete set of cells this pair is supposed to produce, built from the config and the
    # REALIZED splits -- independently of whatever happens to be on disk. Compared against the
    # parsed rows at the end: a planned cell that produced no row (crashed, skipped, dropped
    # regime) otherwise leaves a table that reads as finished.
    expected_keys: set[tuple] = set()

    # ERM is the only execution path: ordinary probes on the frozen embeddings, no adaptation.
    # `mname` stays a literal so rows and resume keys keep the method="erm" column the canonical
    # artifacts and CELL_KEY_FIELDS depend on.
    mname = EV.ERM_METHOD
    method_meta = EV.erm_metadata()
    for seed in seeds:
        seed_split_specs = [spec for spec in split_specs if spec[0] == seed]
        for spec in seed_split_specs:
            _, split_regime, split_label, train, val, test, groups, has_target, domain_basis, source_val, source_test = spec
            for family in active_probes:
                meta = {
                    "model": model_name,
                    "benchmark": bench_mod.BENCHMARK,
                    "method": mname,
                    **method_meta,
                    "split_regime": split_regime,
                    "domain_basis": domain_basis,
                    "holdout": split_label,
                    "probe_family": family,
                }
                base = (seed, split_regime, split_label, mname, family)
                has_source_diag = len(source_val) > 0 and len(source_test) > 0
                # Same key shape as `done` / `rerun_keys`: every scope of every budget this
                # cell is planned to emit.
                if has_target:
                    expected_keys.update(
                        (*base, "target", b, sc)
                        for b in target_budgets for sc in _scopes("target", b)
                    )
                expected_keys.update(
                    (*base, "source", b, sc)
                    for b in source_budgets for sc in _scopes("source", b, has_source_diag)
                )
                if has_target:
                    todo = _missing(base, "target", target_budgets)
                    if todo:
                        rerun_keys.update((*base, "target", b, sc) for b in todo for sc in _scopes("target", b))
                        for budget in todo:
                            jobs.append(
                                delayed(runstate._probe_cell_target)(
                                    probe_fn_tgt,
                                    emb,
                                    train,
                                    val,
                                    test,
                                    y,
                                    groups,
                                    {**meta, "budget_type": "target"},
                                    seed,
                                    family,
                                    [budget],
                                    write_predictions=write_predictions,
                                )
                            )
                todo_src = _missing(base, "source", source_budgets, has_source_diag)
                if todo_src:
                    rerun_keys.update((*base, "source", b, sc) for b in todo_src for sc in _scopes("source", b, has_source_diag))
                    for budget in todo_src:
                        jobs.append(
                            delayed(runstate._probe_cell)(
                                probe_fn_src,
                                emb,
                                train,
                                val,
                                test,
                                y,
                                groups,
                                {**meta, "budget_type": "source"},
                                seed,
                                family,
                                [budget],
                                source_val,
                                source_test,
                                write_predictions=write_predictions,
                            )
                        )

    rows = runstate.prune_partial_budgets(rows, rows_path, preds_path if write_predictions else None, rerun_keys)

    if jobs:
        n_jobs = perf._effective_n_jobs(emb)
        print(f"  probe jobs={len(jobs)} n_jobs={n_jobs}", flush=True)
        for cell_rows, cell_preds in Parallel(n_jobs=n_jobs, return_as="generator", prefer="threads")(jobs):
            if write_predictions and cell_preds:
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
        # Only the binary per-sample bootstrap CI consumes predictions; the multiclass
        # predictions.jsonl is tens of GB and unused here, so never slurp it (OOM guard).
        predictions=IOU.read_jsonl(preds_path) if (write_predictions and bench_mod.LABEL_KIND == "binary") else None,
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
            "compatibility_rank": compat.rank(benchmark_name, model_name),
            "adaptation_severity": compat.adaptation_severity(benchmark_name, model_name),
            "s2_only_mode": s2_only,
            "declared_modalities": sorted(declared),
            "available_modalities": sorted(available),
            "effective_modalities": sorted(declared & available),
        },
    )

    perf_path = results_dir / "perf.jsonl"
    n_events = perf.write_log(perf_path)
    print(f"  perf: {n_events} events logged to {perf_path}", flush=True)

    # LAST write of the pair. Everything above -- probe_results.{jsonl,csv}, summary.csv,
    # deltas.csv -- is now final, which is exactly what this marker certifies: without it,
    # run_signature.txt only says the pair STARTED, and a pair killed mid-probe-loop leaves stale
    # derived CSVs beside a newer probe_results.jsonl with nothing to say they disagree.
    #
    # Validated against the PARSED rows on disk, not the in-memory list: the point is to certify
    # what a later reader will actually find. Raises IncompleteRunError on any shortfall, which
    # run_pair records as a pair failure -> non-zero shard exit.
    artifacts.write_run_complete(
        results_dir,
        signature=signature,
        expected_keys=expected_keys,
        # Absent file -> [] so write_run_complete reports it as a missing REQUIRED
        # artifact, rather than the caller dying here on FileNotFoundError.
        rows=artifacts.parse_jsonl_rows(rows_path) if rows_path.exists() else [],
        regime_problems=regime_base.REGIME_PROBLEMS[regime_problems_before:],
        cell_failures=perf.CELL_FAILURES[cell_failures_before:],
        extra={"model": model_name, "benchmark": benchmark_name},
    )


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
    strict_mode,
    write_predictions,
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
            strict_mode,
            write_predictions,
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
        strict_mode,
        enc_kwargs,
        write_predictions,
    )


def shard_exit_code(failures, regime_problems) -> int:
    """Non-zero if the shard's results table is incomplete for ANY reason.

    A dropped regime is not a lesser failure than a crashed pair -- it is a worse one, because the
    table it leaves behind looks finished. Every downstream consumer reads a dropped regime as
    "this regime simply wasn't evaluated" rather than "this run is unusable", and the exit code is
    what a launcher or collation step actually gates on; a banner in a multi-hour log is not.
    """
    return 1 if (failures or regime_problems) else 0


def main() -> int:
    if LAUNCH_GPU_SHARDS and gputils.SHARD_ENV not in os.environ:
        return gputils.fan_out(GPU_SHARDS)
    regime_base.clear_regime_problems()
    enc_kwargs = {"device": gputils.device()}

    all_pairs = [(mod, bm) for bm in BENCHMARKS for mod in compat.eligible_models(bm)]
    if RB_MODELS is not None:
        _eligible = {mod for mod, _ in all_pairs}
        _unknown = [m for m in RB_MODELS if m not in _eligible]
        if _unknown:
            raise ValueError(
                f"RB_MODELS has model(s) {_unknown} not eligible for "
                f"BENCHMARKS={BENCHMARKS}; eligible: {sorted(_eligible)}"
            )
        all_pairs = [(mod, bm) for mod, bm in all_pairs if mod in RB_MODELS]
        print(f"[main] RB_MODELS filter active -> {sorted(set(m for m, _ in all_pairs))}", flush=True)
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
                strict_mode=STRICT_MODE,
                write_predictions=WRITE_PREDICTIONS,
                enc_kwargs=enc_kwargs,
            )
        except NotImplementedError as exc:
            print(f"   [shard {shard}] {mod}/{bm} skipped (not implemented): {exc}", flush=True)
        except cacheutils.MissingEmbeddingCache:
            raise
        except Exception as exc:
            import traceback

            failures.append((mod, bm, f"{type(exc).__name__}: {exc}"))
            action = "raising" if STRICT_MODE else "continuing; re-run to resume"
            print(
                f"!! [shard {shard}] {mod}/{bm} FAILED: {type(exc).__name__}: {exc} ({action})",
                flush=True,
            )
            traceback.print_exc()
            if STRICT_MODE:
                raise

    regime_base.report_regime_problems()

    if failures:
        bar = "!" * 78
        print(f"\n{bar}\n[shard {shard}/{nshards}] {len(failures)} (model, benchmark) pair(s) FAILED:", flush=True)
        for mod, bm, reason in failures:
            print(f"  - {mod}/{bm}: {reason}", flush=True)
        print(f"{bar}", flush=True)
    if regime_base.REGIME_PROBLEMS:
        print(
            f"[shard {shard}/{nshards}] exiting non-zero: {len(regime_base.REGIME_PROBLEMS)} "
            f"declared regime(s) did not run (results are incomplete)",
            flush=True,
        )
    return shard_exit_code(failures, regime_base.REGIME_PROBLEMS)


if __name__ == "__main__":
    _tee_stdout_to_log()
    sys.exit(main())
