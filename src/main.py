"""Orchestrator for the frozen-embedding robustness pipeline.

Edit the config block below, then run:

    cd src && python main.py
"""

from __future__ import annotations

import os
import sys

for _thread_var in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"]:
    os.environ.setdefault(_thread_var, "1")

import numpy as np  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402

from evals import compat, split_artifacts  # noqa: E402
from evals import evals as EV  # noqa: E402
from evals import probes as _probes  # noqa: E402
from evals.regimes import base as regime_base  # noqa: E402
from utils import artifacts, cacheutils, gputils, runstate  # noqa: E402
from utils import ioutils as IOU  # noqa: E402
from utils import perfutils as perf  # noqa: E402

# === Configuration ===========================================================
# The final protocol lives here: committed, reviewable, and unchangeable by a stale shell or a
# Slurm export. The ONLY thing that differs across machines is which benchmarks/models this box
# runs -- edit BENCHMARKS / ACTIVE_MODELS per machine; everything else is synced identically.
BENCHMARKS = ["cropharvest", "eurocropsml", "breizhcrops", "pastis"]
ACTIVE_MODELS = None  # None = all eligible models for BENCHMARKS; or a list to restrict this box
SEEDS = [0, 1, 2]
ACTIVE_PROBES = ["logistic"]
SPLIT_REGIMES = ["random_id", "official", "geographic_ood", "spatial_cluster_ood"]
RUN_STAGES = ["gen_embeddings", "probing"]
PROBE_CAP = None       # None = uncapped source-head training; or an int cap (probe-capacity check)
PROBE_TUNING = False   # True = sweep the probe hyperparameter grid
S2_ONLY = False        # True = restrict every model to the shared Sentinel-2 input (fairness ablation)
BUDGET_REGIMES = {
    "source": [0.05, 0.10, 0.25, 1.0],
    "target": [0, 5, 10, 25, 50, EV.TARGET_ID_UPPER_BOUND],
}
MAX_DENSE_PIXELS = 50_000  # sampled pixels per PASTIS fold partition
OVERWRITE_MODE = False
STRICT_MODE = False
# Predictions are OPTIONAL and OFF by default. A full-grid PASTIS launch would otherwise stream billions
# of per-pixel JSON records; Stage 5 paired contrasts do not need them. Opt in explicitly per run.
WRITE_PREDICTIONS = False
LAUNCH_GPU_SHARDS = True
GPU_SHARDS = None
# =============================================================================

# The two knobs deep probing code reads are pushed into their modules from the committed constants
# above -- no env var can override them, and a worker subprocess (which re-imports this file) gets
# exactly these values.
perf.PROBE_CAP = PROBE_CAP
_probes.configure(tuning=PROBE_TUNING)

runstate.validate_run_stages(RUN_STAGES)  # fail fast on a bad config edit

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
    max_dense_pixels,
    split_regimes,
    run_stages,
    active_probes,
    budget_regimes,
    s2_only,
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
    probe_fn_la = {
        "binary": EV.run_probes_label_access,
        "multiclass": EV.run_probes_multiclass_label_access,
    }[bench_mod.LABEL_KIND]
    supported = getattr(bench_mod, "SPLIT_REGIMES", split_regimes)
    unsupported = [r for r in split_regimes if r not in supported]
    if unsupported:
        raise ValueError(f"Unknown/unsupported split regimes for {benchmark_name}: {unsupported}. Supported: {supported}")
    split_regimes = [r for r in split_regimes if r in supported]

    perf.reset()
    bench = cacheutils.cached_bench(bench_mod.BENCHMARK)
    suffix = "__s2only" if s2_only else ""
    artifact = cacheutils.artifact_name(s2_only)
    y, _native_groups = bench_mod.make_targets(bench)
    results_dir = cacheutils.OUTPUT_DIR / "results" / model_name / (benchmark_name + suffix)

    # PHASE B -- FAIL-FAST SPLIT VALIDATION. When probing is requested, consume and structurally
    # validate every requested split from the single canonical data/splits/ location (committed
    # config; no machine-specific root) immediately after loading the benchmark and validating regime
    # compatibility -- BEFORE any embedding extraction, cache load, embedding digest, or
    # results-directory mutation. An invalid/missing split therefore refuses the pair WITHOUT running
    # the encoder or touching existing results (byte-for-byte, even under OVERWRITE_MODE=True).
    # Partitions AND per-sample domains come entirely from the artifacts. Embedding-only runs
    # (probing disabled) never require split artifacts.
    split_specs: list[split_artifacts.LoadedTabularSplit] = []
    label_access_by_cell: dict[tuple[int, str], split_artifacts.LoadedLabelAccess] = {}
    if probing:
        splits_root = cacheutils.SCRATCH / "splits"
        split_specs = split_artifacts.load_tabular_splits(
            splits_root, bench_mod.BENCHMARK, bench.sample_ids, split_regimes, seeds
        )
        # Fail-fast (same discipline as the split load, BEFORE any embedding work): load + structurally
        # validate the frozen label-access order for every geographic_ood headline target, resolving its
        # ranked stable ids to current row indices. The id-map is built lazily, only when such a target
        # is actually present (so non-geographic runs never depend on it).
        _la_id_map: dict[str, int] | None = None
        for _ls in split_specs:
            if _ls.regime == split_artifacts.LABEL_ACCESS_REGIME and _ls.split.supports_target_labels:
                if _la_id_map is None:
                    _la_id_map = {str(s): i for i, s in enumerate(np.asarray(bench.sample_ids).tolist())}
                label_access_by_cell[(_ls.seed, _ls.split.label)] = split_artifacts.load_label_access(
                    splits_root, bench_mod.BENCHMARK, _ls.seed, _ls.split, _la_id_map
                )

    bench_for_emb = bench.s2_only() if s2_only else bench
    if gen_embeddings:
        emb = cacheutils.extract_and_cache(
            bench_for_emb, bench_mod.BENCHMARK, model_name, artifact, **enc_kwargs
        )
    else:
        emb = cacheutils.load_cached_embeddings(
            bench_for_emb, bench_mod.BENCHMARK, model_name, artifact, enc_kwargs.get("weights_path")
        )

    data_quality = getattr(bench, "data_quality", None)
    if not probing:
        if data_quality:
            IOU.write_json(results_dir / "data_quality.json", data_quality)
        n_events = perf.write_log(results_dir / "perf.jsonl")
        print(f"  embedding stage complete; perf: {n_events} events logged", flush=True)
        return
    emb_digest = cacheutils.embedding_digest(bench_mod.BENCHMARK, model_name, artifact)
    rows_path = results_dir / "probe_results.jsonl"
    preds_path = results_dir / "predictions.jsonl"

    # split_specs were loaded + structurally validated (checksum + complete accounting) ABOVE, before
    # any embedding work AND before any mutation (validate-before-mutation still holds: a refused split
    # here leaves existing rows, environment.json, run_manifest.json, and run_complete.json untouched).
    manifest = runstate.build_run_manifest(
        model_name, benchmark_name, artifact, emb_digest, split_regimes, seeds, enc_kwargs,
        active_probes=active_probes, budget_regimes=budget_regimes, max_dense_pixels=max_dense_pixels,
        write_predictions=write_predictions,
    )
    signature = runstate.run_manifest_digest(manifest)
    runstate.check_run_manifest(results_dir, manifest, overwrite_mode=overwrite_mode)

    if overwrite_mode:
        for p in [
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
            results_dir / "domain_census.json",      # legacy per-model artifact -- retired, remove on resume
            results_dir / artifacts.RUN_MANIFEST_FILE,
            results_dir / artifacts.ENVIRONMENT_FILE,
            results_dir / artifacts.RUN_COMPLETE_FILE,
        ]:
            if p.exists():
                p.unlink()
    # ORDER MATTERS. Every check that can REFUSE this resume already ran above (split loading, then
    # check_run_manifest). Only now, once the resume is known to be allowed, do we mutate.
    artifacts.write_environment(results_dir, overwrite_mode=overwrite_mode)
    runstate.publish_run_manifest(results_dir, manifest)
    # This pair is about to be made incomplete again, so any completion marker from a previous run
    # must not survive it.
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
            r.get("label_access_route", ""),
        )
        for r in rows
    }

    jobs = []

    def _scopes(budget_type, b, source_diag=False):
        if budget_type == "target":
            return ("full", "held_out") if b == 0 else ("held_out",)
        # source_test is the untouched within-source reference, evaluated alongside the primary eval
        # (target_test for OOD) when the 80/10/10 split carries a source_test partition.
        return ("test", "source_test") if source_diag else ("test",)

    def _missing(base, budget_type, expected, source_diag=False):
        return [
            b for b in expected
            if not all((*base, budget_type, b, sc, "") in done for sc in _scopes(budget_type, b, source_diag))
        ]

    rerun_keys: set = set()
    # split_specs were loaded + structurally validated ABOVE, before any mutation. Nothing appends to
    # REGIME_PROBLEMS here -- split validity is a generation-time concern -- but `regime_problems_before`
    # is retained for the completeness marker.
    regime_problems_before = len(regime_base.REGIME_PROBLEMS)
    cell_failures_before = len(perf.CELL_FAILURES)

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
        seed_split_specs = [ls for ls in split_specs if ls.seed == seed]
        for loaded in seed_split_specs:
            split_regime = loaded.regime
            st = loaded.split                 # SourceTargetSplit whose arrays are CURRENT row indices
            groups = loaded.domains           # per-sample domain basis (worst-group scoring)
            split_label, has_target, domain_basis = st.label, st.has_target, st.group_kind
            supports_target_labels = st.supports_target_labels
            # Schema-v2 partition routing, by route capability:
            #  * random_id (has_target=False): evaluate IN DISTRIBUTION on source_test.
            #  * official (has_target=True, supports_target_labels=False): fit source_train, calibrate
            #    source_val, evaluate ZERO-SHOT on target_test -- NO target-label access, NO target sweep.
            #  * geographic/spatial (supports_target_labels=True): the source budgets ALSO evaluate
            #    zero-shot on target_test, AND the target budgets draw few-shot labels ONLY from
            #    target_label_pool and are scored on the SAME target_test.
            train, val = st.source_train, st.source_val
            test = st.target_test if has_target else st.source_test
            # The 80/10/10 within-source reference: source_test is evaluated as an UNTOUCHED diagnostic
            # (never trained or tuned on) whenever the split carries one -- geographic/spatial. official's
            # 90/10 pool has no source_test; random_id's source_test IS its primary eval.
            source_test_ref = st.source_test if (has_target and len(st.source_test) > 0) else np.empty(0, dtype=np.int64)
            for family in active_probes:
                meta = {
                    "model": model_name,
                    "benchmark": bench_mod.BENCHMARK,
                    "method": mname,
                    **method_meta,
                    "split_regime": split_regime,
                    "domain_basis": domain_basis,
                    "holdout": split_label,
                    "target_role": st.target_role,
                    "probe_family": family,
                    "label_access_route": "",   # non-label-access default; label-access rows override
                }
                base = (seed, split_regime, split_label, mname, family)
                has_source_diag = len(source_test_ref) > 0
                # Same key shape as `done` / `rerun_keys`: every scope of every budget this
                # cell is planned to emit. Target-budget sweeps run ONLY when the regime supports
                # target labels; official (has_target but supports_target_labels=False) is zero-shot.
                is_label_access = supports_target_labels and split_regime == split_artifacts.LABEL_ACCESS_REGIME
                # The source-budget sweep runs for EVERY regime (unchanged experiment; empty route id).
                expected_keys.update(
                    (*base, "source", b, sc, "")
                    for b in source_budgets for sc in _scopes("source", b, has_source_diag)
                )
                if is_label_access:
                    # geographic_ood headline: the 13-route label-access suite REPLACES the old target-
                    # budget sweep (no duplicate legacy target rows). ONE job per cell so source_only is
                    # fit exactly once and its fitted scorer is reused for the complete-target diagnostic;
                    # resume is at cell granularity so different routes can never be confused.
                    la = label_access_by_cell[(seed, split_label)]
                    cell_keys = [
                        (*base, "label_access", b, es, route)
                        for (route, b, es) in split_artifacts.label_access_expected_rows()
                    ]
                    expected_keys.update(cell_keys)
                    if not all(k in done for k in cell_keys):
                        rerun_keys.update(cell_keys)
                        jobs.append(
                            delayed(runstate._probe_cell_label_access)(
                                probe_fn_la, emb, train, val, st.target_label_pool, st.target_test,
                                la.matched_source_ranked_idx, la.fixed_source_removal_ranked_idx,
                                la.target_ranked_idx, y, groups,
                                {**meta, "budget_type": "label_access"}, seed, family,
                                write_predictions=write_predictions,
                            )
                        )
                elif supports_target_labels:
                    # non-geographic target-label regime (spatial_cluster_ood): the legacy target sweep.
                    expected_keys.update(
                        (*base, "target", b, sc, "")
                        for b in target_budgets for sc in _scopes("target", b)
                    )
                    todo = _missing(base, "target", target_budgets)
                    if todo:
                        rerun_keys.update((*base, "target", b, sc, "") for b in todo for sc in _scopes("target", b))
                        for budget in todo:
                            jobs.append(
                                delayed(runstate._probe_cell_target)(
                                    probe_fn_tgt, emb, train, val, st.target_label_pool, st.target_test,
                                    y, groups, {**meta, "budget_type": "target"}, seed, family, [budget],
                                    write_predictions=write_predictions,
                                )
                            )
                todo_src = _missing(base, "source", source_budgets, has_source_diag)
                if todo_src:
                    rerun_keys.update((*base, "source", b, sc, "") for b in todo_src for sc in _scopes("source", b, has_source_diag))
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
                                None,               # no source_validation diagnostic (source_val is the calibration set)
                                source_test_ref,    # the untouched within-source reference (source_test scope)
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
            "label_access_route",
        ],
        metrics=metrics,
        # Supervision sizes vary by holdout (regional pool sizes differ), so aggregate them rather than
        # key on them -- otherwise every region would be its own summary row. label_budget_unit is
        # constant within a group and preserved verbatim.
        count_aggregates=["n_source_labels", "n_target_labels", "n_total_labels"],
        passthrough=["label_budget_unit"],
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

    # Stage 5: paired label-access contrasts (pure post-processing on the rows). No-op unless the run
    # carries the geographic_ood label-access suite. Hard-fails here on a missing/duplicate operand or an
    # unresolvable source_ID_reference anchor; write_run_complete re-validates + hashes the artifacts.
    from evals import contrasts

    contrasts.compute_and_write(results_dir, rows)

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
    # run_manifest.json only says the pair STARTED, and a pair killed mid-probe-loop leaves stale
    # derived CSVs beside a newer probe_results.jsonl with nothing to say they disagree.
    #
    # Validated against the PARSED rows on disk, not the in-memory list: the point is to certify
    # what a later reader will actually find. Raises IncompleteRunError on any shortfall, which
    # run_pair records as a pair failure -> non-zero shard exit.
    artifacts.write_run_complete(
        results_dir,
        run_manifest_sha256=signature,
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
    """Run one configured model/benchmark pair."""
    bench_mod = EV.load_benchmark(benchmark_name)
    if bench_mod.LABEL_KIND == "segmentation":
        runstate._run_segmentation_pair(
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
        )
        return
    _run_tabular_pair(
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
    if cacheutils.frozen_run_identity()["final_commit"] is None:
        raise RuntimeError(
            "the deployed checkout has no .git commit -- the frozen run requires committed "
            "provenance recorded in run_manifest.json. Deploy a real git checkout."
        )
    regime_base.clear_regime_problems()
    enc_kwargs = {"device": gputils.device()}

    all_pairs = [(mod, bm) for bm in BENCHMARKS for mod in compat.eligible_models(bm)]
    if ACTIVE_MODELS is not None:
        _eligible = {mod for mod, _ in all_pairs}
        _unknown = [m for m in ACTIVE_MODELS if m not in _eligible]
        if _unknown:
            raise ValueError(
                f"ACTIVE_MODELS has model(s) {_unknown} not eligible for "
                f"BENCHMARKS={BENCHMARKS}; eligible: {sorted(_eligible)}"
            )
        all_pairs = [(mod, bm) for mod, bm in all_pairs if mod in ACTIVE_MODELS]
        print(f"[main] ACTIVE_MODELS filter active -> {sorted(set(m for m, _ in all_pairs))}", flush=True)
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
                max_dense_pixels=MAX_DENSE_PIXELS,
                split_regimes=SPLIT_REGIMES,
                run_stages=RUN_STAGES,
                active_probes=ACTIVE_PROBES,
                budget_regimes=BUDGET_REGIMES,
                s2_only=S2_ONLY,
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
