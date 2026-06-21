"""Orchestrator for the frozen-embedding robustness pipeline.

    benchmark spec -> get_input -> encode (frozen) -> cache
                  -> {ERM, ACTIVE_METHODS} -> probe -> tables

Edit the config block below, then::

    cd src && python main.py                 # single process
    cd src && python utils/gputils.py        # split (model, benchmark) work across all GPUs

Everything is resumable: assembled benchmarks are pickle-cached, model embeddings
are cached (atomic writes), and probe results are appended to a
JSON-lines log as each cell completes. Re-running skips all finished work, so a
crash only loses the in-flight cell.
"""

from __future__ import annotations

import importlib
import os
from typing import Any

import numpy as np
from joblib import Parallel, delayed

from evals import compat
from evals import evals as EV
from utils import (
    cacheutils,
    gputils,
)
from utils import ioutils as IOU
from utils import (
    perfutils as perf,
)

# === Configuration ===========================================================
BENCHMARKS = ["cropharvest", "eurocropsml", "breizhcrops", "pastis_r"]
ACTIVE_METHODS = []
RUN_STAGES = ["gen_embeddings", "probing"]
# Master list of regimes this run wants. The effective set per benchmark is the
# intersection with that benchmark's own `SPLIT_REGIMES` allowlist (e.g. temporal_ood
# needs multiple years, climate_ood needs coordinates, phenology_ood is binary-only).
# geographic_ood = leave-one-region-out; climate_ood = leave-one-Köppen-zone-out;
# temporal_ood = forward (train past / test latest year).
SPLIT_REGIMES = ["random_id", "geographic_ood", "climate_ood", "temporal_ood", "phenology_ood"]
# Probe-capacity ablation axis. "logistic" is the primary protocol; "mlp"/"knn" test
# whether a geographic gap is caused by the linear probe (vanishes under mlp) or the
# encoder (persists). Each family gets the SAME-size val-selected grid and class-balanced
# training (see evals/probes.py), so the comparison is fair. mlp/knn are the slow part of
# the run; if it's too long, drop them here (this is the one knob to scope it).
ACTIVE_PROBES = ["logistic", "mlp", "knn"]

MAX_SAMPLES = None  # benchmark samples per benchmark (None = all)
MAX_DENSE_PIXELS = 50_000  # sampled pixels per PASTIS fold partition
N_JOBS = -1  # cores for the (single-threaded sklearn) probe sweep; -1 = all
OVERWRITE_MODE = "skip"  # choose between skip, override
SEEDS = [0, 1, 2]  # random seeds for probe reproducibility (3 = credibility floor for the gap CIs)
# =============================================================================

BENCH_SHUFFLE_SEED = 0  # fixed so cached embeddings stay aligned to the bench row order

# LABEL_KIND -> (probe runner, metric list)
DISPATCH = {
    "binary": (EV.run_probes, EV.METRICS_BINARY),
    "multiclass": (EV.run_probes_multiclass, EV.METRICS_MULTICLASS),
}
DISPATCH_TARGET = {
    "binary": (EV.run_probes_target, EV.METRICS_BINARY),
    "multiclass": (EV.run_probes_multiclass_target, EV.METRICS_MULTICLASS),
}
_VALID_RUN_STAGES = {"gen_embeddings", "probing"}


def load_benchmark(benchmark_name: str):
    return importlib.import_module(f"evals.benchmarks.{benchmark_name}")


def load_regime(regime_name: str):
    """Import a split-regime module (evals/regimes/<regime_name>.py).

    Each regime owns both its domain assignment and splitting. The module exposes
    ``GROUP_KIND``, ``assign_domains(bench)``, ``HAS_TARGET``, and
    ``iter_splits(y, domains, *, seed, holdouts, n_folds)``.
    """
    return importlib.import_module(f"evals.regimes.{regime_name}")


def _run_stage_set(run_stages: list[str]) -> set[str]:
    """Validate the configured run stages."""
    stages = set(run_stages)
    unknown = stages - _VALID_RUN_STAGES
    if unknown:
        raise ValueError(f"Unknown RUN_STAGES entries: {sorted(unknown)}. Valid entries: {sorted(_VALID_RUN_STAGES)}")
    if not stages:
        raise ValueError("RUN_STAGES must include at least one stage.")
    return stages


def _effective_n_jobs(embeddings: dict[str, np.ndarray]) -> int:
    """Bound concurrent probe fits when embedding matrices are large."""
    cpu_count = os.cpu_count() or 1
    requested = N_JOBS
    if requested < 0:
        requested = max(1, cpu_count + 1 + requested)
    elif requested == 0:
        requested = 1

    # The probe sweep runs with prefer="threads" (see Parallel(...) below), so the embedding
    # matrix is SHARED across jobs, not copied per job. Only each cell's working set (the
    # subset view + scaler + fitted probe) grows with thread count, so the old per-process
    # caps (1 job >2GB, 2 jobs >200MB) were far too conservative -- e.g. they throttled
    # EuroCropsML's ~360MB embeddings to 2 cores. Only throttle the genuinely huge matrices.
    max_bytes = max(np.asarray(arr).nbytes for arr in embeddings.values())
    if max_bytes >= 4_000_000_000:
        return min(requested, 4)
    if max_bytes >= 1_000_000_000:
        return min(requested, 8)
    return requested


# --------------------------------------------------------------------------- #
# Methods
# --------------------------------------------------------------------------- #


def build_methods(label_kind: str, seed: int):
    """name -> (cls_or_none, kwargs).  ERM (no transform) is always included."""
    methods: dict[str, tuple[Any, dict]] = {"erm": (None, {})}
    for name in ACTIVE_METHODS:
        mod = importlib.import_module(f"methods.{name}")
        for vname, base_kwargs in mod.variants(label_kind).items():
            methods[vname] = (getattr(mod, name.title()), {**base_kwargs, "seed": seed})
    return methods


# --------------------------------------------------------------------------- #
# One probe cell (runs in a worker process; embeddings are memmapped by joblib)
# --------------------------------------------------------------------------- #


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
) -> tuple[list[dict], list[dict]]:
    """Fit the (optional) method transform and run the source-budget-swept probe.

    Self-contained and picklable so joblib can run many cells across cores. Returns
    ``(rows, per_sample_predictions)``. ``val`` is the regime's held-out validation
    set, forwarded to the binary probe for threshold calibration.
    """
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
) -> tuple[list[dict], list[dict]]:
    """Target-budget variant: passes the *full* target pool so the probe runner
    can sample N target labels for few-shot training and use the rest for testing.

    ``val`` is the regime's source-side held-out validation set, forwarded to the
    binary probe for threshold calibration (no peeking at the target region).
    """
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
        )
    perf.set_identity(None)
    return rows, preds


# --------------------------------------------------------------------------- #
# Run
# --------------------------------------------------------------------------- #


# A benchmark lists a regime in SPLIT_REGIMES to assert it SHOULD run. If that regime then
# yields nothing — missing external data (no Köppen grid), a single-domain/degenerate input
# (one year -> no temporal split), absent coordinates — the result is a silent hole. We refuse
# to let that pass quietly: every such case is shouted at the call site, collected here, and
# re-listed at the end of the run. Set STRICT_REGIMES=1 to turn it into a hard failure instead.
STRICT_REGIMES = os.environ.get("STRICT_REGIMES", "0").lower() not in ("0", "", "false", "no")
_REGIME_PROBLEMS: list[tuple[str, str, str]] = []


def _regime_problem(benchmark: str, regime: str, reason: str) -> None:
    """Surface a declared regime that did not run — loudly, never silently."""
    _REGIME_PROBLEMS.append((benchmark, regime, reason))
    if STRICT_REGIMES:
        raise RuntimeError(f"[STRICT_REGIMES] declared regime did not run — {benchmark}/{regime}: {reason}")
    bar = "!" * 78
    print(
        f"\n{bar}\n!! REGIME DECLARED BUT DID NOT RUN — {benchmark}/{regime}\n!! {reason}"
        f"\n!! (it is in this benchmark's SPLIT_REGIMES; set STRICT_REGIMES=1 to make this fatal)\n{bar}\n",
        flush=True,
    )


def _report_regime_problems() -> None:
    """Print a consolidated list of regimes that were declared but did not run."""
    if not _REGIME_PROBLEMS:
        return
    bar = "=" * 78
    print(f"\n{bar}\nREGIMES DECLARED BUT NOT RUN ({len(_REGIME_PROBLEMS)}):", flush=True)
    for benchmark, regime, reason in _REGIME_PROBLEMS:
        print(f"  - {benchmark}/{regime}: {reason}", flush=True)
    print(f"{bar}\n", flush=True)


def _iter_splits(split_regime, bench, y, holdouts, seed):
    """Yield split metadata and regime-assigned domain labels.

    A regime owns the domain basis (for example geography or phenology) and the
    train/test splitting rule over those domains. A declared regime that assigns no
    domains, or produces no splits, is reported via :func:`_regime_problem` rather than
    skipped silently.
    """
    regime = load_regime(split_regime)
    bench_name = getattr(bench, "name", "?")
    try:
        # External-data domains (e.g. climate via a Köppen grid) raise when the data or
        # coordinates are absent. That is a declared regime failing to run -> surface it.
        domains = np.asarray(regime.assign_domains(bench), dtype=object)
    except Exception as exc:
        _regime_problem(bench_name, split_regime, f"domain assignment failed ({type(exc).__name__}: {exc})")
        return
    if len(domains) != len(y):
        raise ValueError(
            f"{split_regime}.assign_domains returned {len(domains)} domains for {len(y)} labels"
        )
    n_unknown = int(np.isin(domains.astype(str), ("unknown", "nan")).sum())
    if n_unknown:
        print(
            f"   [{bench_name}/{split_regime}] {n_unknown}/{len(domains)} samples have no domain "
            f"(unknown/nan coords) and are excluded from this regime's holdouts",
            flush=True,
        )
    n_splits = 0
    for split in regime.iter_splits(y, domains, seed=seed, holdouts=holdouts):
        n_splits += 1
        yield split.label, split.train, split.val, split.test, domains, regime.HAS_TARGET, regime.GROUP_KIND
    if n_splits == 0:
        labels = sorted({str(d) for d in domains})
        shown = labels[:8] + (["…"] if len(labels) > 8 else [])
        _regime_problem(
            bench_name, split_regime, f"produced 0 splits (domain labels seen: {shown})"
        )


def _segmentation_fold_configs(bench_mod, regimes):
    """Yield (split_regime, holdout_label, train_folds, val_folds, test_folds) for the dense path.

    Each regime owns its own fold logic in ``evals/regimes/<regime>.py`` via
    ``iter_fold_splits(bench_mod)`` — ``random_id`` yields the published 1-3 / 4 / 5
    in-distribution assignment, ``geographic_ood`` yields leave-one-fold-out. This just
    dispatches over the benchmark's ``SPLIT_REGIMES`` allowlist and skips any regime with no
    dense (fold-based) realization (e.g. climate/temporal/phenology, which PASTIS — a single
    region and season — cannot express).
    """
    for regime_name in regimes:
        fold_iter = getattr(load_regime(regime_name), "iter_fold_splits", None)
        if fold_iter is None:
            _regime_problem(
                getattr(bench_mod, "BENCHMARK", "?"),
                regime_name,
                "no dense (segmentation) realization — regime exposes no iter_fold_splits",
            )
            continue
        for label, train_folds, val_folds, test_folds in fold_iter(bench_mod):
            yield (regime_name, label, train_folds, val_folds, test_folds)


def run_segmentation(benchmark_name, model_name, seeds, max_samples, run_stages, **enc_kwargs) -> None:
    """Run PASTIS-R segmentation over its fold-based geographic regimes.

    PASTIS is dense (per-pixel) and too large to hold in memory, so it streams tiles
    by fold from disk rather than indexing a flat array — hence a separate runner from
    the classification ``run``. Its domain is the spatial fold, and it executes the
    regimes in the benchmark's ``SPLIT_REGIMES`` allowlist via
    :func:`_segmentation_fold_configs`: ``random_id`` (published 1-3/4/5, the
    in-distribution baseline) and ``geographic_ood`` (leave-one-fold-out, the deployment regime).
    Rows carry ``domain_basis="geography"`` so the schema matches the classification
    benchmarks. (Phenology and the few-shot/oracle budget sweep are intentionally NOT
    run here — see the experimental-design notes: phenology is label-confounded on a
    single-region/year dataset, and the target-label unit for dense segmentation is a
    design question, not a mechanism.)
    """
    stages = _run_stage_set(run_stages)
    gen_embeddings = "gen_embeddings" in stages
    probing = "probing" in stages
    bench_mod = load_benchmark(benchmark_name)
    bench_kwargs = dict(max_samples=max_samples, shuffle=True, seed=BENCH_SHUFFLE_SEED)
    tag = cacheutils.bench_tag(bench_mod.BENCHMARK, bench_kwargs)
    perf.reset()
    bench = cacheutils.cached_bench(bench_mod.BENCHMARK, tag, **bench_kwargs)
    if gen_embeddings:
        emb_dir = cacheutils.extract_dense_and_cache(
            bench,
            bench_mod.BENCHMARK,
            model_name,
            tag,
            overwrite=OVERWRITE_MODE,
            **enc_kwargs,
        )
    else:
        emb_dir = cacheutils.require_dense_cache(bench, bench_mod.BENCHMARK, model_name, tag)
    results_dir = cacheutils.OUTPUT_DIR / "results" / model_name / benchmark_name
    if not probing:
        n_events = perf.write_log(results_dir / "perf.jsonl")
        print(f"  embedding stage complete; perf: {n_events} events logged", flush=True)
        return
    rows_path = results_dir / "probe_results.jsonl"
    if OVERWRITE_MODE == "override":
        for path in (rows_path, results_dir / "probe_results.csv", results_dir / "summary.csv"):
            if path.exists():
                path.unlink()
    rows = IOU.read_jsonl(rows_path)
    done = {
        (r.get("seed"), r.get("method"), r.get("split_regime"), r.get("holdout"), r.get("probe_family"))
        for r in rows
    }
    regimes = getattr(bench_mod, "SPLIT_REGIMES", ["random_id"])
    fold_configs = list(_segmentation_fold_configs(bench_mod, regimes))
    for seed in seeds:
        for method_name, (cls, kwargs) in build_methods(bench_mod.LABEL_KIND, seed).items():
            for split_regime, holdout, train_folds, val_folds, test_folds in fold_configs:
                families_to_run = [
                    f for f in ACTIVE_PROBES
                    if (seed, method_name, split_regime, holdout, f) not in done
                ]
                if not families_to_run:
                    continue
                x_train, y_train, groups_train, _ = cacheutils.load_dense_samples(
                    emb_dir, train_folds, MAX_DENSE_PIXELS, seed
                )
                x_val, y_val, _, _ = cacheutils.load_dense_samples(
                    emb_dir, val_folds, MAX_DENSE_PIXELS, seed + 10_000
                )
                x_test, y_test, _, tile_ids_test = cacheutils.load_dense_samples(
                    emb_dir, test_folds, MAX_DENSE_PIXELS, seed + 20_000
                )
                transform = None
                if cls is not None:
                    transform = cls(**kwargs)
                    paired = x_test if getattr(cls, "USES_TARGET", False) else None
                    transform.fit(x_train, y_train, groups_train, x_paired=paired)
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
                    # Source-fraction sweep (always).
                    EV.run_probes_segmentation(
                        cell_rows,
                        x_train,
                        x_val,
                        x_test,
                        y_train,
                        y_val,
                        y_test,
                        seed,
                        transform=transform,
                        meta=seg_meta,
                        tile_ids_test=tile_ids_test,
                        family=family,
                    )
                    # Tile-level target-budget sweep (zero-shot / few-shot / oracle) only on
                    # the deployment regime; random_id stays the source-only in-distribution split.
                    if split_regime == "geographic_ood":
                        EV.run_probes_segmentation_target(
                            cell_rows,
                            x_train,
                            y_train,
                            x_test,
                            y_test,
                            tile_ids_test,
                            x_val,
                            y_val,
                            seed,
                            transform=transform,
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
    perf.write_log(results_dir / "perf.jsonl")


def run(benchmark_name, model_name, seeds, max_samples, split_regimes, run_stages, **enc_kwargs) -> None:
    stages = _run_stage_set(run_stages)
    gen_embeddings = "gen_embeddings" in stages
    probing = "probing" in stages
    bench_mod = load_benchmark(benchmark_name)
    if bench_mod.LABEL_KIND == "segmentation":
        run_segmentation(benchmark_name, model_name, seeds, max_samples, run_stages, **enc_kwargs)
        return
    probe_fn_src, metrics = DISPATCH[bench_mod.LABEL_KIND]
    probe_fn_tgt, _ = DISPATCH_TARGET[bench_mod.LABEL_KIND]
    holdouts = bench_mod.HOLDOUTS

    # Effective regimes = run's master list ∩ this benchmark's supported regimes.
    # A benchmark that omits a regime never runs it.
    # never runs it, even if it is in the master list.
    supported = getattr(bench_mod, "SPLIT_REGIMES", split_regimes)
    split_regimes = [r for r in split_regimes if r in supported]

    bench_kwargs = dict(max_samples=max_samples, shuffle=True, seed=BENCH_SHUFFLE_SEED)
    tag = cacheutils.bench_tag(bench_mod.BENCHMARK, bench_kwargs)
    perf.reset()
    bench = cacheutils.cached_bench(bench_mod.BENCHMARK, tag, **bench_kwargs)
    y, _native_groups = bench_mod.make_targets(bench)
    if gen_embeddings:
        emb = cacheutils.extract_and_cache(
            bench, bench_mod.BENCHMARK, model_name, tag, overwrite=OVERWRITE_MODE, **enc_kwargs
        )
    else:
        emb = cacheutils.load_cached_embeddings(bench, bench_mod.BENCHMARK, model_name, tag)

    results_dir = cacheutils.OUTPUT_DIR / "results" / model_name / benchmark_name
    if not probing:
        n_events = perf.write_log(results_dir / "perf.jsonl")
        print(f"  embedding stage complete; perf: {n_events} events logged", flush=True)
        return
    rows_path = results_dir / "probe_results.jsonl"
    preds_path = results_dir / "predictions.jsonl"

    if OVERWRITE_MODE == "override":
        for p in [
            rows_path,
            preds_path,
            results_dir / "probe_results.csv",
            results_dir / "summary.csv",
            results_dir / "deltas.csv",
        ]:
            if p.exists():
                p.unlink()

    rows = IOU.read_jsonl(rows_path)
    done = {
        (
            r.get("seed"),
            r.get("split_regime"),
            r.get("holdout"),
            r.get("method"),
            r.get("probe_family"),
            r.get("budget_type"),
        )
        for r in rows
    }

    jobs = []

    def uses_target_flag(cls):
        return getattr(cls, "USES_TARGET", False)

    for seed in seeds:
        for mname, (cls, kwargs) in build_methods(bench_mod.LABEL_KIND, seed).items():
            for split_regime in split_regimes:
                for split_label, train, val, test, groups, has_target, domain_basis in _iter_splits(
                    split_regime, bench, y, holdouts, seed
                ):
                    for family in ACTIVE_PROBES:
                        meta = {
                            "model": model_name,
                            "benchmark": bench_mod.BENCHMARK,
                            "method": mname,
                            "split_regime": split_regime,
                            "domain_basis": domain_basis,
                            "holdout": split_label,
                            "probe_family": family,
                        }
                        # ---- Target-budget sweep ----
                        if (
                            has_target
                            and (seed, split_regime, split_label, mname, family, "target") not in done
                        ):
                            jobs.append(
                                delayed(_probe_cell_target)(
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
                                )
                            )
                        # ---- Source-fraction sweep ----
                        if (seed, split_regime, split_label, mname, family, "source") not in done:
                            jobs.append(
                                delayed(_probe_cell)(
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
                                )
                            )

    if jobs:
        n_jobs = _effective_n_jobs(emb)
        print(f"  probe jobs={len(jobs)} n_jobs={n_jobs}", flush=True)
        for cell_rows, cell_preds in Parallel(n_jobs=n_jobs, return_as="generator", prefer="threads")(jobs):
            IOU.append_jsonl(rows_path, cell_rows)
            rows.extend(cell_rows)
            if cell_preds:
                IOU.append_jsonl(preds_path, cell_preds)

    IOU.write_csv(results_dir / "probe_results.csv", rows)
    summary = IOU.summarize_rows(
        rows,
        keys=["model", "method", "probe_family", "split_regime", "domain_basis", "budget_type", "label_budget"],
        metrics=metrics,
    )
    IOU.write_csv(results_dir / "summary.csv", summary)
    IOU.write_json(results_dir / "metric_roles.json", EV.METRIC_ROLES[bench_mod.LABEL_KIND])

    # Delta table: ID (random_id, source=1.0) minus OOD (geographic_ood, target=0),
    # inherent-difficulty decomposition (when target=-1 rows exist),
    # and worst-region metrics.
    deltas = IOU.compute_deltas(rows, metrics, predictions=IOU.read_jsonl(preds_path))
    IOU.write_csv(results_dir / "deltas.csv", deltas)

    perf_path = results_dir / "perf.jsonl"
    n_events = perf.write_log(perf_path)
    print(f"  perf: {n_events} events logged to {perf_path}", flush=True)


if __name__ == "__main__":
    enc_kwargs = {"device": gputils.device()}

    # The compatibility matrix (evals/compat.py) decides which models run on each
    # benchmark; we only specify BENCHMARKS. Each pair is one unit of work.
    all_pairs = [(mod, bm) for bm in BENCHMARKS for mod in compat.eligible_models(bm)]
    work = gputils.take_shard(all_pairs)
    shard, nshards = gputils.shard_indices()
    for mod, bm in work:
        print(f"\n========== [shard {shard}/{nshards}] {mod} / {bm} ==========", flush=True)
        print(f"  split_regimes={SPLIT_REGIMES}", flush=True)
        print(f"  run_stages={RUN_STAGES}", flush=True)
        try:
            run(bm, mod, SEEDS, MAX_SAMPLES, SPLIT_REGIMES, RUN_STAGES, **enc_kwargs)
        except NotImplementedError as exc:
            print(f"   [shard {shard}] {mod}/{bm} skipped: {exc}", flush=True)
        except cacheutils.MissingEmbeddingCache:
            raise
        except Exception as exc:
            import traceback

            print(
                f"!! [shard {shard}] {mod}/{bm} FAILED: {type(exc).__name__}: {exc} (skipping; re-run to resume)",
                flush=True,
            )
            traceback.print_exc()

    _report_regime_problems()
