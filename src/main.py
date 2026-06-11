"""Orchestrator for the frozen-embedding robustness pipeline.

    task spec -> get_input -> corrupt -> encode (frozen) -> cache
              -> {ERM, ACTIVE_METHODS} -> task probe -> tables

Edit the config block below, then::

    cd src && python main.py                 # single process
    cd src && python utils/gputils.py        # split (encoder, task) work across all GPUs

Everything is resumable: assembled benchmarks are pickle-cached, encoder embeddings
are cached per condition (atomic writes), and probe results are appended to a
JSON-lines log as each cell completes. Re-running skips all finished work, so a
crash only loses the in-flight cell.
"""

from __future__ import annotations

import importlib
from typing import Any

from joblib import Parallel, delayed

from evals import evals as EV  # noqa: E402
from utils import (  # noqa: E402
    cacheutils,
    gputils,
    perf,
)
from utils import ioutils as IOU  # noqa: E402

# === Configuration ===========================================================
ACTIVE_ENCODERS = ["presto", "olmoearth", "tessera", "agrifm"]  # encoders: presto, olmoearth, tessera, agrifm
TASKS = ["bin-crop-class", "crop-class", "pheno-reg", "yield-reg"]  # which tasks; see src/evals/tasks/
ACTIVE_METHODS = []  # post-hoc methods: grit, dfr, tent, … (empty list = ERM baseline only)
ACTIVE_HOLDOUTS = None  # holdout regions (None = all defaults for each task)
ACTIVE_CONDITIONS = None  # stress conditions (None = all; or subset: ["clean", "sensor_off_s2", ...])
MAX_SAMPLES = None  # benchmark samples per task (None = all)
SEEDS = [42]  # random seeds for probe reproducibility
CLEAN_ONLY = False  # True = only the clean condition
ACTIVE_AXES = ["sensorial", "geographic", "temporal"]  # which robustness axes to evaluate
ENCODER_KWARGS = {}  # per-encoder kwargs (device is auto-set to the visible GPU); e.g. {"batch_size": 4096}
N_JOBS = -1  # cores for the (single-threaded sklearn) probe sweep; -1 = all (the main CPU win)
OVERWRITE_MODE = "skip"  # "skip" = skip (encoder, task) if results already exist;
                          # "override" = re-run everything, replacing old results
# =============================================================================

BENCH_SHUFFLE_SEED = 0  # fixed so cached embeddings stay aligned to the bench row order

# TASK_KIND -> (probe runner, metric list, primary headline metric)
# Source-fraction budgets (secondary diagnostic):
DISPATCH = {
    "binary": (EV.run_probes, EV.METRICS_BINARY, "calibrated_f1"),
    "multiclass": (EV.run_probes_multiclass, EV.METRICS_MULTICLASS, "macro_f1"),
    "regression": (EV.run_probes_regression, EV.METRICS_REGRESSION, "rmse"),
}
# Target-region budgets (main experiment):
DISPATCH_TARGET = {
    "binary": (EV.run_probes_target, EV.METRICS_BINARY, "calibrated_f1"),
    "multiclass": (EV.run_probes_multiclass_target, EV.METRICS_MULTICLASS, "macro_f1"),
    "regression": (EV.run_probes_regression_target, EV.METRICS_REGRESSION, "rmse"),
}


def load_task(task_name: str):
    return importlib.import_module(f"evals.tasks.{task_name.replace('-', '_')}")


# --------------------------------------------------------------------------- #
# Methods
# --------------------------------------------------------------------------- #


def build_methods(task_kind: str, seed: int):
    """name -> (cls_or_none, kwargs).  ERM (no transform) is always included."""
    methods: dict[str, tuple[Any, dict]] = {"erm": (None, {})}
    for name in ACTIVE_METHODS:
        mod = importlib.import_module(f"methods.{name}")
        for vname, base_kwargs in mod.variants(task_kind).items():
            methods[vname] = (getattr(mod, name.title()), {**base_kwargs, "seed": seed})
    return methods


# --------------------------------------------------------------------------- #
# One probe cell (runs in a worker process; embeddings are memmapped by joblib)
# --------------------------------------------------------------------------- #


def _probe_cell(
    probe_fn, emb_clean, emb_cond, train, test, y, groups, cls, kwargs, uses_target, cond_clean, meta, seed
) -> list[dict]:
    """Fit the (optional) method transform and run the source-budget-swept probe.

    Self-contained and picklable so joblib can run many cells across cores. Slicing
    happens here; the full embedding arrays are shared read-only (memmapped) by joblib.
    """
    x_clean_tr, x_cond_te = emb_clean[train], emb_cond[test]
    y_tr, y_te, g_tr = y[train], y[test], groups[train]
    mname = meta.get("method", "?")
    identity = {k: meta[k] for k in ("seed", "holdout", "condition", "method") if k in meta}
    transform = None
    if cls is not None:
        x_paired = x_cond_te if uses_target else (None if cond_clean else emb_cond[train])
        transform = cls(**kwargs)
        with perf.measure(f"method.fit/{mname}", identity=identity,
                          n_samples=len(train), n_features=x_clean_tr.shape[1]):
            transform.fit(x_clean_tr, y_tr, g_tr, x_paired=x_paired)
    rows: list[dict] = []
    perf.set_identity(identity)
    with perf.measure(
        f"probe.run/{meta.get('task', '?')}/{mname}",
        n_samples_train=len(train),
        n_samples_test=len(test),
        n_features=x_clean_tr.shape[1],
    ):
        probe_fn(rows, x_clean_tr, x_cond_te, y_tr, y_te, seed, transform=transform, meta=meta, groups_train=g_tr)
    perf.set_identity(None)
    return rows


def _probe_cell_target(
    probe_fn, emb_clean, emb_cond, train, test, y, groups, cls, kwargs, uses_target, cond_clean, meta, seed
) -> list[dict]:
    """Target-budget variant: passes the *full* target pool so the probe runner
    can sample N target labels for few-shot training and use the rest for testing.

    Budget = 0 -> strict geographic holdout (train only on source).
    Budget > 0 -> move that many target samples from test into training.
    """
    x_source_tr, x_target_full = emb_clean[train], emb_cond[test]
    y_source_tr, y_target_full = y[train], y[test]
    g_source_tr = groups[train]
    mname = meta.get("method", "?")
    identity = {k: meta[k] for k in ("seed", "holdout", "condition", "method") if k in meta}
    transform = None
    if cls is not None:
        x_paired = x_target_full if uses_target else (None if cond_clean else emb_cond[train])
        transform = cls(**kwargs)
        with perf.measure(f"method.fit/{mname}", identity=identity,
                          n_samples=len(train), n_features=x_source_tr.shape[1]):
            transform.fit(x_source_tr, y_source_tr, g_source_tr, x_paired=x_paired)
    rows: list[dict] = []
    perf.set_identity(identity)
    with perf.measure(
        f"probe.target/{meta.get('task', '?')}/{mname}",
        n_samples_source=len(train),
        n_samples_target=len(test),
        n_features=x_source_tr.shape[1],
    ):
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
        )
    perf.set_identity(None)
    return rows


# --------------------------------------------------------------------------- #
# Run
# --------------------------------------------------------------------------- #


def run(task_name, encoder_name, conditions, seeds, max_samples, **enc_kwargs) -> None:
    task = load_task(task_name)
    probe_fn_src, metrics, primary = DISPATCH[task.TASK_KIND]
    probe_fn_tgt, _, _ = DISPATCH_TARGET[task.TASK_KIND]
    all_holdouts = task.HOLDOUTS or EV.STRICT_HOLDOUTS
    if isinstance(ACTIVE_HOLDOUTS, dict):
        holdouts = ACTIVE_HOLDOUTS.get(task_name, all_holdouts)
    else:
        holdouts = ACTIVE_HOLDOUTS or all_holdouts

    gi_kwargs = {"timesteps": 12} if task.BENCHMARK in ("eurocropsml", "sickle") else {}
    bench_kwargs = dict(max_samples=max_samples, shuffle=True, seed=BENCH_SHUFFLE_SEED, **gi_kwargs)
    tag = cacheutils.bench_tag(task.BENCHMARK, bench_kwargs)
    perf.reset()
    bench = cacheutils.cached_bench(task.BENCHMARK, tag, **bench_kwargs)
    y, groups = task.make_targets(bench)
    emb = cacheutils.extract_and_cache(
        bench, task.BENCHMARK, encoder_name, tag, conditions, overwrite=OVERWRITE_MODE, **enc_kwargs
    )
    cond_names = [c[0] for c in conditions]

    results_dir = cacheutils.OUTPUT_DIR / "results" / encoder_name / task_name
    rows_path = results_dir / "probe_results.jsonl"

    if OVERWRITE_MODE == "override":
        for p in [rows_path, results_dir / "probe_results.csv", results_dir / "summary.csv"]:
            if p.exists():
                p.unlink()

    rows = IOU.read_jsonl(rows_path)
    done = {(r.get("seed"), r.get("holdout"), r.get("condition"), r.get("method"), r.get("budget_type")) for r in rows}

    jobs = []
    def uses_target_flag(cls):
        return getattr(cls, "USES_TARGET", False)
    for seed in seeds:
        for mname, (cls, kwargs) in build_methods(task.TASK_KIND, seed).items():
            for holdout in holdouts:
                try:
                    train, _val, test, _tv = EV.make_strict_holdout_splits(y, groups, holdout, seed)
                except ValueError:
                    continue
                for cond in cond_names:
                    meta = {
                        "encoder": encoder_name,
                        "benchmark": task.BENCHMARK,
                        "task": task_name,
                        "method": mname,
                        "holdout": holdout,
                        "condition": cond,
                        "train_regime": "clean",
                    }
                    # ---- Target-region budgets (primary experiment) ----
                    if (seed, holdout, cond, mname, "target") not in done:
                        jobs.append(
                            delayed(_probe_cell_target)(
                                probe_fn_tgt,
                                emb["clean"],
                                emb[cond],
                                train,
                                test,
                                y,
                                groups,
                                cls,
                                kwargs,
                                uses_target_flag(cls),
                                cond == "clean",
                                {**meta, "budget_type": "target"},
                                seed,
                            )
                        )
                    # ---- Source-fraction budgets (secondary diagnostic) ----
                    if (seed, holdout, cond, mname, "source") not in done:
                        jobs.append(
                            delayed(_probe_cell)(
                                probe_fn_src,
                                emb["clean"],
                                emb[cond],
                                train,
                                test,
                                y,
                                groups,
                                cls,
                                kwargs,
                                uses_target_flag(cls),
                                cond == "clean",
                                {**meta, "budget_type": "source"},
                                seed,
                            )
                        )

    if jobs:
        for cell_rows in Parallel(n_jobs=N_JOBS, return_as="generator")(jobs):
            IOU.append_jsonl(rows_path, cell_rows)
            rows.extend(cell_rows)

    IOU.write_csv(results_dir / "probe_results.csv", rows)
    summary = IOU.summarize_rows(
        rows,
        keys=["encoder", "method", "condition", "budget_type", "label_budget"],
        metrics=metrics,
    )
    IOU.write_csv(results_dir / "summary.csv", summary)

    perf_path = results_dir / "perf.jsonl"
    n_events = perf.write_log(perf_path)
    print(f"  perf: {n_events} events logged to {perf_path}", flush=True)

    _print_headline(task_name, primary, summary)


def _print_headline(task_name, primary, summary) -> None:
    # Target-budget 0 (strict geographic holdout) — main headline
    print(f"\n=== {task_name}: target-budget 0 (strict holdout) ===")
    print(f"{'method':<22} {'mean_' + primary:>16}")
    for r in summary:
        if r.get("condition") == "clean" and r.get("budget_type") == "target" and r.get("label_budget") == 0:
            print(f"{r['method']:<22} {r['mean_' + primary]:>16.4f}")

    # Source-budget 1.0 (full source labels) — secondary headline
    print(f"\n=== {task_name}: source-budget 1.0 (full source labels) ===")
    print(f"{'method':<22} {'mean_' + primary:>16}")
    for r in summary:
        if r.get("condition") == "clean" and r.get("budget_type") == "source" and r.get("label_budget") == 1.0:
            print(f"{r['method']:<22} {r['mean_' + primary]:>16.4f}")


if __name__ == "__main__":
    # Step 1: filter by ACTIVE_AXES (which robustness dimensions are active)
    if CLEAN_ONLY:
        all_conditions = [EV.CONDITIONS[0]]
    else:
        all_conditions = EV.filter_conditions_by_axes(EV.CONDITIONS, ACTIVE_AXES)
    # Step 2: further filter by ACTIVE_CONDITIONS (explicit name list)
    if ACTIVE_CONDITIONS is not None:
        cond_map = {c[0]: c for c in all_conditions}
        unknown = set(ACTIVE_CONDITIONS) - cond_map.keys()
        if unknown:
            raise ValueError(f"Unknown conditions: {sorted(unknown)}. Known: {list(cond_map)}")
        conditions = [cond_map[n] for n in ACTIVE_CONDITIONS]
    else:
        conditions = all_conditions

    enc_kwargs = dict(ENCODER_KWARGS)
    enc_kwargs.setdefault("device", gputils.device())  # the one GPU this shard sees (or cpu)

    # (encoder, task) is the unit of work; gputils keeps just this shard's slice.
    work = gputils.take_shard([(enc, tsk) for enc in ACTIVE_ENCODERS for tsk in TASKS])
    shard, nshards = gputils.shard_indices()
    for enc, tsk in work:
        print(f"\n========== [shard {shard}/{nshards}] {enc} / {tsk} ==========", flush=True)
        run(tsk, enc, conditions, SEEDS, MAX_SAMPLES, **enc_kwargs)
