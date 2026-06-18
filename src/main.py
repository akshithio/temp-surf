"""Orchestrator for the frozen-embedding robustness pipeline.

    task spec -> get_input -> degrade -> encode (frozen) -> cache
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
import os
from typing import Any

import numpy as np
from joblib import Parallel, delayed

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
ACTIVE_ENCODERS = ["presto", "olmoearth", "galileo", "agrifm", "tessera"]
TASKS = ["bin-crop-class", "crop-class", "pastis-crop-seg"]
ACTIVE_AXES = ["geographic"]  # choose between temporal, sensorial, geographic
ACTIVE_METHODS = []

ACTIVE_CONDITIONS = None  # applicable only if either temporal or sensorial is active
SPLIT_REGIMES = ["random_id", "grouped_ood", "geographic_ood"]
GROUPED_FOLDS = 5  # number of random group folds for grouped_ood

MAX_SAMPLES = None  # benchmark samples per task (None = all)
MAX_DENSE_PIXELS = 50_000  # sampled pixels per PASTIS fold partition and condition
N_JOBS = -1  # cores for the (single-threaded sklearn) probe sweep; -1 = all
OVERWRITE_MODE = "skip"  # choose between skip, override
SEEDS = [0, 1]  # random seeds for probe reproducibility
# =============================================================================

BENCH_SHUFFLE_SEED = 0  # fixed so cached embeddings stay aligned to the bench row order

# TASK_KIND -> (probe runner, metric list)
DISPATCH = {
    "binary": (EV.run_probes, EV.METRICS_BINARY),
    "multiclass": (EV.run_probes_multiclass, EV.METRICS_MULTICLASS),
}
DISPATCH_TARGET = {
    "binary": (EV.run_probes_target, EV.METRICS_BINARY),
    "multiclass": (EV.run_probes_multiclass_target, EV.METRICS_MULTICLASS),
}


def load_task(task_name: str):
    return importlib.import_module(f"evals.tasks.{task_name.replace('-', '_')}")


def _effective_n_jobs(embeddings: dict[str, np.ndarray]) -> int:
    """Bound concurrent probe fits when embedding matrices are large."""
    cpu_count = os.cpu_count() or 1
    requested = N_JOBS
    if requested < 0:
        requested = max(1, cpu_count + 1 + requested)
    elif requested == 0:
        requested = 1

    max_bytes = max(np.asarray(arr).nbytes for arr in embeddings.values())
    if max_bytes >= 2_000_000_000:
        return 1
    if max_bytes >= 200_000_000:
        return min(requested, 2)
    return requested


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
    probe_fn,
    emb_train,
    emb_cond,
    train,
    test,
    y,
    groups,
    cls,
    kwargs,
    uses_target,
    cond_clean,
    meta,
    seed,
) -> tuple[list[dict], list[dict]]:
    """Fit the (optional) method transform and run the source-budget-swept probe.

    ``emb_train`` is what the probe trains on: ``emb["baseline"]`` for the
    deployment-realistic regime, ``emb[condition]`` for the degrade\\degrade regime.
    Self-contained and picklable so joblib can run many cells across cores. Returns
    ``(rows, per_sample_predictions)``.
    """
    x_tr, x_cond_te = emb_train[train], emb_cond[test]
    y_tr, y_te, g_tr = y[train], y[test], groups[train]
    mname = meta.get("method", "?")
    identity = {k: meta[k] for k in ("seed", "holdout", "condition", "method") if k in meta}
    transform = None
    if cls is not None:
        x_paired = x_cond_te if uses_target else (None if cond_clean else emb_cond[train])
        transform = cls(**kwargs)
        with perf.measure(f"method.fit/{mname}", identity=identity, n_samples=len(train), n_features=x_tr.shape[1]):
            transform.fit(x_tr, y_tr, g_tr, x_paired=x_paired)
    rows: list[dict] = []
    perf.set_identity(identity)
    with perf.measure(
        f"probe.run/{meta.get('task', '?')}/{mname}",
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
        )
    perf.set_identity(None)
    return rows, preds


def _probe_cell_target(
    probe_fn,
    emb_train,
    emb_cond,
    train,
    test,
    y,
    groups,
    cls,
    kwargs,
    uses_target,
    cond_clean,
    meta,
    seed,
) -> tuple[list[dict], list[dict]]:
    """Target-budget variant: passes the *full* target pool so the probe runner
    can sample N target labels for few-shot training and use the rest for testing.

    Budget = 0 -> strict geographic holdout (train only on source) = the headline OOD.
    Budget > 0 -> move that many target samples from test into training.
    """
    x_source_tr, x_target_full = emb_train[train], emb_cond[test]
    y_source_tr, y_target_full = y[train], y[test]
    g_source_tr = groups[train]
    mname = meta.get("method", "?")
    identity = {k: meta[k] for k in ("seed", "holdout", "condition", "method") if k in meta}
    transform = None
    if cls is not None:
        x_paired = x_target_full if uses_target else (None if cond_clean else emb_cond[train])
        transform = cls(**kwargs)
        with perf.measure(
            f"method.fit/{mname}", identity=identity, n_samples=len(train), n_features=x_source_tr.shape[1]
        ):
            transform.fit(x_source_tr, y_source_tr, g_source_tr, x_paired=x_paired)
    rows: list[dict] = []
    perf.set_identity(identity)
    with perf.measure(
        f"probe.target/{meta.get('task', '?')}/{mname}",
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
        )
    perf.set_identity(None)
    return rows, preds


# --------------------------------------------------------------------------- #
# Run
# --------------------------------------------------------------------------- #


def _iter_splits(split_regime, y, groups, holdouts, seed):
    """Yield (split_label, train_idx, test_idx, has_target) for a split regime.

    has_target marks an OOD split (train/test are different regions) where the
    few-shot target-budget sweep is meaningful; random_id has no target region.
    """
    if split_regime == "random_id":
        train, _val, test = EV.make_splits(y, seed)  # ID upper anchor: train+test share regions
        yield "random_id", train, test, False
    elif split_regime == "grouped_ood":
        yield from (
            (lbl, tr, te, True) for lbl, tr, te in EV.make_grouped_holdout_folds(y, groups, seed, n_folds=GROUPED_FOLDS)
        )
    elif split_regime == "geographic_ood":
        for holdout in holdouts:
            try:
                train, _val, test, _tv = EV.make_strict_holdout_splits(y, groups, holdout, seed)
            except ValueError:
                continue
            yield holdout, train, test, True
    else:
        raise ValueError(f"Unknown split_regime: {split_regime}")


def run_segmentation(task_name, encoder_name, conditions, seeds, max_samples, **enc_kwargs) -> None:
    """Run the fixed PASTIS-R folds without materializing the full release."""
    task = load_task(task_name)
    conditions = [condition for condition in conditions if condition[1] != "climate"]
    bench_kwargs = dict(max_samples=max_samples, shuffle=True, seed=BENCH_SHUFFLE_SEED)
    tag = cacheutils.bench_tag(task.BENCHMARK, bench_kwargs)
    perf.reset()
    bench = cacheutils.cached_bench(task.BENCHMARK, tag, **bench_kwargs)
    cache_dirs = cacheutils.extract_dense_and_cache(
        bench,
        task.BENCHMARK,
        encoder_name,
        tag,
        conditions,
        overwrite=OVERWRITE_MODE,
        **enc_kwargs,
    )
    results_dir = cacheutils.OUTPUT_DIR / "results" / encoder_name / task_name
    rows_path = results_dir / "probe_results.jsonl"
    if OVERWRITE_MODE == "override":
        for path in (rows_path, results_dir / "probe_results.csv", results_dir / "summary.csv"):
            if path.exists():
                path.unlink()
    rows = IOU.read_jsonl(rows_path)
    done = {
        (r.get("seed"), r.get("condition"), r.get("train_regime"), r.get("method"))
        for r in rows
    }
    for seed in seeds:
        for method_name, (cls, kwargs) in build_methods(task.TASK_KIND, seed).items():
            for condition, _sensor_off, _temporal_drop in conditions:
                train_regimes = ["clean\\degrade"] if condition == "baseline" else [
                    "clean\\degrade",
                    "degrade\\degrade",
                ]
                for train_regime in train_regimes:
                    key = (seed, condition, train_regime, method_name)
                    if key in done:
                        continue
                    train_condition = "baseline" if train_regime == "clean\\degrade" else condition
                    x_train, y_train, groups_train = cacheutils.load_dense_samples(
                        cache_dirs[train_condition], task.TRAIN_FOLDS, MAX_DENSE_PIXELS, seed
                    )
                    x_val, y_val, _ = cacheutils.load_dense_samples(
                        cache_dirs[condition], task.VAL_FOLDS, MAX_DENSE_PIXELS, seed + 10_000
                    )
                    x_test, y_test, _ = cacheutils.load_dense_samples(
                        cache_dirs[condition], task.TEST_FOLDS, MAX_DENSE_PIXELS, seed + 20_000
                    )
                    transform = None
                    if cls is not None:
                        transform = cls(**kwargs)
                        paired = x_test if getattr(cls, "USES_TARGET", False) else None
                        transform.fit(x_train, y_train, groups_train, x_paired=paired)
                    cell_rows: list[dict] = []
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
                        meta={
                            "encoder": encoder_name,
                            "benchmark": task.BENCHMARK,
                            "task": task_name,
                            "method": method_name,
                            "split_regime": "official_folds",
                            "holdout": "fold_5",
                            "condition": condition,
                            "train_regime": train_regime,
                        },
                    )
                    IOU.append_jsonl(rows_path, cell_rows)
                    rows.extend(cell_rows)
    IOU.write_csv(results_dir / "probe_results.csv", rows)
    summary = IOU.summarize_rows(
        rows,
        keys=[
            "encoder",
            "method",
            "evaluation_split",
            "train_regime",
            "condition",
            "budget_type",
            "label_budget",
        ],
        metrics=EV.METRICS_SEGMENTATION,
    )
    IOU.write_csv(results_dir / "summary.csv", summary)
    perf.write_log(results_dir / "perf.jsonl")


def run(task_name, encoder_name, conditions, seeds, max_samples, split_regimes, **enc_kwargs) -> None:
    task = load_task(task_name)
    if task.TASK_KIND == "segmentation":
        run_segmentation(task_name, encoder_name, conditions, seeds, max_samples, **enc_kwargs)
        return
    probe_fn_src, metrics = DISPATCH[task.TASK_KIND]
    probe_fn_tgt, _ = DISPATCH_TARGET[task.TASK_KIND]
    holdouts = task.HOLDOUTS or EV.STRICT_HOLDOUTS

    bench_kwargs = dict(max_samples=max_samples, shuffle=True, seed=BENCH_SHUFFLE_SEED)
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
    # Resume key: every dimension that defines a cell. (split_regime + train_regime are new.)
    done = {
        (
            r.get("seed"),
            r.get("split_regime"),
            r.get("holdout"),
            r.get("condition"),
            r.get("train_regime"),
            r.get("method"),
            r.get("budget_type"),
        )
        for r in rows
    }

    jobs = []

    def uses_target_flag(cls):
        return getattr(cls, "USES_TARGET", False)

    for seed in seeds:
        for mname, (cls, kwargs) in build_methods(task.TASK_KIND, seed).items():
            for split_regime in split_regimes:
                for split_label, train, test, has_target in _iter_splits(split_regime, y, groups, holdouts, seed):
                    for cond in cond_names:
                        # train_regime: "clean\degrade" = deployment-realistic (train on clean source);
                        # "degrade\degrade" = oracle (train on the SAME degradation as test).
                        # For baseline condition the two coincide, so only emit "clean\degrade".
                        train_regimes = (
                            ["clean\\degrade"] if cond == "baseline" else ["clean\\degrade", "degrade\\degrade"]
                        )
                        for train_regime in train_regimes:
                            emb_train = emb["baseline"] if train_regime == "clean\\degrade" else emb[cond]
                            meta = {
                                "encoder": encoder_name,
                                "benchmark": task.BENCHMARK,
                                "task": task_name,
                                "method": mname,
                                "split_regime": split_regime,
                                "holdout": split_label,
                                "condition": cond,
                                "train_regime": train_regime,
                            }
                            # ---- Target-budget sweep (deployment headline at budget 0) ----
                            if (
                                has_target
                                and (seed, split_regime, split_label, cond, train_regime, mname, "target") not in done
                            ):
                                jobs.append(
                                    delayed(_probe_cell_target)(
                                        probe_fn_tgt,
                                        emb_train,
                                        emb[cond],
                                        train,
                                        test,
                                        y,
                                        groups,
                                        cls,
                                        kwargs,
                                        uses_target_flag(cls),
                                        cond == "baseline",
                                        {**meta, "budget_type": "target"},
                                        seed,
                                    )
                                )
                            # ---- Source-fraction sweep (ID anchor for random_id; secondary diagnostic for OOD) ----
                            if (seed, split_regime, split_label, cond, train_regime, mname, "source") not in done:
                                jobs.append(
                                    delayed(_probe_cell)(
                                        probe_fn_src,
                                        emb_train,
                                        emb[cond],
                                        train,
                                        test,
                                        y,
                                        groups,
                                        cls,
                                        kwargs,
                                        uses_target_flag(cls),
                                        cond == "baseline",
                                        {**meta, "budget_type": "source"},
                                        seed,
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
        keys=["encoder", "method", "split_regime", "train_regime", "condition", "budget_type", "label_budget"],
        metrics=metrics,
    )
    IOU.write_csv(results_dir / "summary.csv", summary)

    # Geographic drop: delta = metric(random_id, baseline) - metric(geographic_ood, baseline, target=0)
    # with relative + floor-normalized drop, a region×seed CI, and a per-sample CI from predictions.
    deltas = IOU.compute_deltas(rows, metrics, predictions=IOU.read_jsonl(preds_path))
    IOU.write_csv(results_dir / "deltas.csv", deltas)

    perf_path = results_dir / "perf.jsonl"
    n_events = perf.write_log(perf_path)
    print(f"  perf: {n_events} events logged to {perf_path}", flush=True)


if __name__ == "__main__":
    # "baseline" always runs as the unstressed baseline
    conditions: list[tuple[str, str, float]] = [("baseline", "none", 0.0)]

    # Build pool of stress conditions filtered by ACTIVE_AXES
    stress_pool = [c for c in EV.filter_conditions_by_axes(EV.CONDITIONS, ACTIVE_AXES) if c[0] != "baseline"]

    # ACTIVE_CONDITIONS controls which stress conditions to add:
    #   None → all stress conditions within ACTIVE_AXES
    #   []   → no stress (baseline only)
    #   [...] → explicit subset of stress conditions
    if ACTIVE_CONDITIONS is None:
        conditions.extend(stress_pool)
    elif ACTIVE_CONDITIONS:
        cond_map = {c[0]: c for c in stress_pool}
        unknown = set(ACTIVE_CONDITIONS) - cond_map.keys()
        if unknown:
            raise ValueError(
                f"Unknown stress conditions: {sorted(unknown)}. "
                f"Available within ACTIVE_AXES={ACTIVE_AXES}: {list(cond_map)}"
            )
        conditions.extend(cond_map[n] for n in ACTIVE_CONDITIONS)
    # else ACTIVE_CONDITIONS == []: just baseline, already set

    # Filter split regimes by ACTIVE_AXES (which robustness dimensions are active)
    split_regimes = EV.filter_split_regimes_by_axes(SPLIT_REGIMES, ACTIVE_AXES)
    if not split_regimes:
        raise ValueError(
            f"ACTIVE_AXES={ACTIVE_AXES} leaves no split regimes enabled. "
            f"Available: random_id (always), grouped_ood (needs 'geographic'), "
            f"geographic_ood (needs 'geographic')."
        )

    enc_kwargs = {"device": gputils.device()}  # the one GPU this shard sees (or cpu)

    # (encoder, task) is the unit of work; gputils keeps just this shard's slice.
    work = gputils.take_shard([(enc, tsk) for enc in ACTIVE_ENCODERS for tsk in TASKS])
    shard, nshards = gputils.shard_indices()
    for enc, tsk in work:
        print(f"\n========== [shard {shard}/{nshards}] {enc} / {tsk} ==========", flush=True)
        print(f"  split_regimes={split_regimes}  conditions={[c[0] for c in conditions]}", flush=True)
        try:
            run(tsk, enc, conditions, SEEDS, MAX_SAMPLES, split_regimes, **enc_kwargs)
        except NotImplementedError as exc:  # known (encoder, task) incompatibility -> clean skip, no traceback
            print(f"   [shard {shard}] {enc}/{tsk} skipped: {exc}", flush=True)
        except Exception as exc:  # isolate each pair: a missing weight / bad encoder skips just that pair
            import traceback

            print(
                f"!! [shard {shard}] {enc}/{tsk} FAILED: {type(exc).__name__}: {exc} (skipping; re-run to resume)",
                flush=True,
            )
            traceback.print_exc()
