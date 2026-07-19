"""Split-preprocessing generator: freeze the four regimes' partitions to data/splits/.

Behavior-preserving extraction. This driver calls the EXISTING regime code
(``evals.regimes.base.iter_splits`` / ``segmentation_fold_configs``) exactly as the runtime does,
then serializes the realized partitions as stable IDs via ``evals.split_artifacts``. It constructs
no splits itself and changes no partition membership; current fallbacks, dropped folds, and
exclusions are recorded as audit metadata, never altered.

No command-line arguments: edit the CONFIG block below and run it.

    python tools/generate_splits.py

``AUDIT_ONLY = True`` constructs and validates every split and reports, but publishes NO split
artifacts under ``data/splits/`` (no assignments.csv / exclusions.csv / manifest.json / generation.json
/ index.json). It is not a pure no-op: loading a benchmark still reads and MAY populate the existing
benchmark pickle cache under ``data/cache/benchmark/`` via ``cacheutils.cached_bench`` (the same
loader the runtime uses). It writes nothing under ``data/splits/``.

Each ``<seed>`` is the existing experiment seed (SEEDS = [0, 1, 2]); seed-dependent regimes (e.g.
random_id) therefore produce three distinct canonical splits, one per seed -- not a 3x3 grid.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from evals import evals as EV  # noqa: E402
from evals import split_artifacts as SA  # noqa: E402
from evals.regimes import base as regime_base  # noqa: E402
from utils import cacheutils  # noqa: E402  (benchmark loader only -- not the embedding cache/manifests)

# === Configuration ===========================================================
SPLITS_ROOT = cacheutils.SCRATCH / "splits"          # data/splits/
BENCHMARKS = ["cropharvest", "eurocropsml", "breizhcrops", "pastis"]
REGIMES = ["random_id", "official", "geographic_ood", "spatial_cluster_ood"]
SEEDS = [0, 1, 2]
MAX_SAMPLES = None
AUDIT_ONLY = True     # True = validate + report; publishes NO artifacts under data/splits/ (may
#                       populate/read the benchmark pickle cache via cached_bench). False = publish.
OVERWRITE = True      # when publishing, regenerate an existing canonical leaf in place
# =============================================================================


def _jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (set, frozenset)):
        return sorted(_jsonable(v) for v in obj)
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if hasattr(obj, "item") and not isinstance(obj, (str, bytes)):
        try:
            return obj.item()
        except Exception:
            return obj
    return obj


def _dropped_from_events(events):
    out = []
    for e in events:
        if e.get("kind") in ("dropped_holdout", "dropped_split"):
            out.append({"label": e.get("holdout"), "reason": e.get("reason"), "stage": e.get("stage")})
    return out


def _requested_tabular(bench, regime, holdouts, y):
    """The holdout labels a regime WILL attempt -- derived from the regime spec / domain census,
    independently of what it managed to yield (so a fully-dropped split is still identified)."""
    if regime == "random_id":
        return ["random_id"]
    if regime == "spatial_cluster_ood":
        from evals.regimes import spatial_cluster_ood as sc
        return [str(sc._spec(bench).get("label", "spatial_clusters"))]
    from evals.regimes import geographic_ood as geo
    domains = np.asarray(regime_base.load_regime(regime).assign_domains(bench, holdouts), dtype=object)
    if isinstance(holdouts, dict) and holdouts.get("strategy") == "leave_one_domain_out":
        return [r["domain"] for r in geo.domain_census(y, domains, holdouts)]
    if isinstance(holdouts, dict):
        return [str(holdouts.get("label", regime))]
    return [str(h) for h in (holdouts or [])]


def _requested_dense(bench, bench_mod, regime):
    """Dense analogue: the fold-config labels a regime declares, independent of the cache/yield."""
    if regime == "random_id":
        return ["random_patch"]
    if regime == "spatial_cluster_ood":
        from evals.regimes import spatial_cluster_ood as sc
        return [str(sc._spec(bench).get("label", "spatial_clusters"))]
    if regime == "official":
        return [f"fold_{sorted(bench_mod.TEST_FOLDS)[0]}"]
    from evals.regimes import geographic_ood as geo
    return [str(label) for label, *_ in geo.iter_fold_splits(bench_mod)]


def generate_tabular(root, bench, bench_mod, regime, seed, *, audit_only, overwrite):
    y, _groups = bench_mod.make_targets(bench)
    sample_ids = getattr(bench, "sample_ids", None)
    if sample_ids is None:
        raise SA.SplitArtifactError(f"{bench_mod.BENCHMARK}: benchmark exposes no stable sample_ids")
    holdouts = regime_base.holdouts_for(bench_mod, regime)
    val_group = regime_base.val_group_for(bench_mod, regime)
    params = {
        "holdouts": _jsonable(holdouts),
        "val_group": _jsonable(val_group),
        "assembly_seed": 0,
        "max_samples": MAX_SAMPLES,
    }

    regime_base.clear_split_audit_events()
    regime_base.clear_domain_census()
    problems_before = len(regime_base.REGIME_PROBLEMS)
    events = regime_base.SPLIT_AUDIT_EVENTS

    leaves, yielded = [], []
    prev = 0
    for (label, train, val, test, domains, has_target, group_kind, source_val, source_test) in \
            regime_base.iter_splits(regime, bench, y, holdouts, seed, val_group=val_group):
        window = list(events[prev:len(events)])
        prev = len(events)
        spec, eligible = SA.build_tabular_leaf(
            bench_mod.BENCHMARK, regime, seed, label=label,
            train=train, val=val, test=test, source_val=source_val, source_test=source_test,
            domains=domains, labels=y, sample_ids=sample_ids, has_target=has_target,
            group_kind=group_kind, params=params, audit_events=window,
        )
        if audit_only:
            SA.validate_leaf(spec, eligible)
        else:
            SA.publish_leaf(root, spec, eligible, overwrite=overwrite)
        yielded.append(str(label))
        leaves.append({
            "regime": regime, "seed": int(seed), "holdout": str(label),
            "holdout_dirname": SA.holdout_dirname(str(label)), "target_unit": "sample",
            "n_excluded": len(spec.exclusions),
        })

    all_events = _jsonable(list(events))
    dropped = _dropped_from_events(all_events)
    requested = _requested_tabular(bench, regime, holdouts, y)
    problems = [list(p) for p in regime_base.REGIME_PROBLEMS[problems_before:]]
    if not audit_only:
        SA.write_generation(
            root, bench_mod.BENCHMARK, regime, seed,
            requested=requested, yielded=yielded, dropped=dropped,
            audit_events=all_events, regime_problems=problems,
        )
    return leaves, {"requested": requested, "yielded": yielded, "dropped": dropped, "regime_problems": problems}


def _dense_cache(bench):
    all_patch_ids = [int(p) for p in bench.patch_ids(None)]
    fold_of = {int(p.patch_id): int(p.fold) for p in bench.patches}
    class_sets = bench.patch_class_sets(all_patch_ids)
    patch_latlon = {int(k): v for k, v in bench.patch_latlon.items()}
    return dict(all_patch_ids=all_patch_ids, fold_of=fold_of, class_sets=class_sets, patch_latlon=patch_latlon)


def generate_dense(root, bench, bench_mod, regime, seed, *, audit_only, overwrite, dense_cache):
    regime_base.clear_split_audit_events()
    problems_before = len(regime_base.REGIME_PROBLEMS)
    events = regime_base.SPLIT_AUDIT_EVENTS

    leaves, yielded = [], []
    prev = 0
    for _regime_name, cfg in regime_base.segmentation_fold_configs(
        bench_mod, [regime], seed=seed, emb_dir=None, bench=bench
    ):
        window = list(events[prev:len(events)])
        prev = len(events)
        params = {
            "train_folds": sorted(int(f) for f in cfg.train_folds),
            "val_folds": sorted(int(f) for f in cfg.val_folds),
            "test_folds": sorted(int(f) for f in cfg.test_folds),
            "assembly_seed": 0,
            "max_samples": MAX_SAMPLES,
        }
        spec, eligible = SA.build_dense_leaf(
            bench_mod.BENCHMARK, regime, seed, cfg=cfg, bench=bench, params=params,
            audit_events=window, **dense_cache,
        )
        if audit_only:
            SA.validate_leaf(spec, eligible)
        else:
            SA.publish_leaf(root, spec, eligible, overwrite=overwrite)
        yielded.append(str(cfg.label))
        leaves.append({
            "regime": regime, "seed": int(seed), "holdout": str(cfg.label),
            "holdout_dirname": SA.holdout_dirname(str(cfg.label)), "target_unit": "patch",
            "n_excluded": len(spec.exclusions),
        })

    all_events = _jsonable(list(events))
    dropped = _dropped_from_events(all_events)
    requested = _requested_dense(bench, bench_mod, regime)
    problems = [list(p) for p in regime_base.REGIME_PROBLEMS[problems_before:]]
    if not audit_only:
        SA.write_generation(
            root, bench_mod.BENCHMARK, regime, seed,
            requested=requested, yielded=yielded, dropped=dropped,
            audit_events=all_events, regime_problems=problems,
        )
    return leaves, {"requested": requested, "yielded": yielded, "dropped": dropped, "regime_problems": problems}


def generate_benchmark(root, benchmark, *, audit_only, overwrite):
    bench_mod = EV.load_benchmark(benchmark)
    # Same canonical assembly the runtime consumes (in the reconciled cache scheme cached_bench takes
    # only the benchmark name), so generated splits align 1:1 with the runtime's row/patch order.
    bench = cacheutils.cached_bench(bench_mod.BENCHMARK)
    dense = getattr(bench_mod, "LABEL_KIND", "") == "segmentation"
    supported = getattr(bench_mod, "SPLIT_REGIMES", REGIMES)
    regimes = [r for r in REGIMES if r in supported]

    dense_cache = _dense_cache(bench) if dense else None
    all_leaves = []
    for seed in SEEDS:
        for regime in regimes:
            if dense:
                leaves, summ = generate_dense(
                    root, bench, bench_mod, regime, seed,
                    audit_only=audit_only, overwrite=overwrite, dense_cache=dense_cache,
                )
            else:
                leaves, summ = generate_tabular(
                    root, bench, bench_mod, regime, seed, audit_only=audit_only, overwrite=overwrite,
                )
            all_leaves.extend(leaves)
            drp = f" DROPPED={[d['label'] for d in summ['dropped']]}" if summ["dropped"] else ""
            prb = f" PROBLEMS={len(summ['regime_problems'])}" if summ["regime_problems"] else ""
            print(f"  {benchmark}/{regime}/seed={seed}: {len(leaves)} leaf(s){drp}{prb}", flush=True)
    if not audit_only:
        SA.write_index(root, benchmark, all_leaves)
    return all_leaves


def main() -> int:
    root = Path(SPLITS_ROOT)
    mode = (
        "AUDIT-ONLY (no split artifacts written under data/splits/; benchmark pickle cache may be populated/read)"
        if AUDIT_ONLY else f"PUBLISH -> {root}"
    )
    print(f"[generate_splits] mode={mode}; benchmarks={BENCHMARKS}; regimes={REGIMES}; seeds={SEEDS}", flush=True)
    failed = []
    total = 0
    for benchmark in BENCHMARKS:
        print(f"[{benchmark}]", flush=True)
        try:
            leaves = generate_benchmark(root, benchmark, audit_only=AUDIT_ONLY, overwrite=OVERWRITE)
            total += len(leaves)
        except Exception as exc:  # noqa: BLE001 -- report per-benchmark, fail the whole run nonzero
            failed.append((benchmark, f"{type(exc).__name__}: {exc}"))
            print(f"  !! {benchmark} FAILED: {type(exc).__name__}: {exc}", flush=True)
    print(f"\n[generate_splits] {total} leaf(s) across {len(BENCHMARKS) - len(failed)}/{len(BENCHMARKS)} benchmark(s).", flush=True)
    if failed:
        for b, why in failed:
            print(f"  FAILED {b}: {why}", flush=True)
        return 1
    print("[generate_splits] OK", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
