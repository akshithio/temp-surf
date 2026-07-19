"""Split-preprocessing generator: freeze the four regimes' schema-v2 partitions to data/splits/.

Each regime emits explicit :class:`~evals.regimes.base.SourceTargetSplit` /
:class:`~evals.regimes.base.DenseSourceTargetSplit` objects (source_train / source_val / source_test /
target_label_pool / target_test, plus the ``has_target`` and ``supports_target_labels`` route
capabilities) built by the deterministic exact-size partitioners in ``evals.partition``. This driver
turns each realized split into one ``assignments.csv`` leaf (every eligible stable id, tagged
assigned / purged / excluded) and, once at the end, one central log ``data/logs/splits.json`` with the
run provenance and one summary entry per leaf (CSV path, SHA-256, counts, purge, exclusions, target
role/capability, validation).

No command-line arguments: edit the CONFIG block below and run it.

    python tools/generate_splits.py

``AUDIT_ONLY = True`` constructs and validates every split and reports, but writes NO assignments.csv
under ``data/splits/`` and NO ``data/logs/splits.json``. It is not a pure no-op: loading a benchmark
still reads and MAY populate the benchmark pickle cache under ``data/cache/benchmark/`` via
``cacheutils.cached_bench`` (the same loader the runtime uses).

Each ``<seed>`` is the existing experiment seed (SEEDS = [0, 1, 2]); seed-dependent regimes (e.g.
random_id) therefore produce three distinct canonical splits, one per seed -- not a 3x3 grid. Cell
boundaries for spatial_cluster_ood are frozen at ``split_spec.CLUSTER_SEED`` and never vary with the
run seed.
"""

from __future__ import annotations

import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
_REPO = Path(__file__).resolve().parents[1]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from evals import evals as EV  # noqa: E402
from evals import split_artifacts as SA  # noqa: E402
from evals import split_spec  # noqa: E402
from evals.regimes import base as regime_base  # noqa: E402
from utils import cacheutils  # noqa: E402  (benchmark loader only -- not the embedding cache/manifests)

# === Configuration ===========================================================
SPLITS_ROOT = cacheutils.SCRATCH / "splits"          # data/splits/
LOGS_PATH = SA.default_log_path(SPLITS_ROOT)          # data/logs/splits.json
BENCHMARKS = ["cropharvest", "eurocropsml", "breizhcrops", "pastis"]
REGIMES = ["random_id", "official", "geographic_ood", "spatial_cluster_ood"]
SEEDS = [0, 1, 2]
AUDIT_ONLY = False    # True = validate + report; writes NO assignments.csv / splits.json (may
#                       populate/read the benchmark pickle cache via cached_bench). False = write.
# =============================================================================


def _purge_km(benchmark: str, regime: str) -> float:
    """The source<->target purge radius applied by this regime (0 for the non-purging regimes)."""
    if regime in ("geographic_ood", "spatial_cluster_ood"):
        return float(split_spec.ALL_SPECS[benchmark].purge_km)
    return 0.0


def _cell_names() -> list[str]:
    from evals.regimes import spatial_cluster_ood as sc
    return list(sc._CELL_NAMES)


def _expected_tabular_labels(regime: str, bench_mod) -> list[str]:
    """The EXACT holdout leaves this (benchmark, regime) must realize, from the benchmark + split spec.
    geographic_ood includes the supplementary stress targets; a dropped one is a coverage failure."""
    if regime == "random_id":
        return ["random_id"]
    if regime == "official":
        return [str(h) for h in getattr(bench_mod, "OFFICIAL_HOLDOUTS", [])]
    if regime == "spatial_cluster_ood":
        return _cell_names()
    spec = split_spec.ALL_SPECS[bench_mod.BENCHMARK]  # geographic_ood
    return [str(t) for t in (*spec.geographic_targets, *spec.supplementary_targets)]


def _expected_dense_labels(regime: str, bench_mod) -> list[str]:
    if regime == "random_id":
        return ["random_patch"]
    if regime == "official":
        return [f"fold_{sorted(bench_mod.TEST_FOLDS)[0]}"]
    if regime == "spatial_cluster_ood":
        return _cell_names()
    return [str(t) for t in split_spec.ALL_SPECS[bench_mod.BENCHMARK].geographic_targets]  # tile-LODO


def _check_expected_coverage(regime: str, benchmark: str, seed: int, expected: list[str], yielded: list[str]) -> None:
    """Fail generation unless the realized leaves are EXACTLY the expected set -- one leaf per expected
    holdout, none missing, duplicated, or unexpected. So a dropped holdout (an absent LODO target)
    fails generation rather than silently thinning the split set."""
    got = list(yielded)
    if len(got) != len(set(got)):
        dup = sorted({h for h in got if got.count(h) > 1})
        raise SA.SplitArtifactError(f"{benchmark}/{regime}/seed={seed}: duplicate holdout leaf(s) {dup}")
    missing = sorted(set(expected) - set(got))
    unexpected = sorted(set(got) - set(expected))
    if missing or unexpected:
        raise SA.SplitArtifactError(
            f"{benchmark}/{regime}/seed={seed}: realized leaves {sorted(got)} != expected "
            f"{sorted(expected)} (missing {missing}, unexpected {unexpected}) -- refuse to freeze an "
            f"incomplete split set"
        )


def generate_tabular(root, bench, bench_mod, regime, seed, *, audit_only):
    """Build (and, unless audit_only, write) every tabular leaf for one (benchmark, regime, seed).
    Returns the list of central-log summary entries (each with its CSV's SHA-256 when written)."""
    regime_mod = regime_base.load_regime(regime)
    y, _groups = bench_mod.make_targets(bench)
    sample_ids = getattr(bench, "sample_ids", None)
    if sample_ids is None:
        raise SA.SplitArtifactError(f"{bench_mod.BENCHMARK}: benchmark exposes no stable sample_ids")
    # The per-sample domain basis (region/cell) is a property of the regime, recorded once per leaf.
    domains = regime_mod.sample_domains(bench, bench_mod)
    purge_km = _purge_km(bench_mod.BENCHMARK, regime)

    regime_base.clear_split_audit_events()
    events = regime_base.SPLIT_AUDIT_EVENTS
    entries, prev = [], 0
    for split in regime_mod.iter_source_target_splits(bench, bench_mod, seed):
        window = list(events[prev:len(events)])
        prev = len(events)
        rows, summary = SA.build_tabular_leaf(
            bench_mod.BENCHMARK, regime, seed,
            split=split, domains=domains, labels=y, sample_ids=sample_ids,
            audit_events=window, purge_km=purge_km,
        )
        if not audit_only:
            _path, sha = SA.write_assignments(root, bench_mod.BENCHMARK, regime, seed, str(split.label), rows)
            summary["sha256"] = sha
        entries.append(summary)
    _check_expected_coverage(
        regime, bench_mod.BENCHMARK, seed, _expected_tabular_labels(regime, bench_mod), [e["holdout"] for e in entries]
    )
    return entries


def _dense_cache(bench):
    all_patch_ids = [int(p) for p in bench.patch_ids(None)]
    class_sets = bench.patch_class_sets(all_patch_ids)
    patch_latlon = {int(k): v for k, v in bench.patch_latlon.items()}
    return dict(all_patch_ids=all_patch_ids, class_sets=class_sets, patch_latlon=patch_latlon)


def generate_dense(root, bench, bench_mod, regime, seed, *, audit_only, dense_cache):
    """Build (and, unless audit_only, write) every PASTIS patch-level leaf for one (regime, seed)."""
    regime_mod = regime_base.load_regime(regime)
    # Per-patch domain basis (fold / Sentinel tile / spatial cell), a regime property recorded once.
    domain_of = {int(k): str(v) for k, v in regime_mod.patch_domains(bench, bench_mod).items()}
    purge_km = _purge_km(bench_mod.BENCHMARK, regime)

    regime_base.clear_split_audit_events()
    events = regime_base.SPLIT_AUDIT_EVENTS
    entries, prev = [], 0
    for dense_split in regime_mod.iter_dense_source_target_splits(bench, bench_mod, seed):
        window = list(events[prev:len(events)])
        prev = len(events)
        rows, summary = SA.build_dense_leaf(
            bench_mod.BENCHMARK, regime, seed, dense_split=dense_split, audit_events=window,
            all_patch_ids=dense_cache["all_patch_ids"], domain_of=domain_of,
            class_sets=dense_cache["class_sets"], patch_latlon=dense_cache["patch_latlon"], purge_km=purge_km,
        )
        if not audit_only:
            _path, sha = SA.write_assignments(root, bench_mod.BENCHMARK, regime, seed, str(dense_split.label), rows)
            summary["sha256"] = sha
        entries.append(summary)
    _check_expected_coverage(
        regime, bench_mod.BENCHMARK, seed, _expected_dense_labels(regime, bench_mod), [e["holdout"] for e in entries]
    )
    return entries


def generate_benchmark(root, benchmark, *, audit_only):
    bench_mod = EV.load_benchmark(benchmark)
    # Same canonical assembly the runtime consumes, so generated splits align 1:1 with row/patch order.
    bench = cacheutils.cached_bench(bench_mod.BENCHMARK)
    dense = getattr(bench_mod, "LABEL_KIND", "") == "segmentation"
    supported = getattr(bench_mod, "SPLIT_REGIMES", REGIMES)
    regimes = [r for r in REGIMES if r in supported]

    dense_cache = _dense_cache(bench) if dense else None
    entries = []
    for seed in SEEDS:
        for regime in regimes:
            if dense:
                leaves = generate_dense(root, bench, bench_mod, regime, seed, audit_only=audit_only, dense_cache=dense_cache)
            else:
                leaves = generate_tabular(root, bench, bench_mod, regime, seed, audit_only=audit_only)
            entries.extend(leaves)
            print(f"  {benchmark}/{regime}/seed={seed}: {len(leaves)} leaf(s)", flush=True)
    return entries


def _code_revision() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=_REPO, text=True).strip()
    except Exception:  # noqa: BLE001 -- provenance best-effort; never block generation
        return "unknown"


def _split_config() -> dict:
    """The complete frozen policy needed to understand generation, as ONE plain explicit dict:
    eligible populations + exclusions + removed classes, CropHarvest canonical-region merges, the
    headline/supplementary/source-only units, official anchor definitions, purge radii, the exact-size
    sizing rules, PASTIS folds/tile counts/void class, and the run/cluster seeds + K-means config."""
    S = split_spec
    return {
        "sizing_rules": {
            "source": "source_val = source_test = ceil(0.1*N); source_train = remainder",
            "target": "target_test = ceil(0.2*N); target_label_pool = remainder",
            "official": "val = floor(0.1*N); train = remainder",
        },
        "run_seeds": list(S.RUN_SEEDS),
        "cluster_seed": S.CLUSTER_SEED,
        "n_clusters": S.N_CLUSTERS,
        "kmeans_n_init": 10,
        "benchmarks": {
            "cropharvest": {
                "eligible_population": S.CROPHARVEST.population,
                "purge_km": S.CROPHARVEST.purge_km,
                "id_kind": S.CROPHARVEST.id_kind,
                "population_exclusions": {"excluded_files": list(S.CROPHARVEST_EXCLUDED_FILES)},
                "canonical_region_merges": [list(r) for r in S.CROPHARVEST_REGION_MERGES],
                "canonical_regions": list(S.CROPHARVEST_CANONICAL_REGIONS),
                "headline_targets": list(S.CROPHARVEST.geographic_targets),
                "supplementary_targets": list(S.CROPHARVEST.supplementary_targets),
                "source_only_units": list(S.CROPHARVEST.source_only_units),
                "official_anchor": {"holdout": "togo", "definition": S.CROPHARVEST_OFFICIAL},
            },
            "eurocropsml": {
                "eligible_population": S.EUROCROPML.population,
                "raw_population": S.EUROCROPS_RAW_POPULATION,
                "purge_km": S.EUROCROPML.purge_km,
                "removed_classes": dict(S.EUROCROPS_REMOVED_CLASSES),
                "n_classes_after": S.EUROCROPS_N_CLASSES_AFTER,
                "country_population": dict(S.EUROCROPS_COUNTRY_POPULATION),
                "headline_targets": list(S.EUROCROPML.geographic_targets),
                "official_anchors": S.EUROCROPS_OFFICIAL,
            },
            "breizhcrops": {
                "eligible_population": S.BREIZHCROPS.population,
                "purge_km": S.BREIZHCROPS.purge_km,
                "headline_targets": list(S.BREIZHCROPS.geographic_targets),
                "official_anchor": S.BREIZHCROPS_OFFICIAL,
            },
            "pastis": {
                "eligible_population": S.PASTIS.population,
                "purge_km": S.PASTIS.purge_km,
                "headline_targets": list(S.PASTIS.geographic_targets),
                "tile_patch_counts": dict(S.PASTIS_TILE_PATCHES),
                "official_folds": S.PASTIS_OFFICIAL,
                "void_class_ignored": S.PASTIS_VOID_CLASS,
            },
        },
    }


def build_provenance() -> dict:
    """The run-level header of ``data/logs/splits.json`` -- timestamp, code revision, input
    identifiers, the run seeds this generation covered, and the complete frozen split configuration."""
    return {
        "generation_timestamp": datetime.now(UTC).isoformat(),
        "code_revision": _code_revision(),
        "inputs": {
            "data_root": str(cacheutils.SCRATCH),
            "benchmarks": list(BENCHMARKS),
            "data_fingerprint": os.environ.get("DATA_FINGERPRINT"),
        },
        "run_seeds": list(SEEDS),
        "split_config": _split_config(),
    }


def main() -> int:
    root = Path(SPLITS_ROOT)
    mode = "AUDIT-ONLY (no assignments.csv / splits.json written)" if AUDIT_ONLY else f"WRITE -> {root} + {LOGS_PATH}"
    print(f"[generate_splits] mode={mode}; benchmarks={BENCHMARKS}; regimes={REGIMES}; seeds={SEEDS}", flush=True)
    failed, entries = [], []
    for benchmark in BENCHMARKS:
        print(f"[{benchmark}]", flush=True)
        try:
            entries.extend(generate_benchmark(root, benchmark, audit_only=AUDIT_ONLY))
        except Exception as exc:  # noqa: BLE001 -- report per-benchmark, fail the whole run nonzero
            failed.append((benchmark, f"{type(exc).__name__}: {exc}"))
            print(f"  !! {benchmark} FAILED: {type(exc).__name__}: {exc}", flush=True)
    if not AUDIT_ONLY and not failed:
        SA.write_splits_log(LOGS_PATH, provenance=build_provenance(), entries=entries)
        print(f"[generate_splits] wrote {LOGS_PATH} ({len(entries)} leaf entries)", flush=True)
    print(f"\n[generate_splits] {len(entries)} leaf(s) across {len(BENCHMARKS) - len(failed)}/{len(BENCHMARKS)} benchmark(s).", flush=True)
    if failed:
        for b, why in failed:
            print(f"  FAILED {b}: {why}", flush=True)
        return 1
    print("[generate_splits] OK", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
