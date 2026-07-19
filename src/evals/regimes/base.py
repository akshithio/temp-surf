"""Shared types for split regimes."""

from __future__ import annotations

import importlib
import warnings
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

from utils import ioutils as IOU


def _empty() -> np.ndarray:
    return np.empty(0, dtype=np.int64)


# --------------------------------------------------------------------------- #
# Schema-v2 explicit source/target representation. Every regime emits SourceTargetSplit /
# DenseSourceTargetSplit directly (the overloaded v1 Split/DenseSplit train/val/test contract and its
# iter_splits/segmentation_fold_configs iteration path are gone); the generator serializes these via
# evals.split_artifacts and the runtime consumes them from data/splits/.
# --------------------------------------------------------------------------- #
#: The five explicit v2 partitions. Source partitions are the exact 80/10/10 of the purged source
#: pool; target partitions are the exact 80/20 of the held-out target region.
SOURCE_PARTITIONS: tuple[str, ...] = ("source_train", "source_val", "source_test")
TARGET_PARTITIONS: tuple[str, ...] = ("target_label_pool", "target_test")
V2_PARTITIONS: tuple[str, ...] = SOURCE_PARTITIONS + TARGET_PARTITIONS


def route_partition_problems(
    has_target: bool, supports_target_labels: bool, n_target_label_pool: int, n_target_test: int,
) -> list[str]:
    """The fail-closed route-capability contract between the two flags and the target partition
    sizes. Returns a (possibly empty) list of violations; callers raise their own error type."""
    problems: list[str] = []
    if supports_target_labels and not has_target:
        problems.append("supports_target_labels=True requires has_target=True")
    if not has_target and (n_target_label_pool or n_target_test):
        problems.append("has_target=False requires both target partitions empty")
    if has_target and n_target_test == 0:
        problems.append("has_target=True requires a non-empty target_test")
    if supports_target_labels and n_target_label_pool == 0:
        problems.append("supports_target_labels=True requires a non-empty target_label_pool")
    if not supports_target_labels and n_target_label_pool:
        problems.append("supports_target_labels=False requires an empty target_label_pool")
    return problems


def require_bool_flags(has_target: Any, supports_target_labels: Any) -> None:
    """Reject anything but an exact ``bool`` for the two route flags (no truthy/falsy coercion)."""
    for name, v in (("has_target", has_target), ("supports_target_labels", supports_target_labels)):
        if not isinstance(v, bool):
            raise ValueError(f"{name} must be a bool, got {type(v).__name__}: {v!r}")


def validate_route_partitions(has_target: bool, supports_target_labels: bool, n_pool: int, n_test: int) -> None:
    """Raise ValueError if the route flags are non-bool or the route/partition invariants are violated."""
    require_bool_flags(has_target, supports_target_labels)
    problems = route_partition_problems(has_target, supports_target_labels, n_pool, n_test)
    if problems:
        raise ValueError("route-capability invariants violated: " + "; ".join(problems))


#: The machine-readable role of a target in headline aggregation. ``headline`` targets enter the
#: equal-region mean / worst-region metrics; ``supplementary_stress`` targets (CropHarvest's one-class
#: regions) are held out and scored as SOURCE-ONLY stress evidence but MUST NOT enter the headline
#: mean/worst. This is distinct from ``supports_target_labels`` (official is supports=False yet
#: headline), so aggregation can tell a stress target from a zero-shot release target.
TARGET_ROLE_HEADLINE = "headline"
TARGET_ROLE_SUPPLEMENTARY_STRESS = "supplementary_stress"
TARGET_ROLES = (TARGET_ROLE_HEADLINE, TARGET_ROLE_SUPPLEMENTARY_STRESS)


def validate_target_role(target_role: Any, supports_target_labels: bool) -> None:
    """Reject an unknown role, and enforce that a supplementary stress target draws no target labels."""
    if target_role not in TARGET_ROLES:
        raise ValueError(f"target_role must be one of {TARGET_ROLES}, got {target_role!r}")
    if target_role == TARGET_ROLE_SUPPLEMENTARY_STRESS and supports_target_labels:
        raise ValueError("a supplementary_stress target must have supports_target_labels=False (zero-shot only)")


@dataclass(frozen=True)
class SourceTargetSplit:
    """One realized tabular split (schema v2): explicit partitions + first-class route capabilities.

    ``target_label_pool``/``target_test`` are empty when ``has_target`` is False (source-only, e.g.
    ``random_id``). ``supports_target_labels`` is False for ``official`` (a fixed release evaluation
    set with no label-budget access) even though it HAS a target geography. The route-capability
    invariants are enforced at construction (fail-closed): they cannot silently disagree with the
    target partition sizes.
    """

    label: str
    source_train: np.ndarray
    source_val: np.ndarray
    source_test: np.ndarray
    target_label_pool: np.ndarray = field(default_factory=_empty)
    target_test: np.ndarray = field(default_factory=_empty)
    domain: str | None = None
    has_target: bool = True
    supports_target_labels: bool = True
    group_kind: str = "geography"
    target_role: str = TARGET_ROLE_HEADLINE

    def __post_init__(self) -> None:
        validate_route_partitions(
            self.has_target, self.supports_target_labels, len(self.target_label_pool), len(self.target_test)
        )
        validate_target_role(self.target_role, self.supports_target_labels)

    def as_partitions(self) -> dict[str, np.ndarray]:
        """Partition-name -> index array, in canonical :data:`V2_PARTITIONS` order."""
        return {
            "source_train": self.source_train,
            "source_val": self.source_val,
            "source_test": self.source_test,
            "target_label_pool": self.target_label_pool,
            "target_test": self.target_test,
        }


@dataclass(frozen=True)
class DenseSourceTargetSplit:
    """One realized PASTIS patch-level split (schema v2). Allocation is over patch IDs only; the
    evaluation streams pixels afterwards, but patch membership is immutable and never split. The
    route-capability invariants are enforced at construction, exactly as for the tabular split."""

    label: str
    source_train_patches: frozenset[int]
    source_val_patches: frozenset[int]
    source_test_patches: frozenset[int]
    target_label_pool_patches: frozenset[int] = frozenset()
    target_test_patches: frozenset[int] = frozenset()
    has_target: bool = True
    supports_target_labels: bool = True
    group_kind: str = "geography"
    target_role: str = TARGET_ROLE_HEADLINE

    def __post_init__(self) -> None:
        validate_route_partitions(
            self.has_target, self.supports_target_labels,
            len(self.target_label_pool_patches), len(self.target_test_patches),
        )
        validate_target_role(self.target_role, self.supports_target_labels)

    def as_partitions(self) -> dict[str, frozenset[int]]:
        return {
            "source_train": self.source_train_patches,
            "source_val": self.source_val_patches,
            "source_test": self.source_test_patches,
            "target_label_pool": self.target_label_pool_patches,
            "target_test": self.target_test_patches,
        }


def route_capabilities(regime_module: Any) -> tuple[bool, bool]:
    """``(has_target, supports_target_labels)`` for a regime module -- fail-closed.

    Both ``HAS_TARGET`` and ``SUPPORTS_TARGET_LABELS`` MUST be declared (no default inference), each
    must be a real ``bool``, and ``supports_target_labels=True`` requires ``has_target=True``. Any
    violation raises ValueError so a regime can never silently ship an ambiguous route capability.
    """
    name = getattr(regime_module, "NAME", getattr(regime_module, "__name__", "?"))
    for attr in ("HAS_TARGET", "SUPPORTS_TARGET_LABELS"):
        if not hasattr(regime_module, attr):
            raise ValueError(f"regime {name!r} must declare {attr}")
        if not isinstance(getattr(regime_module, attr), bool):
            raise ValueError(f"regime {name!r}: {attr} must be a bool, got {getattr(regime_module, attr)!r}")
    has_target = regime_module.HAS_TARGET
    supports = regime_module.SUPPORTS_TARGET_LABELS
    if supports and not has_target:
        raise ValueError(f"regime {name!r}: SUPPORTS_TARGET_LABELS=True requires HAS_TARGET=True")
    return has_target, supports


REGIME_PROBLEMS: list[tuple[str, str, str]] = []

#: Per-domain eligibility rows accumulated by leave-one-domain-out regimes, written out as
#: ``domain_census.json`` so that every domain the run considered -- including the ones it
#: excluded, and why -- is auditable from the artifact rather than only from the logs.
DOMAIN_CENSUS: list[dict[str, Any]] = []


def clear_regime_problems() -> None:
    REGIME_PROBLEMS.clear()


def clear_domain_census() -> None:
    DOMAIN_CENSUS.clear()


#: Structured, behavior-neutral audit events emitted from the split-construction path itself
#: (silent stratification fallbacks, dropped folds/holdouts, purges). The split-preprocessing
#: generator clears this before each (regime, seed) call and snapshots it into ``generation.json``
#: / ``manifest.json`` / ``exclusions.csv`` so current imperfect behavior is *recorded exactly*
#: rather than inferred from stdout or class counts. Appending an event NEVER changes split
#: membership; the runtime does not consume this list.
SPLIT_AUDIT_EVENTS: list[dict[str, Any]] = []


def clear_split_audit_events() -> None:
    SPLIT_AUDIT_EVENTS.clear()


def emit_split_audit_event(kind: str, **fields: Any) -> None:
    """Record a behavior-neutral split-construction audit event (see SPLIT_AUDIT_EVENTS)."""
    SPLIT_AUDIT_EVENTS.append({"kind": str(kind), **fields})


def load_regime(regime_name: str):
    """Import a split-regime module."""
    return importlib.import_module(f"evals.regimes.{regime_name}")


def holdouts_for(bench_mod, regime_name: str):
    if regime_name == "official":
        return getattr(bench_mod, "OFFICIAL_HOLDOUTS", getattr(bench_mod, "HOLDOUTS", []))
    if regime_name == "geographic_ood":
        if hasattr(bench_mod, "GEOGRAPHIC_SPLIT"):
            return bench_mod.GEOGRAPHIC_SPLIT
        return getattr(bench_mod, "GEOGRAPHIC_HOLDOUTS", getattr(bench_mod, "HOLDOUTS", []))
    if regime_name == "spatial_cluster_ood":
        # coordinate-only spherical-K-means cells: no curated holdouts, no benchmark override
        return {}
    return getattr(bench_mod, "HOLDOUTS", [])


def val_group_for(bench_mod, regime_name: str):
    if regime_name == "official":
        return getattr(bench_mod, "OFFICIAL_VAL_HOLDOUT", getattr(bench_mod, "VAL_HOLDOUT", None))
    if regime_name == "geographic_ood":
        return getattr(bench_mod, "GEOGRAPHIC_VAL_HOLDOUT", None)
    if regime_name == "spatial_cluster_ood":
        return None
    return None


def regime_problem(benchmark: str, regime: str, reason: str, *, strict_mode: bool) -> None:
    """Surface a declared regime that did not run."""
    REGIME_PROBLEMS.append((benchmark, regime, reason))
    if strict_mode:
        raise RuntimeError(f"declared regime did not run -- {benchmark}/{regime}: {reason}")
    bar = "!" * 78
    print(
        f"\n{bar}\n!! REGIME DECLARED BUT DID NOT RUN -- {benchmark}/{regime}\n!! {reason}"
        f"\n!! (STRICT_MODE is False for this run; it would be a hard failure with STRICT_MODE=True)\n{bar}\n",
        flush=True,
    )


def report_regime_problems() -> None:
    """Print a consolidated list of regimes that were declared but did not run."""
    if not REGIME_PROBLEMS:
        return
    bar = "=" * 78
    print(f"\n{bar}\nREGIMES DECLARED BUT NOT RUN ({len(REGIME_PROBLEMS)}):", flush=True)
    for benchmark, regime, reason in REGIME_PROBLEMS:
        print(f"  - {benchmark}/{regime}: {reason}", flush=True)
    print(f"{bar}\n", flush=True)


# The v1 split-iteration path (iter_splits over regime.assign_domains/iter_splits, plus
# _dense_split_from_tuple / segmentation_fold_configs over regime.iter_dense_splits) has been
# removed. Every regime now emits SourceTargetSplit / DenseSourceTargetSplit directly, the generator
# (tools/generate_splits.py) serializes them via evals.split_artifacts, and a requested regime that
# yields zero leaves is refused at consumption time (split_artifacts.load_*_splits), not surfaced as a
# runtime regime_problem. REGIME_PROBLEMS / regime_problem / report_regime_problems remain the
# run-level "declared regime did not run" channel consumed by main.py / runstate.py.


def _append_prediction_rows(
    predictions: list[dict[str, Any]] | None,
    *,
    meta: dict[str, Any],
    seed: int,
    budget_type: str,
    label_budget: float | int,
    n_train_sub: int,
    sample_ids: np.ndarray,
    groups_test: np.ndarray | None,
    per_sample: dict[str, np.ndarray] | None,
    evaluation_split: str = "test",
) -> None:
    """Append one prediction row per test sample for a completed probe budget cell."""
    if predictions is None or per_sample is None:
        return

    sample_ids = np.asarray(sample_ids)
    groups_arr = np.asarray(groups_test) if groups_test is not None else np.asarray([""] * len(sample_ids))
    base = {
        **meta,
        "budget_type": budget_type,
        "label_budget": label_budget,
        "evaluation_split": evaluation_split,
        "seed": seed,
        "n_train_sub": int(n_train_sub),
    }
    y_true = np.asarray(per_sample["y_true"])
    if "prob" in per_sample:
        prob = np.asarray(per_sample["prob"], dtype=np.float64)
        pred_default = np.asarray(per_sample["pred_default"], dtype=np.int64)
        pred_calibrated = np.asarray(per_sample["pred_calibrated"], dtype=np.int64)
        for i in range(len(y_true)):
            predictions.append({
                **base,
                "sample_id": int(sample_ids[i]),
                "group": str(groups_arr[i]),
                "y_true": int(y_true[i]),
                "prob": float(prob[i]),
                "pred_default": int(pred_default[i]),
                "pred_calibrated": int(pred_calibrated[i]),
            })
        return

    classes = np.asarray(per_sample.get("classes", []), dtype=np.int64)
    proba = np.asarray(per_sample.get("proba", np.zeros((len(y_true), 0))), dtype=np.float64)
    pred = np.asarray(per_sample["pred"], dtype=np.int64)
    class_to_col = {int(c): j for j, c in enumerate(classes)}
    for i in range(len(y_true)):
        pred_col = class_to_col.get(int(pred[i]))
        true_col = class_to_col.get(int(y_true[i]))
        probs = proba[i].astype(float).tolist() if proba.size else []
        predictions.append({
            **base,
            "sample_id": int(sample_ids[i]),
            "group": str(groups_arr[i]),
            "y_true": int(y_true[i]),
            "pred": int(pred[i]),
            "prob_pred": float(proba[i, pred_col]) if pred_col is not None and proba.size else float("nan"),
            "prob_true": float(proba[i, true_col]) if true_col is not None and proba.size else float("nan"),
            "classes": classes.astype(int).tolist(),
            "probs": probs,
        })


def _safe_balanced_accuracy(y_true: np.ndarray, pred: np.ndarray) -> float:
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="y_pred contains classes not in y_true")
        return float(balanced_accuracy_score(y_true, pred))


def _score_binary_group(per_sample: dict[str, np.ndarray], mask: np.ndarray) -> dict[str, float]:
    y = np.asarray(per_sample["y_true"])[mask]
    pred_default = np.asarray(per_sample["pred_default"])[mask]
    pred_cal = np.asarray(per_sample["pred_calibrated"])[mask]
    return {
        "f1": float(f1_score(y, pred_default, zero_division=0)),
        "balanced_accuracy": _safe_balanced_accuracy(y, pred_default),
        "calibrated_f1": float(f1_score(y, pred_cal, zero_division=0)),
        "calibrated_balanced_accuracy": _safe_balanced_accuracy(y, pred_cal),
    }


def _score_multiclass_group(per_sample: dict[str, np.ndarray], mask: np.ndarray) -> dict[str, float]:
    y = np.asarray(per_sample["y_true"])[mask]
    pred = np.asarray(per_sample["pred"])[mask]
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="y_pred contains classes not in y_true")
        return {
            "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
            "weighted_f1": float(f1_score(y, pred, average="weighted", zero_division=0)),
            "balanced_accuracy": _safe_balanced_accuracy(y, pred),
            "accuracy": float(accuracy_score(y, pred)),
        }


def _worst_group_scores(
    per_sample: dict[str, np.ndarray] | None,
    groups_test: np.ndarray | None,
) -> dict[str, Any]:
    """Compute first-class subpopulation worst-group metrics for one evaluated split.

    This captures the "average can hide a bad deployment subgroup" view. For
    ``random_id`` rows it is a true subpopulation metric: train/test include all
    domains, and the row also reports the worst test domain. For strict OOD rows
    with one target domain, worst-group equals the target-domain score.
    """
    if per_sample is None or groups_test is None:
        return {}
    groups_arr = np.asarray(groups_test, dtype=object)
    y_true = np.asarray(per_sample["y_true"])
    if len(groups_arr) != len(y_true) or len(y_true) == 0:
        return {}

    if "prob" in per_sample:
        scorer = _score_binary_group
        metric_names = ["f1", "balanced_accuracy", "calibrated_f1", "calibrated_balanced_accuracy"]
        primary = "calibrated_f1"
    else:
        scorer = _score_multiclass_group
        metric_names = ["macro_f1", "weighted_f1", "balanced_accuracy", "accuracy"]
        primary = "macro_f1"

    by_metric: dict[str, list[tuple[str, float]]] = {name: [] for name in metric_names}
    for group in sorted({str(g) for g in groups_arr.tolist()}):
        mask = np.array([str(g) == group for g in groups_arr], dtype=bool)
        if not mask.any():
            continue
        scores = scorer(per_sample, mask)
        for name, value in scores.items():
            if np.isfinite(value):
                by_metric[name].append((group, value))

    out: dict[str, Any] = {"n_groups_scored": int(len({str(g) for g in groups_arr.tolist()}))}
    for name, values in by_metric.items():
        if not values:
            continue
        group, value = min(values, key=lambda item: item[1])
        out[f"worst_group_{name}"] = float(value)
        out[f"worst_group_name_{name}"] = group
    if primary in by_metric and by_metric[primary]:
        group, value = min(by_metric[primary], key=lambda item: item[1])
        out["worst_group"] = group
        out["worst_group_metric"] = primary
        out["worst_group_score"] = float(value)
    return out


def _value_counts(values: np.ndarray) -> dict[str, int]:
    """Stable JSON-friendly counts for class/domain labels."""
    arr = np.asarray(values)
    if arr.size == 0:
        return {}
    labels, counts = np.unique(arr.astype(str), return_counts=True)
    return {str(label): int(count) for label, count in zip(labels, counts, strict=True)}


def partition_stats(domains: np.ndarray, labels: np.ndarray, idx: np.ndarray) -> dict[str, Any]:
    """Model-agnostic composition of one partition: counts + domain/class breakdowns.

    Neutral by design -- carries no model or partition-name identity -- so both the legacy
    per-model split manifest and the canonical (model-free) split-preprocessing artifacts compute
    partition composition from a single implementation.
    """
    idx = np.asarray(idx, dtype=np.int64)
    dom = np.asarray(domains)[idx]
    lab = np.asarray(labels)[idx]
    return {
        "n": int(len(idx)),
        "domains": sorted({str(v) for v in dom.tolist()}),
        "domain_counts": _value_counts(dom),
        "class_counts": _value_counts(lab),
    }


def _split_manifest_entry(
    *,
    model_name: str,
    benchmark_name: str,
    seed: int,
    split_regime: str,
    domain_basis: str,
    holdout: str,
    train: np.ndarray,
    val: np.ndarray,
    test: np.ndarray,
    domains: np.ndarray,
    labels: np.ndarray,
) -> dict[str, Any]:
    """Describe one tabular split with domain and class composition."""

    def part(prefix: str, idx: np.ndarray) -> dict[str, Any]:
        stats = partition_stats(domains, labels, idx)
        return {
            f"n_{prefix}": stats["n"],
            f"{prefix}_domains": stats["domains"],
            f"{prefix}_domain_counts": stats["domain_counts"],
            f"{prefix}_class_counts": stats["class_counts"],
        }

    row: dict[str, Any] = {
        "model": model_name,
        "benchmark": benchmark_name,
        "seed": int(seed),
        "split_regime": split_regime,
        "domain_basis": domain_basis,
        "holdout": str(holdout),
    }
    row.update(part("train", train))
    row.update(part("val", val))
    row.update(part("test", test))
    return row


def _dense_fold_stats(emb_dir, folds: set[int], patch_ids: set[int] | None = None) -> dict[str, Any]:
    """Exact dense label/domain stats for cached PASTIS fold partitions."""
    class_counts: dict[str, int] = {}
    domain_counts: dict[str, int] = {}
    n_tiles = 0
    patches: set[int] = set()
    wanted = {int(p) for p in patch_ids} if patch_ids is not None else None
    for fold in sorted(folds):
        fold_dir = emb_dir / f"fold_{int(fold)}"
        for label_path in sorted(fold_dir.glob("*.labels.npy")):
            patch_id = int(label_path.name.split("_", 1)[0])
            if wanted is not None and patch_id not in wanted:
                continue
            labels = np.asarray(np.load(label_path, mmap_mode="r"), dtype=np.int64)
            n_tiles += 1
            patches.add(patch_id)
            domain_counts[str(int(fold))] = domain_counts.get(str(int(fold)), 0) + int(len(labels))
            for label, count in _value_counts(labels).items():
                class_counts[label] = class_counts.get(label, 0) + int(count)
    return {
        "n": int(sum(domain_counts.values())),
        "n_tiles": int(n_tiles),
        "n_patches": int(len(patches)),
        "domains": [str(int(fold)) for fold in sorted(folds)],
        "domain_counts": domain_counts,
        "class_counts": class_counts,
    }


def _segmentation_split_manifest_entry(
    *,
    model_name: str,
    benchmark_name: str,
    seed: int,
    split_regime: str,
    holdout: str,
    train_folds: set[int],
    val_folds: set[int],
    test_folds: set[int],
    train_patches: set[int] | None = None,
    val_patches: set[int] | None = None,
    test_patches: set[int] | None = None,
    emb_dir,
) -> dict[str, Any]:
    """Describe one dense fold split with exact cached-pixel class counts."""

    def add_part(row: dict[str, Any], prefix: str, stats: dict[str, Any]) -> None:
        row[f"n_{prefix}"] = stats["n"]
        row[f"n_{prefix}_tiles"] = stats["n_tiles"]
        row[f"n_{prefix}_patches"] = stats["n_patches"]
        row[f"{prefix}_domains"] = stats["domains"]
        row[f"{prefix}_domain_counts"] = stats["domain_counts"]
        row[f"{prefix}_class_counts"] = stats["class_counts"]

    row: dict[str, Any] = {
        "model": model_name,
        "benchmark": benchmark_name,
        "seed": int(seed),
        "split_regime": split_regime,
        "domain_basis": "geography",
        "holdout": str(holdout),
    }
    add_part(row, "train", _dense_fold_stats(emb_dir, train_folds, train_patches))
    add_part(row, "val", _dense_fold_stats(emb_dir, val_folds, val_patches))
    add_part(row, "test", _dense_fold_stats(emb_dir, test_folds, test_patches))
    return row


def _write_split_manifest(results_dir, rows: list[dict[str, Any]]) -> None:
    """Write the split audit artifact beside the probe outputs."""
    IOU.write_json(results_dir / "split_manifest.json", {"splits": rows})


def _write_domain_census(results_dir, benchmark: str | None = None) -> None:
    """Write the domain eligibility census beside the probe outputs.

    The census is a property of the data, so it is identical across seeds; rows are deduplicated
    on (benchmark, regime, domain). Regimes that do not do leave-one-domain-out contribute
    nothing and no file is produced.

    ``benchmark`` filters to the pair being written. DOMAIN_CENSUS is a process-global
    accumulator, so without this a run covering several benchmarks would write the first
    benchmark's domains into the next benchmark's results directory -- an artifact describing
    data the directory does not contain.
    """
    if not DOMAIN_CENSUS:
        return
    rows: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in DOMAIN_CENSUS:
        if benchmark is not None and str(row["benchmark"]) != str(benchmark):
            continue
        rows[(str(row["benchmark"]), str(row["regime"]), str(row["domain"]))] = row
    if not rows:
        return
    ordered = [rows[k] for k in sorted(rows)]
    IOU.write_json(
        results_dir / "domain_census.json",
        {
            "domains": ordered,
            "n_domains": len(ordered),
            "n_valid_targets": sum(1 for r in ordered if r.get("valid_target")),
            "n_one_class": sum(1 for r in ordered if r.get("one_class")),
            "excluded": [r["domain"] for r in ordered if not r.get("valid_target")],
        },
    )
