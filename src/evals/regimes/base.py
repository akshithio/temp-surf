"""Shared types for split regimes.

A *regime* owns two things:

    (1) the domain basis: how each sample is assigned a domain label
    (2) the split strategy — how those domains become train/val/test
"""

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


@dataclass(frozen=True)
class Split:
    """One train/val/test partition produced by a regime.

    ``label`` identifies the held-out domain (or fold). ``val`` may be empty when a
    regime trains on the full non-target pool and leaves threshold calibration to
    the probe's own internal split. ``domain`` is the raw domain value held out (e.g. the
    region ``"Estonia"`` behind label ``"Estonia"``); the runner uses it to detect a
    leave-one-domain-out regime that silently dropped a domain. Defaults to ``label``.
    """

    label: str
    train: np.ndarray
    test: np.ndarray
    val: np.ndarray = field(default_factory=_empty)
    domain: str | None = None


def geography_domains(bench) -> np.ndarray:
    """Default domain assignment: the benchmark's native region/source groups."""
    return np.asarray(bench.groups, dtype=object)


REGIME_PROBLEMS: list[tuple[str, str, str]] = []


def clear_regime_problems() -> None:
    REGIME_PROBLEMS.clear()


def load_regime(regime_name: str):
    """Import a split-regime module."""
    return importlib.import_module(f"evals.regimes.{regime_name}")


def regime_problem(benchmark: str, regime: str, reason: str, *, overwrite_mode: bool) -> None:
    """Surface a declared regime that did not run."""
    REGIME_PROBLEMS.append((benchmark, regime, reason))
    if overwrite_mode:
        raise RuntimeError(f"declared regime did not run -- {benchmark}/{regime}: {reason}")
    bar = "!" * 78
    print(
        f"\n{bar}\n!! REGIME DECLARED BUT DID NOT RUN -- {benchmark}/{regime}\n!! {reason}"
        f"\n!! (OVERWRITE_MODE is False for this run; it would be a hard failure with OVERWRITE_MODE=True)\n{bar}\n",
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


def iter_splits(split_regime, bench, y, holdouts, seed, *, overwrite_mode: bool, val_group=None):
    """Yield split metadata and regime-assigned domain labels."""
    regime = load_regime(split_regime)
    bench_name = getattr(bench, "name", "?")
    try:
        domains = np.asarray(regime.assign_domains(bench), dtype=object)
    except Exception as exc:
        regime_problem(
            bench_name,
            split_regime,
            f"domain assignment failed ({type(exc).__name__}: {exc})",
            overwrite_mode=overwrite_mode,
        )
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
    yielded_labels: set[str] = set()
    yielded_domains: set[str] = set()
    for split in regime.iter_splits(y, domains, seed=seed, holdouts=holdouts, val_group=val_group):
        n_splits += 1
        yielded_labels.add(str(split.label))
        yielded_domains.add(str(getattr(split, "domain", None) or split.label))
        yield split.label, split.train, split.val, split.test, domains, regime.HAS_TARGET, regime.GROUP_KIND
    if n_splits == 0:
        labels = sorted({str(d) for d in domains})
        shown = labels[:8] + (["..."] if len(labels) > 8 else [])
        regime_problem(
            bench_name,
            split_regime,
            f"produced 0 splits (domain labels seen: {shown})",
            overwrite_mode=overwrite_mode,
        )
    elif getattr(regime, "USES_CURATED_HOLDOUTS", False):
        missing = [str(h) for h in (holdouts or []) if str(h) not in yielded_labels]
        if missing:
            regime_problem(
                bench_name,
                split_regime,
                f"curated holdout(s) dropped (no valid split): {missing}",
                overwrite_mode=overwrite_mode,
            )
    elif getattr(regime, "LEAVE_ONE_DOMAIN_OUT", False):
        attempted = {str(d) for d in domains if str(d) not in ("unknown", "nan")}
        missing = sorted(attempted - yielded_domains)
        if missing:
            regime_problem(
                bench_name,
                split_regime,
                f"domain(s) dropped (no valid split): {missing}",
                overwrite_mode=overwrite_mode,
            )


def segmentation_fold_configs(bench_mod, regimes, *, overwrite_mode: bool):
    """Yield dense fold configs for segmentation regimes."""
    for regime_name in regimes:
        fold_iter = getattr(load_regime(regime_name), "iter_fold_splits", None)
        if fold_iter is None:
            regime_problem(
                getattr(bench_mod, "BENCHMARK", "?"),
                regime_name,
                "no dense (segmentation) realization -- regime exposes no iter_fold_splits",
                overwrite_mode=overwrite_mode,
            )
            continue
        for label, train_folds, val_folds, test_folds in fold_iter(bench_mod):
            yield (regime_name, label, train_folds, val_folds, test_folds)


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
        idx = np.asarray(idx, dtype=np.int64)
        return {
            f"n_{prefix}": int(len(idx)),
            f"{prefix}_domains": sorted({str(v) for v in np.asarray(domains)[idx].tolist()}),
            f"{prefix}_domain_counts": _value_counts(np.asarray(domains)[idx]),
            f"{prefix}_class_counts": _value_counts(np.asarray(labels)[idx]),
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


def _dense_fold_stats(emb_dir, folds: set[int]) -> dict[str, Any]:
    """Exact dense label/domain stats for cached PASTIS fold partitions."""
    class_counts: dict[str, int] = {}
    domain_counts: dict[str, int] = {}
    n_tiles = 0
    patches: set[int] = set()
    for fold in sorted(folds):
        fold_dir = emb_dir / f"fold_{int(fold)}"
        for label_path in sorted(fold_dir.glob("*.labels.npy")):
            labels = np.asarray(np.load(label_path, mmap_mode="r"), dtype=np.int64)
            n_tiles += 1
            patch_id = int(label_path.name.split("_", 1)[0])
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
    add_part(row, "train", _dense_fold_stats(emb_dir, train_folds))
    add_part(row, "val", _dense_fold_stats(emb_dir, val_folds))
    add_part(row, "test", _dense_fold_stats(emb_dir, test_folds))
    return row


def _write_split_manifest(results_dir, rows: list[dict[str, Any]]) -> None:
    """Write the split audit artifact beside the probe outputs."""
    IOU.write_json(results_dir / "split_manifest.json", {"splits": rows})

