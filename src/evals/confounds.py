from __future__ import annotations

from typing import Any

import numpy as np

from evals.regimes.base import TARGET_ROLE_HEADLINE, TARGET_ROLE_SUPPLEMENTARY_STRESS
from utils.ioutils import (
    _LOWER_BETTER,
    _chance,
    _close,
    _sample_delta_ci,
)


def _entropy(counts: np.ndarray) -> float:
    total = counts.sum()
    if total <= 0:
        return 0.0
    p = counts[counts > 0] / total
    return float(-(p * np.log(p)).sum())


def _contingency(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ua, ai = np.unique(np.asarray(a).astype(str), return_inverse=True)
    ub, bi = np.unique(np.asarray(b).astype(str), return_inverse=True)
    counts = np.zeros((len(ua), len(ub)), dtype=np.int64)
    np.add.at(counts, (ai, bi), 1)
    return ua, ub, counts


def confound_pair(a: np.ndarray, b: np.ndarray, max_cells: int = 30) -> dict:
    """Entanglement stats between two per-sample label axes."""
    ua, ub, counts = _contingency(a, b)
    n = int(counts.sum())
    base = {"n_a": int(len(ua)), "n_b": int(len(ub))}
    if n == 0 or len(ua) < 2 or len(ub) < 2:
        return {**base, "nmi": 0.0, "determines_b_given_a": 0.0, "determines_a_given_b": 0.0}
    h_a = _entropy(counts.sum(1))
    h_b = _entropy(counts.sum(0))
    h_ab = _entropy(counts.reshape(-1))
    mi = h_a + h_b - h_ab
    nmi = mi / np.sqrt(h_a * h_b) if h_a > 0 and h_b > 0 else 0.0
    det_b = 1.0 - (h_ab - h_a) / h_b if h_b > 0 else 0.0  # how much a determines b
    det_a = 1.0 - (h_ab - h_b) / h_a if h_a > 0 else 0.0  # how much b determines a
    out = {
        **base,
        "nmi": round(float(np.clip(nmi, 0.0, 1.0)), 4),
        "determines_b_given_a": round(float(np.clip(det_b, 0.0, 1.0)), 4),
        "determines_a_given_b": round(float(np.clip(det_a, 0.0, 1.0)), 4),
    }
    if len(ua) <= max_cells and len(ub) <= max_cells:
        out["contingency"] = {
            str(av): {str(bv): int(counts[i, j]) for j, bv in enumerate(ub) if counts[i, j]} for i, av in enumerate(ua)
        }
    return out


def domain_confound_report(axes: dict[str, np.ndarray | None]) -> dict:
    """Pairwise confound stats between usable per-sample domain axes."""
    usable = [
        name
        for name, values in axes.items()
        if values is not None and len(np.unique(np.asarray(values).astype(str))) > 1
    ]
    report: dict = {
        "n_samples": int(len(axes[usable[0]])) if usable else 0,
        "axis_cardinality": {name: int(len(np.unique(np.asarray(axes[name]).astype(str)))) for name in usable},
        "pairs": {},
    }
    for i in range(len(usable)):
        for j in range(i + 1, len(usable)):
            report["pairs"][f"{usable[i]}__vs__{usable[j]}"] = confound_pair(axes[usable[i]], axes[usable[j]])
    return report


def compute_deltas(
    rows: list[dict[str, Any]],
    metrics: list[str],
    *,
    predictions: list[dict[str, Any]] | None = None,
    id_source_budget: float | int = 1.0,
    ood_target_budget: float | int = 0.0,
    target_id_budget: float | int | None = -1.0,
    n_boot: int = 2000,
    n_boot_sample: int = 1000,
    seed: int = 0,
) -> list[dict[str, Any]]:
    """ID-to-OOD deployment-gap rows for configured metrics."""
    if isinstance(metrics, str):
        metrics = [metrics]
    rng = np.random.default_rng(seed)
    keys = (
        ("model", "benchmark", "method", "probe_family")
        if any("probe_family" in r for r in rows)
        else ("model", "benchmark", "method")
    )

    def _matches(r, split_regime, budget_type, budget, combo, metric, es=None):
        if es is not None:
            actual = r.get("evaluation_split")
            if actual is None:
                actual = "held_out" if budget_type == "target" else "test"
            if actual != es:
                return False
        return (
            r.get("split_regime") == split_regime
            and r.get("budget_type") == budget_type
            and _close(r.get("label_budget"), budget)
            and metric in r
            and r[metric] is not None
            # headline equal-region aggregation NEVER includes supplementary stress targets
            # (CropHarvest one-class regions); they stay visible as source-only stress results elsewhere.
            and r.get("target_role", TARGET_ROLE_HEADLINE) != TARGET_ROLE_SUPPLEMENTARY_STRESS
            and all(r.get(k) == combo[i] for i, k in enumerate(keys))
        )

    def _eval_matches(row: dict[str, Any], accepted: set[str], *, budget_type: str) -> bool:
        actual = row.get("evaluation_split")
        if actual is None:
            actual = "held_out" if budget_type == "target" else "test"
        return str(actual) in accepted

    def vals(split_regime, budget_type, budget, combo, metric, es=None) -> list[float]:
        out: list[float] = []
        for r in rows:
            if _matches(r, split_regime, budget_type, budget, combo, metric, es=es):
                v = float(r[metric])
                if np.isfinite(v):
                    out.append(v)
        return out

    def vals_by_region(split_regime, budget_type, budget, combo, metric, es=None):
        out: dict[Any, list[float]] = {}
        for r in rows:
            if _matches(r, split_regime, budget_type, budget, combo, metric, es=es):
                v = float(r[metric])
                if np.isfinite(v):
                    out.setdefault(r.get("holdout"), []).append(v)
        return out

    def _first_vals(split_regime, budget_type, budget, combo, metric, eval_splits: tuple[str, ...]) -> list[float]:
        for eval_split in eval_splits:
            found = vals(split_regime, budget_type, budget, combo, metric, es=eval_split)
            if found:
                return found
        return []

    def _first_vals_by_region(split_regime, budget_type, budget, combo, metric, eval_splits: tuple[str, ...]):
        for eval_split in eval_splits:
            found = vals_by_region(split_regime, budget_type, budget, combo, metric, es=eval_split)
            if found:
                return found
        return {}

    pred_by: dict[tuple, dict[str, list]] = {}
    for p in predictions or []:
        pred_by.setdefault(tuple(p.get(k) for k in keys), {}).setdefault(p.get("split_regime"), []).append(p)

    combos = sorted({tuple(r.get(k) for k in keys) for r in rows}, key=lambda t: tuple(str(x) for x in t))
    out_rows: list[dict[str, Any]] = []
    for combo in combos:
        id_stat = [
            r
            for r in rows
            if r.get("split_regime") == "random_id"
            and r.get("budget_type") == "source"
            and _close(r.get("label_budget"), id_source_budget)
            and _eval_matches(r, {"test"}, budget_type="source")
            and all(r.get(k) == combo[i] for i, k in enumerate(keys))
        ]

        def _avg(field: str, id_stat=id_stat):
            xs = [float(r[field]) for r in id_stat if r.get(field) is not None]
            return float(np.mean(xs)) if xs else None

        pos_rate, n_cls, majority = _avg("test_pos_rate"), _avg("test_n_classes"), _avg("test_majority_rate")

        id_preds = [
            p
            for p in pred_by.get(combo, {}).get("random_id", [])
            if p.get("budget_type") == "source"
            and _close(p.get("label_budget"), id_source_budget)
            and _eval_matches(p, {"test"}, budget_type="source")
        ]
        _ood_preds_all = [
            p
            for p in pred_by.get(combo, {}).get("geographic_ood", [])
            if p.get("budget_type") == "target" and _close(p.get("label_budget"), ood_target_budget)
        ]
        _ood_full = [p for p in _ood_preds_all if _eval_matches(p, {"full"}, budget_type="target")]
        _ood_test = [p for p in _ood_preds_all if _eval_matches(p, {"test"}, budget_type="target")]
        ood_preds = _ood_full or _ood_test or _ood_preds_all

        for metric in metrics:
            id_vals = vals("random_id", "source", id_source_budget, combo, metric, es="test")
            ood_vals = _first_vals(
                "geographic_ood", "target", ood_target_budget, combo, metric, ("full", "test", "held_out")
            )
            if not id_vals or not ood_vals:
                continue
            idm, oodm = float(np.mean(id_vals)), float(np.mean(ood_vals))
            id_arr, ood_arr = np.asarray(id_vals), np.asarray(ood_vals)
            ood_by_region = _first_vals_by_region(
                "geographic_ood", "target", ood_target_budget, combo, metric, ("full", "test", "held_out")
            )
            if n_boot and ood_by_region:
                region_vals = [np.asarray(v) for v in ood_by_region.values()]
                n_reg = len(region_vals)
                boot = np.empty(n_boot)
                for b in range(n_boot):
                    id_mean = id_arr[rng.integers(0, len(id_arr), len(id_arr))].mean()
                    ood_draw = np.concatenate(
                        [
                            region_vals[ri][rng.integers(0, len(region_vals[ri]), len(region_vals[ri]))]
                            for ri in rng.integers(0, n_reg, n_reg)
                        ]
                    )
                    boot[b] = id_mean - ood_draw.mean()
                lo, hi = float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))
            else:
                lo = hi = float("nan")
            row = dict(zip(keys, combo, strict=False))
            row.update(
                {
                    "metric": metric,
                    "id": idm,
                    "ood": oodm,
                    "delta": idm - oodm,
                    "relative_drop": (idm - oodm) / idm if idm > 0 else float("nan"),
                    "delta_ci_lo": lo,
                    "delta_ci_hi": hi,
                    "n_id": len(id_vals),
                    "n_ood": len(ood_vals),
                    "ood_std": float(np.std(ood_arr)),
                    "ood_min": float(ood_arr.min()),
                    "ood_max": float(ood_arr.max()),
                }
            )
            _seed_holdout: dict[int, list[float]] = {}
            for want in ("full", "test", "held_out"):
                for r in rows:
                    if (
                        r.get("split_regime") == "geographic_ood"
                        and r.get("budget_type") == "target"
                        and _close(r.get("label_budget"), ood_target_budget)
                        and metric in r
                        and r[metric] is not None
                        and (r.get("evaluation_split") or "held_out") == want
                        # worst-region NEVER includes a supplementary stress target
                        and r.get("target_role", TARGET_ROLE_HEADLINE) != TARGET_ROLE_SUPPLEMENTARY_STRESS
                        and all(r.get(k) == combo[i] for i, k in enumerate(keys))
                    ):
                        s = r.get("seed")
                        if s is not None:
                            _seed_holdout.setdefault(int(s), []).append(float(r[metric]))
                if _seed_holdout:
                    break
            if _seed_holdout:
                pick_worst = max if metric in _LOWER_BETTER else min
                seed_worst = np.asarray([pick_worst(vs) for vs in _seed_holdout.values()])
                row.update(
                    {
                        "ood_worst_region": float(np.mean(seed_worst)),
                        "ood_worst_region_std": float(np.std(seed_worst)) if len(seed_worst) > 1 else float("nan"),
                    }
                )
            chance = _chance(metric, pos_rate, n_cls, majority)
            row["chance"] = float(chance) if chance is not None else float("nan")
            row["floor_norm_drop"] = (
                (idm - oodm) / (idm - chance) if (chance is not None and (idm - chance) > 1e-9) else float("nan")
            )
            if id_preds and ood_preds:
                lo_s, hi_s, id_pt, ood_pt = _sample_delta_ci(metric, id_preds, ood_preds, n_boot_sample, rng)
                if not np.isnan(id_pt):
                    row.update(
                        {
                            "delta_sample_pt": id_pt - ood_pt,
                            "delta_sample_ci_lo": lo_s,
                            "delta_sample_ci_hi": hi_s,
                            "n_id_samples": len(id_preds),
                            "n_ood_samples": len(ood_preds),
                        }
                    )
            tid_vals = (
                _first_vals("geographic_ood", "target", target_id_budget, combo, metric, ("held_out", "test"))
                if target_id_budget is not None
                else []
            )
            if tid_vals:
                tidm = float(np.mean(tid_vals))
                ood_matched_vals = _first_vals(
                    "geographic_ood", "target", ood_target_budget, combo, metric, ("held_out", "test")
                )
                ood_matched = float(np.mean(ood_matched_vals)) if ood_matched_vals else oodm
                row.update(
                    {
                        "target_id": tidm,
                        "ood_matched": ood_matched,
                        "inherent_difficulty": idm - tidm,
                        "adjusted_delta": tidm - ood_matched,
                        "adjusted_relative_drop": (tidm - ood_matched) / idm if idm > 0 else float("nan"),
                        "adjusted_floor_norm_drop": (
                            (tidm - ood_matched) / (idm - chance)
                            if (chance is not None and (idm - chance) > 1e-9)
                            else float("nan")
                        ),
                    }
                )
            out_rows.append(row)
    return out_rows
