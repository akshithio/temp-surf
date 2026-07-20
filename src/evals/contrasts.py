"""Stage 5: paired label-access post-processing.

PURE post-processing on completed probe-result rows -- no fitting, inference, embedding generation, or
split generation. Computes the six canonical label-access contrasts (``LABEL_ACCESS_CONTRASTS``), pairing
route values EXACTLY within one (benchmark, model, probe_family, metric, seed, geographic target) cell on
the frozen ``target_test`` evaluation, then aggregates targets with EQUAL region weight within each seed
and reports across-seed mean / std / seed-count / bootstrap CI. The ``source_only`` complete-target
diagnostic is excluded from every contrast; the ``source_ID_reference`` anchor is resolved from the
canonical full-source ``random_id`` result. Missing or duplicate operands are a hard error -- an
incomplete pair is never silently dropped.
"""

from __future__ import annotations

import csv
import hashlib
import io
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from evals import split_artifacts as SA

#: The two Stage-5 artifacts, written under the run's data/output results dir (never data/results).
CONTRAST_FILE = "label_access_contrasts.csv"                  # per-target, per-seed paired values + diffs
CONTRAST_SUMMARY_FILE = "label_access_contrasts_summary.csv"  # equal-region within-seed, then across-seed

#: The repository's confidence-interval convention: a percentile bootstrap (2.5 / 97.5). Here the unit of
#: replication is the SEED -- each seed is a complete independent evaluation -- so the across-seed CI
#: resamples the per-seed (equal-region) values with replacement. Fixed seed => byte-reproducible.
_CI_BOOT = 2000
_CI_SEED = 0
#: Accurate name for the across-seed uncertainty: a hierarchical bootstrap that resamples SEEDS then, in
#: every drawn seed, that seed's TARGET-region rows -- so it captures both seed and target variance.
_CI_CONVENTION = "target_region_bootstrap_2.5_97.5"

#: EXPLICIT, exhaustive metric-direction policy (no substring inference, no "unknown -> higher"). Every
#: metric in the METRICS_* universe is classified as higher_is_better, lower_is_better, or structural
#: (a bookkeeping count with no performance direction). A test asserts the universe is fully covered.
_METRIC_DIRECTION: dict[str, str] = {
    # higher is better -- accuracy / F1 / AUC / IoU families, incl. worst-group + shared variants
    **{m: "higher_is_better" for m in (
        "f1", "auc", "balanced_accuracy", "calibrated_f1", "calibrated_balanced_accuracy",
        "worst_group_f1", "worst_group_balanced_accuracy", "worst_group_calibrated_f1",
        "worst_group_calibrated_balanced_accuracy", "worst_group_score",
        "macro_f1", "weighted_f1", "accuracy", "macro_auc",
        "worst_group_macro_f1", "worst_group_weighted_f1", "worst_group_accuracy",
        "shared_macro_f1", "shared_balanced_accuracy", "shared_accuracy",
        "miou", "pixel_accuracy", "mean_per_tile_miou", "worst_tile_miou",
    )},
    # lower is better -- calibration error, plus target-class coverage FAILURE (more unseen = worse)
    **{m: "lower_is_better" for m in (
        "ece", "brier", "nll", "top_label_ece_all", "union_brier",
        "shared_ece", "shared_nll", "shared_brier",
        "unseen_prevalence", "n_classes_unseen",
    )},
    # structural / count -- no performance direction (bookkeeping)
    **{m: "structural" for m in ("n_classes_seen", "n_tiles_scored")},
}


class ContrastError(RuntimeError):
    """A label-access contrast cannot be computed: a missing or duplicate operand, an unresolvable
    ``source_ID_reference`` anchor, or a MISSING / MALFORMED metric value on an operand. Raised (never
    silently skipped or turned into a NaN) so an incomplete or corrupt contrast set fails loudly."""


def metric_direction(metric: str) -> str:
    """The metric's optimization direction from the EXPLICIT policy. Unknown metrics return ``unknown``
    (NEVER defaulted to higher_is_better) so a newly added metric cannot silently acquire a wrong sign."""
    return _METRIC_DIRECTION.get(str(metric), "unknown")


def _metric_universe() -> set[str]:
    from evals.evals import METRICS_BINARY, METRICS_MULTICLASS, METRICS_SEGMENTATION

    return set(METRICS_BINARY) | set(METRICS_MULTICLASS) | set(METRICS_SEGMENTATION)


def _num(v: Any) -> float:
    """Lenient float for BUDGETS / anchor resolution (never metric values)."""
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def _metric_value(row: dict[str, Any], metric: str, where: str) -> float:
    """The operand's metric value as a strict float. A MISSING key or a MALFORMED (non-numeric) value is
    a hard :class:`ContrastError` -- never a silent NaN. A genuine numeric NaN/Inf emitted by metric
    computation (and round-tripped through JSON as ``float('nan')`` / ``float('inf')`` or ``"nan"`` /
    ``"inf"``) is a VALID value and kept as-is."""
    if metric not in row:
        raise ContrastError(f"{where}: operand is missing metric {metric!r}")
    v = row[metric]
    if isinstance(v, bool):
        raise ContrastError(f"{where}: metric {metric!r} value {v!r} is a boolean, not a number")
    if isinstance(v, (int, float)):
        return float(v)  # includes genuine nan / inf
    if isinstance(v, str):
        try:
            return float(v)  # "0.5" / "nan" / "inf" ok; "abc" is malformed
        except ValueError:
            raise ContrastError(f"{where}: metric {metric!r} value {v!r} is malformed (non-numeric)") from None
    raise ContrastError(f"{where}: metric {metric!r} value {v!r} (type {type(v).__name__}) is not numeric")


def contrast_artifact_names() -> tuple[str, str]:
    """The two Stage-5 artifact filenames, for completion + validation cross-checks."""
    return (CONTRAST_FILE, CONTRAST_SUMMARY_FILE)


def _ibudget(v: Any) -> int:
    try:
        return int(v)
    except (TypeError, ValueError):
        return 0


def has_label_access(rows: list[dict[str, Any]]) -> bool:
    """True iff the rows contain the geographic_ood label-access suite (target_test evaluation)."""
    return any(
        r.get("budget_type") == "label_access"
        and r.get("split_regime") == SA.LABEL_ACCESS_REGIME
        and r.get("evaluation_split") == SA.EVAL_TARGET_TEST
        for r in rows
    )


def _full_source_budget(source_rows: list[dict[str, Any]]) -> float:
    """The canonical full-source budget among random_id source rows: the one closest to 1.0 (ties ->
    the larger). Matches ``evals.evals._id_source_budget``."""
    budgets = [_num(r.get("label_budget")) for r in source_rows]
    return min(budgets, key=lambda b: (abs(b - 1.0), -b))


def _resolve_anchors(rows: list[dict[str, Any]]) -> dict[tuple, dict[str, Any]]:
    """Resolve the ``source_ID_reference`` anchor -- the canonical full-source ``random_id`` in-
    distribution result -- for each (benchmark, model, probe_family, seed). EXACTLY one row per key or a
    hard error (0 => missing, >1 => duplicate)."""
    src = [
        r for r in rows
        if r.get("split_regime") == "random_id" and r.get("budget_type") == "source"
        and r.get("evaluation_split") == "test"
    ]
    if not src:
        return {}
    full = _full_source_budget(src)
    by_key: dict[tuple, list[dict[str, Any]]] = defaultdict(list)
    for r in src:
        if abs(_num(r.get("label_budget")) - full) < 1e-9:
            by_key[(r.get("benchmark"), r.get("model"), r.get("probe_family"), r.get("seed"))].append(r)
    resolved: dict[tuple, dict[str, Any]] = {}
    for key, rs in by_key.items():
        if len(rs) != 1:
            raise ContrastError(
                f"source_ID_reference anchor for {key} is not unique: found {len(rs)} random_id "
                f"full-source (budget={full}) rows -- require exactly one"
            )
        resolved[key] = rs[0]
    return resolved


def _cell_key(r: dict[str, Any]) -> tuple:
    return (r.get("benchmark"), r.get("model"), r.get("probe_family"), r.get("seed"), r.get("holdout"))


def _index_routes(la_rows: list[dict[str, Any]]) -> dict[tuple, dict[tuple, dict[str, Any]]]:
    """cell -> {(route, budget): row}. A repeated (route, budget) within a cell is a DUPLICATE operand
    (hard error) -- completeness alone would not catch two rows that share the 9-field key path."""
    cells: dict[tuple, dict[tuple, dict[str, Any]]] = defaultdict(dict)
    for r in la_rows:
        cell = _cell_key(r)
        rk = (r.get("label_access_route"), _ibudget(r.get("label_budget")))
        if rk in cells[cell]:
            raise ContrastError(f"duplicate label-access operand {rk} in cell {cell}")
        cells[cell][rk] = r
    return cells


def _require(by: dict[tuple, dict[str, Any]], rk: tuple, cell: tuple, contrast: str) -> dict[str, Any]:
    row = by.get(rk)
    if row is None:
        raise ContrastError(f"missing operand {rk} for contrast {contrast!r} in cell {cell}")
    return row


def _paired_row(*, contrast, cell, metric, budget, minuend_route, subtrahend_route, m_row, s_row):
    bench, model, probe, seed, holdout = cell
    # BOTH operands must carry a numeric value for this metric, or it is a hard error (never a silent NaN).
    mv = _metric_value(m_row, metric, f"{contrast}/{minuend_route}@{holdout}/seed={seed}")
    sv = _metric_value(s_row, metric, f"{contrast}/{subtrahend_route}@{holdout}/seed={seed}")
    return {
        # provenance: both route names, both raw values, subtraction order, metric + direction, budget,
        # seed, target, benchmark, model, probe family (item 8).
        "contrast": contrast, "benchmark": bench, "model": model, "probe_family": probe,
        "metric": metric, "metric_direction": metric_direction(metric),
        "seed": seed, "target": holdout,
        "split_regime": SA.LABEL_ACCESS_REGIME, "evaluation_split": SA.EVAL_TARGET_TEST,
        "budget": int(budget),
        "minuend_route": minuend_route, "subtrahend_route": subtrahend_route,
        "subtraction_order": f"{minuend_route} - {subtrahend_route}",
        "minuend_value": mv, "subtrahend_value": sv, "difference": mv - sv,
    }


def _resolve_full_source(rows: list[dict[str, Any]]) -> dict[tuple, dict[str, Any]]:
    """The ordinary full-source geographic row (E1) per label-access cell: ``budget_type=source``,
    ``label_budget=1.0``, ``evaluation_split=test`` under geographic_ood. This is the subtrahend the
    label-access contrasts use INSTEAD of refitting a source-only probe, so a duplicate would silently
    make the choice of leg arbitrary -- it is a hard error."""
    out: dict[tuple, dict[str, Any]] = {}
    for r in rows:
        if (r.get("budget_type") != "source" or r.get("split_regime") != SA.LABEL_ACCESS_REGIME
                or r.get("evaluation_split") != "test"):
            continue
        try:
            if float(r.get("label_budget")) != 1.0:
                continue
        except (TypeError, ValueError):
            continue
        key = _cell_key(r)
        if key in out:
            raise ContrastError(f"duplicate full-source geographic (E1) row for {key}")
        out[key] = r
    return out


def compute_contrasts(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return ``(paired_rows, summary_rows)``. ``paired_rows`` is the per-target, per-seed contrast value
    for every (contrast, metric) pair; ``summary_rows`` aggregates targets with EQUAL region weight within
    each seed, then reports across-seed mean/std/n_seeds/CI. Hard-fails (ContrastError) on any missing or
    duplicate operand and on an unresolvable anchor -- the incomplete pair is never silently dropped."""
    # Frozen target_test rows ONLY -- the canonical comparison population.
    la_rows = [
        r for r in rows
        if r.get("budget_type") == "label_access" and r.get("split_regime") == SA.LABEL_ACCESS_REGIME
        and r.get("evaluation_split") == SA.EVAL_TARGET_TEST
    ]
    if not la_rows:
        return [], []
    metrics = sorted(_metric_universe() & {k for r in la_rows for k in r})
    cells = _index_routes(la_rows)
    anchors = _resolve_anchors(rows)
    e1 = _resolve_full_source(rows)

    paired: list[dict[str, Any]] = []
    for cell in sorted(cells, key=lambda c: tuple(str(x) for x in c)):
        by = cells[cell]
        for metric in metrics:
            for name, minuend_route, subtrahend_route in SA.LABEL_ACCESS_CONTRASTS:
                if subtrahend_route == SA.ALLOCATION_BASELINE:
                    # The allocation EFFECT is measured against the curve's OWN f=0 endpoint (B_d source
                    # units), not against E1 (the complete source pool) -- both legs must hold the same
                    # total budget or the contrast confounds budget size with composition.
                    s_row = _require(by, (minuend_route, 0), cell, name)
                    for f in SA.ALLOCATION_PERCENTS:
                        if int(f) == 0:
                            continue
                        m_row = _require(by, (minuend_route, int(f)), cell, name)
                        paired.append(_paired_row(contrast=name, cell=cell, metric=metric, budget=int(f),
                                                  minuend_route=minuend_route,
                                                  subtrahend_route=SA.ALLOCATION_BASELINE,
                                                  m_row=m_row, s_row=s_row))
                elif subtrahend_route == SA.ANCHOR_GEOGRAPHIC_FULL_SOURCE:
                    # Subtract the ordinary full-source geographic row (E1). Label access does not refit
                    # a source-only probe, so this is the one complete-source leg for the whole suite.
                    base = e1.get(cell)
                    if base is None:
                        raise ContrastError(
                            f"missing full-source geographic (E1) row for {cell} -- budget_type=source, "
                            f"label_budget=1.0, evaluation_split=test is required for contrast {name!r}"
                        )
                    if name == "additive_target_label_gain":
                        for k in SA.LABEL_ACCESS_COUNTS:
                            m_row = _require(by, (minuend_route, int(k)), cell, name)
                            paired.append(_paired_row(contrast=name, cell=cell, metric=metric, budget=int(k),
                                                      minuend_route=minuend_route,
                                                      subtrahend_route=subtrahend_route,
                                                      m_row=m_row, s_row=base))
                    else:
                        m_row = _require(by, (minuend_route, 0), cell, name)
                        paired.append(_paired_row(contrast=name, cell=cell, metric=metric, budget=0,
                                                  minuend_route=minuend_route,
                                                  subtrahend_route=subtrahend_route,
                                                  m_row=m_row, s_row=base))
                elif minuend_route == SA.ANCHOR_SOURCE_ID_REFERENCE:
                    anchor = anchors.get(cell[:4])
                    if anchor is None:
                        raise ContrastError(
                            f"missing source_ID_reference anchor for {cell[:4]} -- the random_id full-source "
                            f"result is required for contrast {name!r}"
                        )
                    s_row = _require(by, (subtrahend_route, 0), cell, name)
                    paired.append(_paired_row(contrast=name, cell=cell, metric=metric, budget=0,
                                              minuend_route=minuend_route, subtrahend_route=subtrahend_route,
                                              m_row=anchor, s_row=s_row))
                else:
                    m_row = _require(by, (minuend_route, 0), cell, name)
                    s_row = _require(by, (subtrahend_route, 0), cell, name)
                    paired.append(_paired_row(contrast=name, cell=cell, metric=metric, budget=0,
                                              minuend_route=minuend_route, subtrahend_route=subtrahend_route,
                                              m_row=m_row, s_row=s_row))
    return paired, _aggregate(paired)


def _region_ci(target_means: np.ndarray) -> tuple[float, float]:
    """Percentile bootstrap over TARGET REGIONS -- the unit of generalization.

    The input is one value per target region, already averaged across seeds. Regions are resampled with
    replacement; seeds are NOT resampled, because seed noise is a property of the estimator rather than
    of the population we generalize to, and is reported separately as ``std_across_seeds``. Bootstrapping
    seeds jointly (the previous seed-first nested convention) mixed the two and let three seeds of one
    region masquerade as three independent observations. Fixed seed -> deterministic."""
    n = int(target_means.size)
    if n < SA.MIN_HEADLINE_TARGETS:
        return float("nan"), float("nan")
    rng = np.random.default_rng(_CI_SEED)
    boot = np.empty(_CI_BOOT)
    for b in range(_CI_BOOT):
        boot[b] = target_means[rng.integers(0, n, n)].mean()
    return float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def _aggregate(paired: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate paired contrasts: pair within (target, seed), average seeds WITHIN each target, then
    bootstrap target regions. Seed variation is reported separately rather than folded into the CI. A
    benchmark aggregate resting on fewer than ``MIN_HEADLINE_TARGETS`` regions is refused a headline
    interval (NaN CI + ``headline=False``) instead of publishing an interval no one should read."""
    groups: dict[tuple, dict[Any, dict[Any, list[float]]]] = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    prov: dict[tuple, dict[str, Any]] = {}
    for r in paired:
        gkey = (r["benchmark"], r["model"], r["probe_family"], r["metric"], r["contrast"], r["budget"])
        groups[gkey][r["target"]][r["seed"]].append(r["difference"])
        prov.setdefault(gkey, r)
    out: list[dict[str, Any]] = []
    for gkey in sorted(groups, key=lambda g: tuple(str(x) for x in g)):
        by_target = groups[gkey]
        targets = sorted(by_target, key=lambda t: str(t))
        # 1) within a target: one value per seed, then 2) the seed-average for that target
        per_target_seed = {
            t: {s: float(np.mean(v)) for s, v in by_target[t].items()} for t in targets
        }
        target_means = np.asarray(
            [float(np.mean(list(per_target_seed[t].values()))) for t in targets], dtype=float
        )
        # seed variation, reported SEPARATELY: spread of the equal-region seed means
        seeds = sorted({s for t in targets for s in per_target_seed[t]}, key=lambda s: (s is None, s))
        per_seed = [
            float(np.mean([per_target_seed[t][s] for t in targets if s in per_target_seed[t]]))
            for s in seeds
        ]
        headline = len(targets) >= SA.MIN_HEADLINE_TARGETS
        lo, hi = _region_ci(target_means)
        r0 = prov[gkey]
        bench, model, probe, metric, contrast, budget = gkey
        out.append({
            "contrast": contrast, "benchmark": bench, "model": model, "probe_family": probe,
            "metric": metric, "metric_direction": metric_direction(metric), "budget": int(budget),
            "minuend_route": r0["minuend_route"], "subtrahend_route": r0["subtrahend_route"],
            "subtraction_order": r0["subtraction_order"], "region_weighting": "equal",
            "mean_difference": float(np.mean(target_means)),
            "std_across_seeds": float(np.std(per_seed)) if per_seed else float("nan"),
            "std_across_targets": float(np.std(target_means)),
            "n_seeds": len(seeds), "n_targets": len(targets), "headline": bool(headline),
            "ci_convention": _CI_CONVENTION, "ci_lo": lo, "ci_hi": hi,
        })
    return out


# --------------------------------------------------------------------------- #
# Deterministic serialization + write + validate (finalization / completion)
# --------------------------------------------------------------------------- #
def _csv_bytes(rows: list[dict[str, Any]]) -> bytes:
    """Deterministic CSV bytes (fixed column order from the first row; every row shares the same keys).
    Recomputing from the same probe rows reproduces these bytes exactly, so a byte comparison is the
    staleness / inconsistency check."""
    if not rows:
        return b""
    fieldnames = list(rows[0].keys())
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=fieldnames, lineterminator="\n")
    w.writeheader()
    w.writerows(rows)
    return buf.getvalue().encode()


def write_contrasts(results_dir: str | Path, paired: list[dict[str, Any]], summary: list[dict[str, Any]]) -> None:
    """Write the two contrast artifacts under the run's data/output results dir (never data/results)."""
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / CONTRAST_FILE).write_bytes(_csv_bytes(paired))
    (results_dir / CONTRAST_SUMMARY_FILE).write_bytes(_csv_bytes(summary))


def compute_and_write(results_dir: str | Path, rows: list[dict[str, Any]]) -> None:
    """Finalization entry point: (re)compute contrasts from the probe rows and write both artifacts.
    Hard-fails on any missing/duplicate operand or unresolved anchor. No-op when the run has no
    label-access rows."""
    if not has_label_access(rows):
        return
    paired, summary = compute_contrasts(rows)
    write_contrasts(results_dir, paired, summary)


def validate_written_contrasts(
    results_dir: str | Path, rows: list[dict[str, Any]]
) -> tuple[list[str], dict[str, dict[str, Any]]]:
    """Completion-time check: recompute the contrasts from the probe rows and require the ON-DISK
    artifacts to match byte-for-byte. Returns ``(problems, artifact_hashes)``. A problem is reported when
    an artifact is missing, or is stale / duplicated / inconsistent with probe_results.jsonl, or the
    contrasts cannot be computed at all. No-op (empty) when the run has no label-access rows."""
    results_dir = Path(results_dir)
    if not has_label_access(rows):
        return [], {}
    try:
        paired, summary = compute_contrasts(rows)
    except ContrastError as exc:
        return [f"label-access contrasts could not be recomputed from probe_results.jsonl: {exc}"], {}
    problems: list[str] = []
    hashes: dict[str, dict[str, Any]] = {}
    for name, recomputed in ((CONTRAST_FILE, paired), (CONTRAST_SUMMARY_FILE, summary)):
        path = results_dir / name
        expected = _csv_bytes(recomputed)
        if not path.exists():
            problems.append(f"{name} is missing")
            continue
        actual = path.read_bytes()
        if actual != expected:
            problems.append(f"{name} is stale, duplicated, or inconsistent with probe_results.jsonl")
        hashes[name] = {"sha256": hashlib.sha256(actual).hexdigest(), "bytes": len(actual)}
    return problems, hashes
