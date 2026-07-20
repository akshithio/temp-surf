"""Result IO and summary aggregation for experiment runners."""

from __future__ import annotations

import csv
import json
import os
import shutil
import uuid
from pathlib import Path
from typing import Any

import numpy as np

# Metrics where LOWER is better (error/calibration metrics) -- the worst region is their MAX,
# not their min, and the deployment gap reads in the opposite direction.
_LOWER_BETTER = frozenset({"brier", "nll", "ece"})


def hf_download_to(repo_id: str, filename: str, dest: Path, revision: str | None = None) -> Path:
    """Download ``filename`` from ``repo_id`` (pinned to ``revision`` if given) into
    ``dest.parent`` and return ``dest``. Pinning an immutable commit revision makes the
    downloaded bytes reproducible (a moved branch tag could otherwise serve different weights)."""
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise ImportError(
            "huggingface-hub is required to download model weights. Run `uv pip install -e .` "
            "from the project env or set an explicit local weights path."
        ) from exc

    dest = Path(dest).expanduser()
    dest.parent.mkdir(parents=True, exist_ok=True)
    downloaded = Path(
        hf_hub_download(repo_id=repo_id, filename=filename, local_dir=str(dest.parent), revision=revision)
    )
    if downloaded.resolve() != dest.resolve():
        shutil.copy2(downloaded, dest)
    return dest


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write rows to CSV using the union of all keys (rows may be heterogeneous)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read a JSON-lines result log."""
    path = Path(path)
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    text = path.read_text()
    lines = text.splitlines()
    final_may_be_partial = bool(lines) and not text.endswith("\n")
    for i, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            if final_may_be_partial and i == len(lines) - 1:
                print(f"   !! Dropping unterminated final JSONL row from {path}", flush=True)
                continue
            raise ValueError(f"Corrupt JSONL row {i + 1} in {path}: {exc}") from exc
    return rows


def rewrite_jsonl_dropping(path: Path, should_drop: Any) -> int:
    """Stream-filter a JSONL file in place, dropping rows where ``should_drop(parsed_row)`` is
    True. Bounded memory (one line at a time) so it is safe on the multi-GB multiclass
    ``predictions.jsonl`` that would OOM a whole-file ``read_jsonl``. Returns the number dropped;
    only rewrites the file if at least one row was dropped.

    A torn final line left by a hard crash is repaired FIRST (``repair_jsonl_tail``), confining the
    damage to the one row already lost. Any JSON error remaining after that is a CORRUPT INTERIOR row --
    a hard error, never silently dropped, because dropping it would hide real data loss."""
    path = Path(path)
    if not path.exists():
        return 0
    repair_jsonl_tail(path)
    tmp = path.parent / (path.name + ".prune.tmp")
    tmp.unlink(missing_ok=True)
    dropped = 0
    with path.open("r", encoding="utf-8") as fin, tmp.open("w", encoding="utf-8") as fout:
        for lineno, line in enumerate(fin, 1):
            s = line.strip()
            if not s:
                continue
            try:
                row = json.loads(s)
            except json.JSONDecodeError as exc:
                tmp.unlink(missing_ok=True)
                raise ValueError(f"corrupt interior JSONL row {lineno} in {path}: {exc}") from exc
            if should_drop(row):
                dropped += 1
                continue
            fout.write(line if line.endswith("\n") else line + "\n")
    if dropped:
        os.replace(tmp, path)
    else:
        tmp.unlink(missing_ok=True)
    return dropped


def repair_jsonl_tail(path: Path) -> int:
    """Truncate an unterminated final line, returning the bytes removed.

    ``read_jsonl`` drops a torn tail in memory but leaves it on disk. That is a latent brick: the
    next append glues its first row onto the torn bytes, turning a droppable trailing fragment
    into a corrupt INTERIOR row, and every subsequent ``read_jsonl`` then raises for good --
    recoverable only by discarding the whole directory. Repairing before the append keeps the
    damage confined to the row that was already lost.

    Scans backwards for the last newline rather than reading the file, which routinely reaches
    ~20 GB for predictions.jsonl.
    """
    path = Path(path)
    if not path.exists():
        return 0
    size = path.stat().st_size
    if size == 0:
        return 0
    with path.open("rb+") as f:
        f.seek(size - 1)
        if f.read(1) == b"\n":
            return 0  # cleanly terminated
        chunk = 1 << 16
        pos, last_newline = size, -1
        while pos > 0:
            start = max(0, pos - chunk)
            f.seek(start)
            buf = f.read(pos - start)
            idx = buf.rfind(b"\n")
            if idx != -1:
                last_newline = start + idx
                break
            pos = start
        keep = last_newline + 1  # 0 when the file is one unterminated row
        f.truncate(keep)
        f.flush()
        os.fsync(f.fileno())
    return size - keep


def append_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    """Append rows to a JSON-lines log (creates parents). Used for crash-resumable results.

    The whole batch is serialized first and written in a single ``write`` + ``fsync`` so a
    crash cannot persist a partial subset of one caller's rows (which would make a resumed run
    treat an interrupted sweep as finished). Pass one logical unit (one cell/family) per call
    so that batch is the atomic resume granularity.

    A torn tail left by a previous hard crash is truncated first -- see ``repair_jsonl_tail``.
    """
    if not rows:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    removed = repair_jsonl_tail(path)
    if removed:
        print(
            f"   !! {path}: truncated {removed} bytes of an unterminated final row before "
            f"appending (a previous run was killed mid-write)",
            flush=True,
        )
    payload = "".join(json.dumps(row, default=_json_default) + "\n" for row in rows)
    with path.open("a", encoding="utf-8") as f:
        f.write(payload)
        f.flush()
        os.fsync(f.fileno())


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)


def read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text())


def write_json(path: Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        tmp.write_text(json.dumps(value, indent=2, sort_keys=True))
        tmp.replace(path)
    finally:
        tmp.unlink(missing_ok=True)


def summarize_rows(
    rows: list[dict[str, Any]],
    keys: list[str],
    metrics: list[str],
    *,
    count_aggregates: list[str] = (),
    passthrough: list[str] = (),
) -> list[dict[str, Any]]:
    """Group rows by ``keys`` and report mean/std of each metric (over seeds/holdouts).

    Adds ``n_rows`` always, ``n_seeds`` / ``n_holdouts`` when those columns exist,
    and aggregates probe-convergence bookkeeping when present.

    ``count_aggregates`` are integer size columns (e.g. the label-access supervision counts) reported as
    ``min_/max_/mean_`` over the group. They are deliberately NOT group keys: label-access rows for the
    same (route, budget, ...) differ in n_source/n_target/n_total ACROSS holdouts (full-pool sizes vary
    by region), so keying on them would fragment the equal-region aggregation. ``passthrough`` are
    columns constant within a group (e.g. ``label_budget_unit``): the single distinct value is preserved,
    or "" when the group has none / disagrees.
    """
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        if not all(key in row for key in keys):
            continue
        grouped.setdefault(tuple(row[key] for key in keys), []).append(row)
    out: list[dict[str, Any]] = []
    for key_values, vals in sorted(grouped.items(), key=lambda kv: tuple(str(x) for x in kv[0])):
        row: dict[str, Any] = dict(zip(keys, key_values, strict=False))
        for metric in metrics:
            present = [float(v[metric]) for v in vals if metric in v and v[metric] is not None]
            finite = [x for x in present if np.isfinite(x)]
            row[f"mean_{metric}"] = float(np.mean(finite)) if finite else float("nan")
            row[f"std_{metric}"] = float(np.std(finite)) if finite else float("nan")
        for col in count_aggregates:
            sizes = [int(v[col]) for v in vals if col in v and v[col] is not None]
            if sizes:
                row[f"min_{col}"] = int(min(sizes))
                row[f"max_{col}"] = int(max(sizes))
                row[f"mean_{col}"] = float(np.mean(sizes))
        for col in passthrough:
            distinct = {v[col] for v in vals if col in v and v[col] not in ("", None)}
            row[col] = next(iter(distinct)) if len(distinct) == 1 else ""
        row["n_rows"] = len(vals)
        if "seed" in vals[0]:
            row["n_seeds"] = len({v["seed"] for v in vals})
        if "holdout" in vals[0]:
            row["n_holdouts"] = len({str(v["holdout"]) for v in vals})
        if "probe_converged" in vals[0]:
            row["all_probes_converged"] = int(all(int(v["probe_converged"]) == 1 for v in vals))
        if "probe_convergence_warnings" in vals[0]:
            row["total_probe_convergence_warnings"] = int(
                sum(int(v["probe_convergence_warnings"]) for v in vals)
            )
        out.append(row)
    return out


def _close(a: Any, b: float) -> bool:
    try:
        return abs(float(a) - b) < 1e-9
    except (TypeError, ValueError):
        return False


# --- chance / no-skill floor per metric (for the floor-normalized drop) ----------------
def _chance(metric: str, pos_rate: float | None, n_classes: float | None, majority_rate: float | None):
    """The no-skill baseline a metric would score, or None if not well-defined.

    Used as the floor in ``floor_norm_drop = (id - ood) / (id - chance)`` -- the fraction
    of *above-chance* performance erased by going OOD. Only defined for higher-is-better
    metrics with a clean no-skill reference; error metrics (brier/nll/ece) and macro/weighted
    F1 return None.
    """
    if metric in ("auc", "macro_auc"):
        return 0.5
    if metric in ("balanced_accuracy", "calibrated_balanced_accuracy"):
        return 1.0 / max(int(n_classes or 2), 2)
    if metric == "accuracy":
        return majority_rate
    if metric in ("f1", "calibrated_f1"):
        return (2.0 * pos_rate / (1.0 + pos_rate)) if (pos_rate and pos_rate > 0) else None  # always-positive no-skill F1
    return None


# --- numpy binary-metric recomputation (for per-sample bootstrap; keeps ioutils sklearn-free) ---
def _f1(y: np.ndarray, pred: np.ndarray) -> float:
    tp = float(((pred == 1) & (y == 1)).sum())
    fp = float(((pred == 1) & (y == 0)).sum())
    fn = float(((pred == 0) & (y == 1)).sum())
    denom = 2 * tp + fp + fn
    return 2 * tp / denom if denom > 0 else 0.0


def _balanced_acc(y: np.ndarray, pred: np.ndarray) -> float:
    tp = ((pred == 1) & (y == 1)).sum()
    fn = ((pred == 0) & (y == 1)).sum()
    tn = ((pred == 0) & (y == 0)).sum()
    fp = ((pred == 1) & (y == 0)).sum()
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    tnr = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    return 0.5 * (tpr + tnr)


def _rankdata_avg(a: np.ndarray) -> np.ndarray:
    """Ranks (1-indexed) with ties resolved by their MIDRANK (average rank of the tied group).

    This is what the Mann-Whitney form of AUC requires; plain argsort-of-argsort breaks ties
    by arbitrary sort order, which biases AUC for discrete-probability predictors like KNN.
    """
    order = np.argsort(a, kind="mergesort")
    a_sorted = a[order]
    ranks_sorted = np.empty(len(a), dtype=float)
    i = 0
    n = len(a)
    while i < n:
        j = i
        while j + 1 < n and a_sorted[j + 1] == a_sorted[i]:
            j += 1
        ranks_sorted[i : j + 1] = (i + j) / 2.0 + 1.0  # average of 1-indexed positions i..j
        i = j + 1
    out = np.empty(n, dtype=float)
    out[order] = ranks_sorted
    return out


def _auc(y: np.ndarray, p: np.ndarray) -> float:
    n_pos = int((y == 1).sum())
    n_neg = len(y) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = _rankdata_avg(p)  # midrank-corrected, so tied (e.g. KNN) probabilities score correctly
    return float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def _ece_np(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> float:
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.digitize(p, edges[1:-1])
    e = 0.0
    for b in range(n_bins):
        m = idx == b
        if m.any():
            e += m.sum() * abs(y[m].mean() - p[m].mean())
    return float(e / len(y))


def _bin_metric(name: str, y, prob, pred_def, pred_cal) -> float:
    if name == "f1":
        return _f1(y, pred_def)
    if name == "calibrated_f1":
        return _f1(y, pred_cal)
    if name == "balanced_accuracy":
        return _balanced_acc(y, pred_def)
    if name == "calibrated_balanced_accuracy":
        return _balanced_acc(y, pred_cal)
    if name == "auc":
        return _auc(y, prob)
    if name == "brier":
        return float(np.mean((prob - y) ** 2))
    if name == "nll":
        pc = np.clip(prob, 1e-12, 1 - 1e-12)
        return float(-np.mean(y * np.log(pc) + (1 - y) * np.log(1 - pc)))
    if name == "ece":
        return _ece_np(y, prob)
    return float("nan")


def _by_seed_arrays(preds: list[dict[str, Any]] | None) -> dict[Any, tuple]:
    """Group per-sample prediction dicts by SEED into per-seed
    ``(y, prob, pred_default, pred_calibrated, region_cluster)`` arrays.

    No cross-seed collapsing: each seed has its OWN test set (random_id and target membership
    both vary with seed), so collapsing by ``(holdout, sample_id)`` would average variable-
    membership predictions. Exact ``(holdout, sample_id)`` duplicates WITHIN a seed (crash-
    recovery re-appends) are de-duplicated so no observation is double-counted.
    """
    out: dict[Any, tuple] = {}
    if not preds:
        return out
    by_seed: dict[Any, dict[tuple, dict[str, Any]]] = {}
    for p in preds:
        if "prob" not in p or "pred_default" not in p or "pred_calibrated" not in p:
            continue
        by_seed.setdefault(p.get("seed"), {})[(p.get("holdout"), p.get("sample_id"))] = p
    for seed, uniq in by_seed.items():
        ps = list(uniq.values())
        out[seed] = (
            np.array([p["y_true"] for p in ps], dtype=np.float64),
            np.array([p["prob"] for p in ps], dtype=np.float64),
            np.array([p["pred_default"] for p in ps], dtype=np.int64),
            np.array([p["pred_calibrated"] for p in ps], dtype=np.int64),
            np.array([str(p.get("holdout", p.get("group", ""))) for p in ps], dtype=object),
        )
    return out


def _cluster_resample(rng: np.random.Generator, clusters: np.ndarray) -> np.ndarray:
    """One two-stage (nested) bootstrap draw: resample whole clusters (regions) with replacement,
    AND resample observations within each drawn cluster with replacement. With a single cluster
    (e.g. the ID anchor) this reduces to a plain per-sample resample."""
    uniq = np.unique(clusters)
    n = len(clusters)
    if len(uniq) <= 1:
        return rng.integers(0, n, n)
    idx_by_cluster = {c: np.flatnonzero(clusters == c) for c in uniq}
    out = []
    for c in rng.choice(uniq, size=len(uniq), replace=True):
        members = idx_by_cluster[c]
        out.append(members[rng.integers(0, len(members), len(members))])  # resample WITHIN the region
    return np.concatenate(out)


def _sample_delta_ci(metric, id_preds, ood_preds, n_boot, rng):
    """Hierarchical seed -> region -> sample bootstrap of the delta, WITHOUT collapsing seeds.

    Each seed is a complete, independent evaluation (its own source-trained model and its own
    test set). Point estimate = mean over seeds of the per-seed (id - ood) metric (matching the
    seed-mean ``delta``). Each bootstrap draw resamples seeds with replacement; within each drawn
    seed it resamples OOD by region/sample cluster and ID by sample, then averages the per-seed
    deltas. Returns ``(ci_lo, ci_hi, id_pt, ood_pt)``; NaN for non-binary metrics.
    """
    id_by_seed = _by_seed_arrays(id_preds)
    ood_by_seed = _by_seed_arrays(ood_preds)
    seeds = sorted(set(id_by_seed) & set(ood_by_seed), key=lambda s: (s is None, s))
    per_id, per_ood = [], []
    for s in seeds:
        ip = _bin_metric(metric, *id_by_seed[s][:4])
        op = _bin_metric(metric, *ood_by_seed[s][:4])
        if not (np.isnan(ip) or np.isnan(op)):
            per_id.append(ip)
            per_ood.append(op)
    if not per_id:
        return float("nan"), float("nan"), float("nan"), float("nan")
    id_pt, ood_pt = float(np.mean(per_id)), float(np.mean(per_ood))
    if not n_boot:
        return float("nan"), float("nan"), id_pt, ood_pt
    seeds_arr = np.array(seeds, dtype=object)
    deltas = []
    for _ in range(n_boot):
        draw = []
        for s in seeds_arr[rng.integers(0, len(seeds_arr), len(seeds_arr))]:
            yi, pi, di, ci, gi = id_by_seed[s]
            yo, po, do, co, go = ood_by_seed[s]
            ii, oo = _cluster_resample(rng, gi), _cluster_resample(rng, go)
            ip = _bin_metric(metric, yi[ii], pi[ii], di[ii], ci[ii])
            op = _bin_metric(metric, yo[oo], po[oo], do[oo], co[oo])
            if not (np.isnan(ip) or np.isnan(op)):
                draw.append(ip - op)
        if draw:
            deltas.append(float(np.mean(draw)))
    if not deltas:
        return float("nan"), float("nan"), id_pt, ood_pt
    return float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5)), id_pt, ood_pt


def compute_deltas(
    rows: list[dict[str, Any]],
    metrics: list[str],
    *,
    predictions: list[dict[str, Any]] | None = None,
    ood_regimes: tuple[str, ...] | None = None,
    id_source_budget: float | int = 1.0,
    ood_target_budget: float | int = 0.0,
    target_id_budget: float | int | None = -1.0,
    n_boot: int = 2000,
    n_boot_sample: int = 1000,
    seed: int = 0,
) -> list[dict[str, Any]]:
    from evals import confounds

    if ood_regimes is None:
        # A regime is OOD-eligible if it carries a target-side row in EITHER schema: the schema-v2
        # label-access suite (geographic_ood) or the pre-label-access target sweep (historical trees).
        # Gating on budget_type=="target" alone silently discovered nothing once geographic_ood moved
        # to label_access rows, which produced an empty deltas.csv that still certified as complete.
        found = sorted({
            str(r.get("split_regime"))
            for r in rows
            if r.get("budget_type") in ("target", "label_access")
            and r.get("split_regime") not in (None, "random_id")
        })
        ood_regimes = tuple(found or ["geographic_ood"])

    def scoped(items, regime):
        out = []
        for item in items or []:
            split_regime = item.get("split_regime")
            if split_regime == "random_id":
                out.append(item)
            elif split_regime == regime:
                out.append({**item, "split_regime": "geographic_ood"})
        return out

    out: list[dict[str, Any]] = []
    for regime in ood_regimes:
        regime_rows = scoped(rows, regime)
        if not regime_rows:
            continue
        deltas = confounds.compute_deltas(
            regime_rows,
            metrics,
            predictions=scoped(predictions or [], regime),
            id_source_budget=id_source_budget,
            ood_target_budget=ood_target_budget,
            target_id_budget=target_id_budget,
            n_boot=n_boot,
            n_boot_sample=n_boot_sample,
            seed=seed,
        )
        for row in deltas:
            row["ood_regime"] = regime
        out.extend(deltas)
    return out
