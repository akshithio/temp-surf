"""Result IO and summary aggregation for experiment runners.

Kept dependency-free (numpy only) so it never imports the eval/method modules:
the caller passes the metric list to summarize, since this project has three task
families (binary, multiclass, regression) with different metrics.

This file was renamed from ``io-utils.py`` (hyphen breaks imports) to ``ioutils.py``.
"""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import Any

import numpy as np


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
    """Read a JSON-lines result log (one row dict per line). Skips blank/corrupt lines."""
    path = Path(path)
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue  # tolerate a half-written final line from a crash
    return rows


def append_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    """Append rows to a JSON-lines log (creates parents). Used for crash-resumable results."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, default=_json_default) + "\n")


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
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True))
    tmp.replace(path)


def summarize_rows(
    rows: list[dict[str, Any]],
    keys: list[str],
    metrics: list[str],
) -> list[dict[str, Any]]:
    """Group rows by ``keys`` and report mean/std of each metric (over seeds/holdouts).

    Adds ``n_rows`` always, ``n_seeds`` / ``n_holdouts`` when those columns exist,
    and aggregates probe-convergence bookkeeping when present.
    """
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(tuple(row[key] for key in keys), []).append(row)
    out: list[dict[str, Any]] = []
    for key_values, vals in sorted(grouped.items(), key=lambda kv: tuple(str(x) for x in kv[0])):
        row: dict[str, Any] = dict(zip(keys, key_values))
        for metric in metrics:
            present = [float(v[metric]) for v in vals if metric in v and v[metric] is not None]
            finite = [x for x in present if np.isfinite(x)]
            row[f"mean_{metric}"] = float(np.mean(finite)) if finite else float("nan")
            row[f"std_{metric}"] = float(np.std(finite)) if finite else float("nan")
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


def load_env_file(path: Path) -> None:
    """Populate os.environ from a simple .env file (used for data/model paths, tokens)."""
    path = Path(path)
    if not path.exists():
        return
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        key, separator, value = line.partition("=")
        if not separator:
            continue
        key = key.strip()
        value = value.strip()
        if value[:1] == value[-1:] and value.startswith(("'", '"')):
            value = value[1:-1]
        if key:
            os.environ.setdefault(key, value)
