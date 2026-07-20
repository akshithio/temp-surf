"""Shared helpers for the schema-v2 split-artifact tests.

The frozen format is one ``assignments.csv`` per leaf (plus, for geographic_ood headline targets, a
sibling ``label_access.csv``) and one central ``data/logs/splits.json``.
``freeze`` writes every built leaf's CSV, records its SHA-256, and writes the single log -- the exact
flow ``tools/generate_splits.py`` uses -- so a test can publish then load back through the runtime.

Tests must use a SUBDIRECTORY of ``tmp_path`` as the splits root (e.g. ``tmp_path / "splits"``), so the
sibling ``.../logs/splits.json`` stays inside the per-test tmp dir.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from evals import split_artifacts as SA

_PROV: dict[str, Any] = {
    "generation_timestamp": "t", "code_revision": "test", "run_seeds": [0, 1, 2], "cluster_seed": 0,
}


def freeze(root, built) -> list[dict[str, Any]]:
    """Freeze built leaves. ``built`` is an iterable of ``(rows, summary)`` from ``SA.build_*_leaf``,
    optionally ``(rows, summary, label_access_rows)`` for a geographic_ood headline leaf.

    Writes each leaf's assignments.csv and (when given) its label_access.csv, records BOTH checksums on
    the entry -- the loaders verify each -- and writes the one central log. Returns the log entries."""
    entries = []
    for item in built:
        rows, summary = item[0], item[1]
        la_rows = item[2] if len(item) > 2 else None
        _path, summary["sha256"] = SA.write_assignments(
            root, summary["benchmark"], summary["regime"], summary["seed"], summary["holdout"], rows
        )
        if la_rows is not None:
            la_path, summary["label_access_sha256"] = SA.write_label_access(
                root, summary["benchmark"], summary["seed"], summary["holdout"], la_rows
            )
            summary["label_access_csv"] = str(la_path.relative_to(root))
        entries.append(summary)
    SA.write_splits_log(SA.default_log_path(root), provenance=_PROV, entries=entries)
    return entries


def attach_label_access(root, benchmark, seed, holdout, la_rows, *, benchmark_budget=None) -> str:
    """Write a leaf's label_access.csv AFTER the log was frozen, recording its sha256 AND the
    ``label_access`` block on the entry.

    The loaders verify that checksum, so a fixture that writes the file without updating the central
    log would be refused -- exactly as a real tampered or regenerated draw would be. They also REQUIRE
    the recorded ``benchmark_budget`` (B_d), without which the fixed-budget allocation curve is
    undefined; when not given it is derived from this leaf alone, the way
    :func:`SA.benchmark_budget` derives it from the eligible cells.
    """
    la_path, sha = SA.write_label_access(root, benchmark, seed, holdout, la_rows)
    n_source = sum(1 for r in la_rows if r["population"] == SA.POP_SOURCE)
    n_pool = sum(1 for r in la_rows if r["population"] == SA.POP_TARGET_POOL)
    if benchmark_budget is None:
        benchmark_budget = SA.benchmark_budget([{"n_source": n_source, "n_target_pool": n_pool}])
    log_path = SA.default_log_path(root)
    log = json.loads(Path(log_path).read_text())
    for entry in log["leaves"]:
        if (entry["benchmark"] == benchmark and entry["regime"] == SA.LABEL_ACCESS_REGIME
                and int(entry["seed"]) == int(seed) and str(entry["holdout"]) == str(holdout)):
            entry["label_access_csv"] = str(la_path.relative_to(root))
            entry["label_access_sha256"] = sha
            entry["label_access"] = {
                "headline_eligible": True,
                "benchmark_budget": int(benchmark_budget),
                "unit": SA.label_access_unit(benchmark),
                "n_source_pool": int(n_source),
                "n_target_pool": int(n_pool),
                "additive_counts": list(SA.LABEL_ACCESS_COUNTS),
            }
    Path(log_path).write_text(json.dumps(log, indent=2) + "\n")
    return sha
