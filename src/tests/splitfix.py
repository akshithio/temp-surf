"""Shared helpers for the schema-v2 split-artifact tests.

The frozen format is one ``assignments.csv`` per leaf (plus, for geographic_ood headline targets, a
sibling ``label_access.csv``) and one central ``data/logs/splits.json``.
``freeze`` writes every built leaf's CSV, records its SHA-256, and writes the single log -- the exact
flow ``tools/generate_splits.py`` uses -- so a test can publish then load back through the runtime.

Tests must use a SUBDIRECTORY of ``tmp_path`` as the splits root (e.g. ``tmp_path / "splits"``), so the
sibling ``.../logs/splits.json`` stays inside the per-test tmp dir.
"""

from __future__ import annotations

from typing import Any

from evals import split_artifacts as SA

_PROV: dict[str, Any] = {
    "generation_timestamp": "t", "code_revision": "test", "run_seeds": [0, 1, 2], "cluster_seed": 0,
}


def freeze(root, built) -> list[dict[str, Any]]:
    """Freeze built leaves. ``built`` is an iterable of ``(rows, summary)`` from ``SA.build_*_leaf``.
    Writes each leaf's assignments.csv, sets its ``sha256``, and writes the one central log. Returns the
    log entries."""
    entries = []
    for rows, summary in built:
        _path, summary["sha256"] = SA.write_assignments(
            root, summary["benchmark"], summary["regime"], summary["seed"], summary["holdout"], rows
        )
        entries.append(summary)
    SA.write_splits_log(SA.default_log_path(root), provenance=_PROV, entries=entries)
    return entries
