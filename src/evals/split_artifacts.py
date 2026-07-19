"""Frozen split assignments under ``data/splits/`` plus one central log ``data/logs/splits.json``.

One CSV per split leaf, one JSON log for the whole generation:

    data/splits/<benchmark>/<regime>/<seed>/<holdout>/assignments.csv
    data/logs/splits.json

``assignments.csv`` is the ONLY file in a leaf. Columns: ``stable_id, partition, status, domain,
reason``. It carries EVERY eligible stable id exactly once:

  * assigned -- ``partition`` is one of the five v2 partitions, ``status`` ``assigned``, blank reason;
  * purged   -- removed by the source<->target distance purge; blank ``partition``, ``status``
                ``purged``, ``reason`` ``purged_near_ood``;
  * excluded -- any other non-assigned eligible id; blank ``partition``, ``status`` ``excluded``,
                ``reason`` the specific cause (``unknown_domain`` / ``no_coords`` / ``unassigned``).

``data/logs/splits.json`` is the ONLY metadata / provenance / summary / checksum file, built once
after generation: run-level provenance (timestamp, code revision, inputs, full split configuration,
run + cluster seeds) plus one entry per (benchmark, regime, seed, holdout) carrying the CSV's
relative path, its SHA-256, the partition / status / class / domain counts, purge distance + count,
exclusion counts by reason, the target role and target-label capability, and the validation result.
Stable ids live ONLY in the CSVs, never duplicated into the log.

Generation is single-process and one-time. The runtime reads the frozen CSVs and uses ``splits.json``
only for discovery + checksum verification; it never reconstructs splits or domains. There is no
per-leaf manifest/JSON, no exclusions.csv, no index/split_ref, no schema versioning, and no
locking/staging/atomic publication.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import os
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np

from evals.regimes.base import (
    V2_PARTITIONS,
    DenseSourceTargetSplit,
    SourceTargetSplit,
    require_bool_flags,
    route_partition_problems,
)

#: The five explicit schema-v2 partitions (single source of truth: ``base.V2_PARTITIONS``).
PARTITIONS: tuple[str, ...] = V2_PARTITIONS
CSV_HEADER: list[str] = ["stable_id", "partition", "status", "domain", "reason"]

STATUS_ASSIGNED = "assigned"
STATUS_PURGED = "purged"
STATUS_EXCLUDED = "excluded"
STATUSES = (STATUS_ASSIGNED, STATUS_PURGED, STATUS_EXCLUDED)

LOG_FILENAME = "splits.json"
_PART_RANK = {p: i for i, p in enumerate(PARTITIONS)}


class SplitArtifactError(RuntimeError):
    """A split artifact is malformed, inconsistent, or references unknown units."""


# --------------------------------------------------------------------------- #
# Paths + checksums
# --------------------------------------------------------------------------- #
def leaf_dir(root: str | os.PathLike, benchmark: str, regime: str, seed: int, holdout: str) -> Path:
    return Path(root) / str(benchmark) / str(regime) / str(int(seed)) / str(holdout)


def assignments_path(root: str | os.PathLike, benchmark: str, regime: str, seed: int, holdout: str) -> Path:
    return leaf_dir(root, benchmark, regime, seed, holdout) / "assignments.csv"


def leaf_rel_path(benchmark: str, regime: str, seed: int, holdout: str) -> str:
    """The leaf CSV path relative to ``data/splits/`` -- the key recorded in the central log."""
    return f"{benchmark}/{regime}/{int(seed)}/{holdout}/assignments.csv"


def default_log_path(splits_root: str | os.PathLike) -> Path:
    """``data/logs/splits.json`` given ``data/splits`` as the splits root."""
    return Path(splits_root).parent / "logs" / LOG_FILENAME


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sid_key(sid: str) -> tuple[int, Any]:
    return (0, int(sid)) if sid.isdigit() else (1, sid)


def _row_sort_key(row: dict[str, str]) -> tuple[int, tuple[int, Any]]:
    if row["status"] == STATUS_ASSIGNED:
        rank = _PART_RANK[row["partition"]]
    else:
        rank = len(PARTITIONS) + (0 if row["status"] == STATUS_PURGED else 1)
    return rank, _sid_key(row["stable_id"])


# --------------------------------------------------------------------------- #
# Validation (the scientific-invariant core)
# --------------------------------------------------------------------------- #
def validate_rows(rows: list[dict[str, str]], *, has_target: Any, supports_target_labels: Any) -> str:
    """Every leaf invariant, over the CSV rows. Raises :class:`SplitArtifactError` on any violation;
    returns ``"passed"``. Enforces: unique stable ids (so the five partitions are pairwise disjoint
    and no id is assigned twice); assigned rows carry a valid partition + non-blank domain;
    non-assigned rows carry a blank partition + a reason; and the fail-closed route-capability
    contract between the two flags and the target partition sizes."""
    ids = [r["stable_id"] for r in rows]
    if len(set(ids)) != len(ids):
        dup = sorted({i for i in ids if ids.count(i) > 1})
        raise SplitArtifactError(f"duplicate stable_id in leaf: {dup[:5]}")
    by_part = {p: 0 for p in PARTITIONS}
    for r in rows:
        st = r["status"]
        if st not in STATUSES:
            raise SplitArtifactError(f"invalid status {st!r} for id {r['stable_id']!r}")
        if st == STATUS_ASSIGNED:
            if r["partition"] not in PARTITIONS:
                raise SplitArtifactError(f"assigned id {r['stable_id']!r} has invalid partition {r['partition']!r}")
            if not str(r["domain"]).strip():
                raise SplitArtifactError(f"assigned id {r['stable_id']!r} has a blank domain")
            by_part[r["partition"]] += 1
        else:
            if r["partition"]:
                raise SplitArtifactError(f"{st} id {r['stable_id']!r} must have a blank partition, got {r['partition']!r}")
            if not str(r["reason"]).strip():
                raise SplitArtifactError(f"{st} id {r['stable_id']!r} must carry a reason")
    try:
        require_bool_flags(has_target, supports_target_labels)
    except ValueError as exc:
        raise SplitArtifactError(str(exc)) from exc
    problems = route_partition_problems(
        has_target, supports_target_labels, by_part["target_label_pool"], by_part["target_test"]
    )
    if problems:
        raise SplitArtifactError("route-capability invariants violated: " + "; ".join(problems))
    return "passed"


def _leaf_summary(
    benchmark: str, regime: str, seed: int, holdout: str, rows: list[dict[str, str]], *,
    target_unit: str, group_kind: str, has_target: bool, supports_target_labels: bool,
    target_role: str, purge_km: float, class_by_id: dict[str, list[str]],
) -> dict[str, Any]:
    """Build the central-log entry for one leaf (everything but the CSV's SHA-256, which the generator
    fills in after writing the file). Carries per-partition stratification stats, not stable ids.

    ``class_by_id`` maps each ASSIGNED id to its class label(s): a one-element list for a tabular
    sample, and the patch's sorted class-presence set for a PASTIS patch (so dense class_counts are
    patch-level presence within each partition, never pixel totals)."""
    status_counts = {s: 0 for s in STATUSES}
    exclusion_counts: dict[str, int] = {}
    partition_stats: dict[str, dict[str, Any]] = {
        p: {"n": 0, "class_counts": {}, "domain_counts": {}} for p in PARTITIONS
    }
    for r in rows:
        status_counts[r["status"]] += 1
        if r["status"] == STATUS_ASSIGNED:
            ps = partition_stats[r["partition"]]
            ps["n"] += 1
            ps["domain_counts"][r["domain"]] = ps["domain_counts"].get(r["domain"], 0) + 1
            for c in class_by_id.get(r["stable_id"], ()):
                ps["class_counts"][c] = ps["class_counts"].get(c, 0) + 1
        elif r["status"] == STATUS_EXCLUDED:
            exclusion_counts[r["reason"]] = exclusion_counts.get(r["reason"], 0) + 1
    for ps in partition_stats.values():  # stable, diff-friendly ordering
        ps["class_counts"] = dict(sorted(ps["class_counts"].items()))
        ps["domain_counts"] = dict(sorted(ps["domain_counts"].items()))
    return {
        "benchmark": benchmark, "regime": regime, "seed": int(seed), "holdout": holdout,
        "target_unit": target_unit, "group_kind": group_kind,
        "has_target": has_target, "supports_target_labels": supports_target_labels, "target_role": target_role,
        "assignments_csv": leaf_rel_path(benchmark, regime, seed, holdout),
        "partition_stats": partition_stats, "status_counts": status_counts,
        "purge_km": float(purge_km), "purge_count": status_counts[STATUS_PURGED],
        "exclusion_counts": dict(sorted(exclusion_counts.items())),
        "n_eligible": len(rows),
        "validation": validate_rows(rows, has_target=has_target, supports_target_labels=supports_target_labels),
    }


# --------------------------------------------------------------------------- #
# Row building from realized regime output
# --------------------------------------------------------------------------- #
def _row(sid: str, partition: str, status: str, domain: str, reason: str) -> dict[str, str]:
    return {"stable_id": sid, "partition": partition, "status": status, "domain": domain, "reason": reason}


def build_tabular_leaf(
    benchmark: str, regime: str, seed: int, *,
    split: SourceTargetSplit, domains: Any, labels: Any, sample_ids: Any,
    audit_events: list[dict[str, Any]], purge_km: float = 0.0,
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    """One realized :class:`SourceTargetSplit` -> (CSV rows, central-log summary).

    Every eligible sample appears exactly once. A purge audit event proves ``purged``; an
    ``unknown``/``nan`` domain proves ``excluded/unknown_domain``; any other non-assigned sample is
    ``excluded/unassigned``. Domains and route capabilities come straight from the split.
    """
    sample_ids = np.asarray(sample_ids, dtype=object)
    domains_s = np.asarray(domains).astype(str)
    labels_s = np.asarray(labels).astype(str)
    id_index = {str(s): i for i, s in enumerate(sample_ids.tolist())}

    partition_of: dict[str, str] = {}
    for part, arr in split.as_partitions().items():
        for i in np.asarray(arr, dtype=np.int64).tolist():
            partition_of[str(sample_ids[i])] = part
    purged: set[str] = set()
    for ev in audit_events:
        if ev.get("kind") == "purge":
            for i in ev.get("purged_indices", ev.get("purged_train_indices", [])):
                if 0 <= int(i) < len(sample_ids):
                    purged.add(str(sample_ids[int(i)]))

    rows: list[dict[str, str]] = []
    for sid in (str(s) for s in sample_ids.tolist()):
        dom = str(domains_s[id_index[sid]])
        if sid in partition_of:
            rows.append(_row(sid, partition_of[sid], STATUS_ASSIGNED, dom, ""))
        elif sid in purged:
            rows.append(_row(sid, "", STATUS_PURGED, dom, "purged_near_ood"))
        elif dom in ("unknown", "nan"):
            rows.append(_row(sid, "", STATUS_EXCLUDED, dom, "unknown_domain"))
        else:
            rows.append(_row(sid, "", STATUS_EXCLUDED, dom, "unassigned"))

    class_by_id = {sid: [str(labels_s[id_index[sid]])] for sid in partition_of}
    summary = _leaf_summary(
        benchmark, regime, seed, str(split.label), rows, target_unit="sample",
        group_kind=str(split.group_kind), has_target=split.has_target,
        supports_target_labels=split.supports_target_labels, target_role=str(split.target_role),
        purge_km=purge_km, class_by_id=class_by_id,
    )
    return rows, summary


def build_dense_leaf(
    benchmark: str, regime: str, seed: int, *,
    dense_split: DenseSourceTargetSplit, audit_events: list[dict[str, Any]],
    all_patch_ids: list[int], domain_of: dict[int, str], class_sets: dict[int, set[int]],
    patch_latlon: dict[int, Any], purge_km: float = 0.0,
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    """One realized :class:`DenseSourceTargetSplit` -> (CSV rows, central-log summary), at patch level.

    Purge audit ``purged_indices`` are positions into ``all_patch_ids`` (the regime clusters/purges in
    that order). A no-coordinate unassigned patch proves ``excluded/no_coords``; any other unassigned
    patch is ``excluded/unassigned``. Patches are never split (whole-patch atomicity)."""
    partition_of: dict[str, str] = {}
    for part, pset in dense_split.as_partitions().items():
        for p in pset:
            partition_of[str(int(p))] = part
    purged: set[str] = set()
    for ev in audit_events:
        if ev.get("kind") == "purge":
            for i in ev.get("purged_indices", []):
                if 0 <= int(i) < len(all_patch_ids):
                    purged.add(str(all_patch_ids[int(i)]))

    rows: list[dict[str, str]] = []
    for pid in all_patch_ids:
        sid = str(int(pid))
        dom = str(domain_of.get(int(pid), ""))
        if sid in partition_of:
            rows.append(_row(sid, partition_of[sid], STATUS_ASSIGNED, dom, ""))
        elif sid in purged:
            rows.append(_row(sid, "", STATUS_PURGED, dom, "purged_near_ood"))
        else:
            ll = patch_latlon.get(int(pid))
            no_coords = ll is None or not np.all(np.isfinite(np.asarray(ll, dtype=float)))
            rows.append(_row(sid, "", STATUS_EXCLUDED, dom, "no_coords" if no_coords else "unassigned"))

    # patch-level class PRESENCE (the patch's class-set), never pixel totals
    class_by_id = {sid: sorted(str(c) for c in class_sets.get(int(sid), set())) for sid in partition_of}
    summary = _leaf_summary(
        benchmark, regime, seed, str(dense_split.label), rows, target_unit="patch",
        group_kind=str(dense_split.group_kind), has_target=dense_split.has_target,
        supports_target_labels=dense_split.supports_target_labels, target_role=str(dense_split.target_role),
        purge_km=purge_km, class_by_id=class_by_id,
    )
    return rows, summary


# --------------------------------------------------------------------------- #
# Writing (plain, single-process)
# --------------------------------------------------------------------------- #
def _csv_bytes(rows: list[dict[str, str]]) -> bytes:
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(CSV_HEADER)
    for r in sorted(rows, key=_row_sort_key):
        w.writerow([r["stable_id"], r["partition"], r["status"], r["domain"], r["reason"]])
    return buf.getvalue().encode()


def write_assignments(
    root: str | os.PathLike, benchmark: str, regime: str, seed: int, holdout: str, rows: list[dict[str, str]],
) -> tuple[Path, str]:
    """Write one leaf's ``assignments.csv`` (deterministic row order) and return ``(path, sha256)``."""
    label = str(holdout)
    if "/" in label or label in ("", ".", ".."):
        raise SplitArtifactError(f"unsafe holdout label for a leaf directory: {holdout!r}")
    data = _csv_bytes(rows)
    path = assignments_path(root, benchmark, regime, seed, holdout)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path, sha256_bytes(data)


def write_splits_log(logs_path: str | os.PathLike, *, provenance: dict[str, Any], entries: list[dict[str, Any]]) -> Path:
    """Write the single central log (run provenance + one entry per leaf, checksums included)."""
    path = Path(logs_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {**provenance, "leaves": list(entries)}
    path.write_text(json.dumps(payload, indent=2) + "\n")
    return path


# --------------------------------------------------------------------------- #
# Reading + runtime mapping (discovery + integrity from the log; splits from the CSVs)
# --------------------------------------------------------------------------- #
class LoadedTabularSplit(NamedTuple):
    """One consumed tabular leaf: a :class:`SourceTargetSplit` of CURRENT row indices, its seed/regime,
    and the per-sample domain array (worst-group scoring)."""

    seed: int
    regime: str
    domains: np.ndarray
    split: SourceTargetSplit


class LoadedDenseSplit(NamedTuple):
    """One consumed dense leaf: a :class:`DenseSourceTargetSplit` (patch sets) + its seed/regime."""

    seed: int
    regime: str
    split: DenseSourceTargetSplit


def read_splits_log(logs_path: str | os.PathLike) -> dict[str, Any]:
    path = Path(logs_path)
    if not path.is_file():
        raise SplitArtifactError(f"no split log at {path} -- run tools/generate_splits.py first")
    try:
        log = json.loads(path.read_text())
    except (ValueError, OSError) as exc:
        raise SplitArtifactError(f"malformed split log at {path}: {exc}") from exc
    if not isinstance(log.get("leaves"), list):
        raise SplitArtifactError(f"split log at {path} has no 'leaves' list")
    return log


def _leaf_entries(log: dict[str, Any], benchmark: str, regime: str, seed: int) -> list[dict[str, Any]]:
    return [
        e for e in log["leaves"]
        if str(e.get("benchmark")) == str(benchmark)
        and str(e.get("regime")) == str(regime)
        and int(e.get("seed")) == int(seed)
    ]


def read_assignments_csv(path: str | os.PathLike) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with open(path, newline="") as fh:
        reader = csv.reader(fh)
        header = next(reader, None)
        if header != CSV_HEADER:
            raise SplitArtifactError(f"malformed assignments header in {path}: {header}")
        for row in reader:
            if len(row) != len(CSV_HEADER):
                raise SplitArtifactError(f"malformed assignments row in {path}: {row}")
            rows.append(dict(zip(CSV_HEADER, row, strict=True)))
    return rows


def verify_leaf_csv(splits_root: str | os.PathLike, entry: dict[str, Any]) -> list[dict[str, str]]:
    """Read one leaf's CSV, verifying its SHA-256 against the central log (integrity gate)."""
    rel = str(entry["assignments_csv"])
    path = Path(splits_root) / rel
    if not path.is_file():
        raise SplitArtifactError(f"split log references a missing CSV: {path}")
    data = path.read_bytes()
    got, want = sha256_bytes(data), str(entry.get("sha256", ""))
    if got != want:
        raise SplitArtifactError(
            f"checksum mismatch for {rel}: log records {want[:12]}..., file hashes to {got[:12]}... "
            f"-- the frozen assignments changed; refuse to consume"
        )
    return read_assignments_csv(path)


def _assert_complete(csv_ids: list[str], eligible: set[str], where: str) -> None:
    """Every current eligible id appears exactly once in the CSV (complete accounting on CURRENT data);
    no unknown id and no missing id."""
    csv_set = set(csv_ids)
    if len(csv_set) != len(csv_ids):
        dup = sorted({i for i in csv_ids if csv_ids.count(i) > 1})
        raise SplitArtifactError(f"{where}: duplicate stable_id in assignments.csv: {dup[:5]}")
    if csv_set != eligible:
        missing = sorted(eligible - csv_set)
        extra = sorted(csv_set - eligible)
        raise SplitArtifactError(
            f"{where}: assignments.csv does not account for the current population exactly once -- "
            f"{len(missing)} missing (e.g. {missing[:5]}), {len(extra)} unexpected (e.g. {extra[:5]})"
        )


def map_to_indices(assignments: dict[str, list[str]], id_map: dict[str, Any]) -> dict[str, np.ndarray]:
    """Resolve the five assigned partitions' stable IDs to current row indices / patch IDs. Refuses an
    unknown id or an id in more than one partition -- the structural runtime identity gate."""
    result: dict[str, np.ndarray] = {}
    seen: set[str] = set()
    for part in PARTITIONS:
        resolved: list[Any] = []
        for sid in assignments.get(part, []):
            if sid not in id_map:
                raise SplitArtifactError(f"unknown id {sid!r} in partition {part!r} not present in current benchmark")
            if sid in seen:
                raise SplitArtifactError(f"id {sid!r} appears in multiple partitions")
            seen.add(sid)
            resolved.append(id_map[sid])
        result[part] = np.asarray(resolved, dtype=np.int64) if resolved else np.empty(0, dtype=np.int64)
    return result


def _check_dense_structure(
    stored_domain: dict[str, str], current: dict[int, Any], *, kind: str, benchmark: str, holdout: str,
) -> None:
    """Every assigned patch's CURRENT structural metadata (``current[patch]``) must equal the value
    frozen as its domain at generation. Refuses a stale artifact rather than mis-consuming it."""
    for patch_str, frozen in stored_domain.items():
        cur = current.get(int(patch_str))
        if cur is None or str(cur) != str(frozen):
            raise SplitArtifactError(
                f"PASTIS {benchmark}/{holdout}: patch {patch_str} was frozen with {kind} {frozen!r} "
                f"but the current benchmark has {cur!r} -- structural metadata changed, refuse to consume"
            )


def load_tabular_splits(
    root: str | os.PathLike, benchmark: str, sample_ids: Any, split_regimes: list[str], seeds: list[int],
) -> list[LoadedTabularSplit]:
    """Consume frozen tabular leaves as :class:`LoadedTabularSplit` (no construction). Discovery +
    integrity come from ``data/logs/splits.json``; partitions and the per-sample ``domains`` array come
    from the verified CSVs. Hard-fails on a missing log, checksum mismatch, incomplete accounting, or a
    requested regime with zero leaves."""
    log = read_splits_log(default_log_path(root))
    ids_list = [str(s) for s in np.asarray(sample_ids).tolist()]
    if len(set(ids_list)) != len(ids_list):
        dups = sorted({s for s in ids_list if ids_list.count(s) > 1})
        raise SplitArtifactError(
            f"{benchmark}: current sample_ids contain {len(dups)} duplicate stable id(s) (e.g. {dups[:5]}) "
            f"-- the id->index map would be ambiguous; refuse to consume splits"
        )
    id_map = {s: i for i, s in enumerate(ids_list)}
    eligible = set(id_map)
    n = len(ids_list)
    loaded: list[LoadedTabularSplit] = []
    for seed in seeds:
        for regime in split_regimes:
            entries = _leaf_entries(log, benchmark, regime, seed)
            if not entries:
                raise SplitArtifactError(
                    f"requested regime {regime!r} yielded zero leaves for {benchmark}/seed={seed} -- refuse to run"
                )
            for entry in entries:
                where = f"{benchmark}/{regime}/{seed}/{entry['holdout']}"
                rows = verify_leaf_csv(root, entry)
                _assert_complete([r["stable_id"] for r in rows], eligible, where)
                parts: dict[str, list[str]] = {p: [] for p in PARTITIONS}
                domains = np.full(n, "__unassigned__", dtype=object)
                for r in rows:
                    if r["status"] == STATUS_ASSIGNED:
                        parts[r["partition"]].append(r["stable_id"])
                        domains[id_map[r["stable_id"]]] = r["domain"]
                idx = map_to_indices(parts, id_map)
                try:  # the split enforces the route-capability contract; surface it as an artifact error
                    split = SourceTargetSplit(
                        label=str(entry["holdout"]),
                        source_train=idx["source_train"], source_val=idx["source_val"], source_test=idx["source_test"],
                        target_label_pool=idx["target_label_pool"], target_test=idx["target_test"],
                        domain=None, has_target=entry["has_target"],
                        supports_target_labels=entry["supports_target_labels"],
                        group_kind=str(entry["group_kind"]), target_role=str(entry["target_role"]),
                    )
                except ValueError as exc:
                    raise SplitArtifactError(f"{where}: {exc}") from exc
                loaded.append(LoadedTabularSplit(seed=int(seed), regime=str(regime), domains=domains, split=split))
    return loaded


def load_dense_splits(
    root: str | os.PathLike, benchmark: str, patch_fold: dict[int, int], patch_tile: dict[int, str | None],
    split_regimes: list[str], seeds: list[int],
) -> dict[int, list[LoadedDenseSplit]]:
    """Consume frozen PASTIS patch-level leaves as :class:`LoadedDenseSplit`. ``patch_fold`` keys are
    the eligible patch universe; ``patch_fold`` / ``patch_tile`` values drive the official-fold /
    geographic-tile structural check (spatial cells are frozen and never rechecked). Discovery +
    integrity from the central log; splits from the verified CSVs."""
    log = read_splits_log(default_log_path(root))
    id_map = {str(int(p)): int(p) for p in patch_fold}
    eligible = set(id_map)
    by_seed: dict[int, list[LoadedDenseSplit]] = {}
    for seed in seeds:
        leaves: list[LoadedDenseSplit] = []
        for regime in split_regimes:
            entries = _leaf_entries(log, benchmark, regime, seed)
            if not entries:
                raise SplitArtifactError(
                    f"requested regime {regime!r} yielded zero leaves for {benchmark}/seed={seed} -- refuse to run"
                )
            for entry in entries:
                holdout = str(entry["holdout"])
                where = f"{benchmark}/{regime}/{seed}/{holdout}"
                rows = verify_leaf_csv(root, entry)
                _assert_complete([r["stable_id"] for r in rows], eligible, where)
                parts: dict[str, list[str]] = {p: [] for p in PARTITIONS}
                stored_domain: dict[str, str] = {}
                for r in rows:
                    if r["status"] == STATUS_ASSIGNED:
                        parts[r["partition"]].append(r["stable_id"])
                        stored_domain[r["stable_id"]] = r["domain"]
                if regime == "official":
                    _check_dense_structure(stored_domain, patch_fold, kind="published fold", benchmark=benchmark, holdout=holdout)
                elif regime == "geographic_ood":
                    _check_dense_structure(stored_domain, patch_tile, kind="Sentinel tile", benchmark=benchmark, holdout=holdout)
                idx = map_to_indices(parts, id_map)

                def pset(part: str, idx: dict[str, np.ndarray] = idx) -> frozenset[int]:
                    return frozenset(int(p) for p in idx[part].tolist())

                try:  # the split enforces the route-capability contract; surface it as an artifact error
                    dsplit = DenseSourceTargetSplit(
                        label=holdout,
                        source_train_patches=pset("source_train"), source_val_patches=pset("source_val"),
                        source_test_patches=pset("source_test"),
                        target_label_pool_patches=pset("target_label_pool"), target_test_patches=pset("target_test"),
                        has_target=entry["has_target"], supports_target_labels=entry["supports_target_labels"],
                        group_kind=str(entry["group_kind"]), target_role=str(entry["target_role"]),
                    )
                except ValueError as exc:
                    raise SplitArtifactError(f"{where}: {exc}") from exc
                leaves.append(LoadedDenseSplit(seed=int(seed), regime=str(regime), split=dsplit))
        by_seed[int(seed)] = leaves
    return by_seed
