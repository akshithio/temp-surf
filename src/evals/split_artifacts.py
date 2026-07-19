"""Canonical split artifacts under ``data/splits/`` -- serialize, load, validate, and map.

There is exactly ONE canonical version of every split, identified by its PATH:

    data/splits/<benchmark>/<regime>/<seed>/<holdout_dirname>/{assignments.csv,exclusions.csv,manifest.json}
    data/splits/<benchmark>/<regime>/<seed>/generation.json
    data/splits/<benchmark>/index.json

No hashing, fingerprinting, or content-addressing of any kind: identity is the path, integrity is
structural (duplicate / overlap / unknown-id / complete-accounting / malformed), and a leaf is
"complete" iff its ``manifest.json`` is present.

This module never constructs a split. The split-preprocessing generator calls the existing regime
code (``evals.regimes.base.iter_splits`` / ``segmentation_fold_configs``), hands the realized
partitions here as stable IDs, and at runtime the loader maps those stable IDs back to current row
indices. Artifacts carry NO model identity.
"""

from __future__ import annotations

import csv
import fcntl
import io
import json
import os
import uuid
from collections import Counter
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import quote, unquote

import numpy as np

SCHEMA_VERSION = 1

#: The five regime-level partitions this phase freezes. Target-budget 80/20 draws, nested few-shot
#: ordering, and probe initialization are NOT stored here -- they remain seeded runtime operations.
PARTITIONS: tuple[str, ...] = ("train", "val", "test", "source_val", "source_test")

_ASSIGN_HEADER = ["stable_id", "partition", "domain"]
_EXCL_HEADER = ["stable_id", "reason", "status"]

#: Exclusion status vocabulary. ``proven`` = backed by a structured audit event or directly by the
#: data; ``inferred`` = derived without direct proof; ``unknown`` = unassigned by current behavior
#: with no established reason. Reasons are never guessed.
EXCLUSION_STATUSES = ("proven", "inferred", "unknown")
UNASSIGNED_REASON = "unassigned_current_behavior"


class SplitArtifactError(RuntimeError):
    """A split artifact is malformed, inconsistent, or references unknown units."""


def _require_schema_version(obj: dict[str, Any], where: str) -> None:
    """Parser-contract check: the artifact must declare ``schema_version == SCHEMA_VERSION``.

    This is NOT support for multiple historical versions -- this build reads and writes exactly one
    schema. A missing, non-integer, older, or unknown version is a hard SplitArtifactError, so a
    stale or foreign artifact can never be silently half-parsed.
    """
    if "schema_version" not in obj:
        raise SplitArtifactError(f"{where}: missing schema_version (this build requires {SCHEMA_VERSION})")
    ver = obj["schema_version"]
    if isinstance(ver, bool) or not isinstance(ver, int):
        raise SplitArtifactError(f"{where}: schema_version must be an integer, got {ver!r}")
    if ver != SCHEMA_VERSION:
        raise SplitArtifactError(
            f"{where}: unsupported schema_version {ver} (this build reads/writes {SCHEMA_VERSION})"
        )


# --------------------------------------------------------------------------- #
# Filesystem-safe holdout names (exact label preserved in manifest.json)
# --------------------------------------------------------------------------- #
def holdout_dirname(label: str) -> str:
    """Filesystem-safe, collision-free, reversible encoding of a holdout label.

    Percent-encodes anything outside ``[A-Za-z0-9._-]`` (reversible via :func:`holdout_label`,
    never a hash). The exact original label is preserved in ``manifest.json``.
    """
    encoded = quote(str(label), safe="._-")
    if encoded in ("", ".", ".."):
        encoded = "holdout%2E" + encoded.replace(".", "")
    return encoded


def holdout_label(dirname: str) -> str:
    """Invert :func:`holdout_dirname`."""
    return unquote(str(dirname))


def benchmark_dir(root: str | os.PathLike, benchmark: str) -> Path:
    return Path(root) / str(benchmark)


def regime_seed_dir(root: str | os.PathLike, benchmark: str, regime: str, seed: int) -> Path:
    return benchmark_dir(root, benchmark) / str(regime) / str(int(seed))


def leaf_dir(root: str | os.PathLike, benchmark: str, regime: str, seed: int, label: str) -> Path:
    return regime_seed_dir(root, benchmark, regime, seed) / holdout_dirname(label)


# --------------------------------------------------------------------------- #
# Leaf spec + validation
# --------------------------------------------------------------------------- #
@dataclass
class LeafSpec:
    benchmark: str
    regime: str
    seed: int
    holdout_label: str
    target_unit: str  # "sample" | "patch"
    domain_basis: str
    group_kind: str
    has_target: bool
    params: dict[str, Any]
    #: partition -> ordered list of stable IDs (strings). Missing partitions treated as empty.
    partitions: dict[str, list[str]]
    #: partition -> {"n", "domains", "domain_counts", "class_counts"} (neutral; no model identity).
    partition_stats: dict[str, dict[str, Any]]
    #: realized per-ID domain label (tabular: regime domain; dense: fold). One entry per ASSIGNED id;
    #: persisted in assignments.csv so runtime never recomputes it (no assign_domains/KMeans at run).
    domains: dict[str, str] = field(default_factory=dict)
    #: list of {"stable_id", "reason", "status"} covering every excluded eligible unit.
    exclusions: list[dict[str, Any]] = field(default_factory=list)
    #: structured audit events relevant to this leaf (purge / stratification fallbacks).
    audit: list[dict[str, Any]] = field(default_factory=list)


def _as_str_list(ids: Any) -> list[str]:
    return [str(x) for x in (ids or [])]


def _check_domain_coverage(assigned: set[str], domain_by_id: dict[str, str]) -> None:
    """Every assigned id has exactly one domain record; no domain record references a non-assigned id."""
    missing = sorted(assigned - set(domain_by_id))
    if missing:
        raise SplitArtifactError(f"{len(missing)} assigned id(s) have no domain record (e.g. {missing[:5]})")
    unknown = sorted(set(domain_by_id) - assigned)
    if unknown:
        raise SplitArtifactError(f"{len(unknown)} domain record(s) reference non-assigned id(s) (e.g. {unknown[:5]})")


def _check_partitions_accounting(
    partitions: dict[str, list[str]], exclusion_rows: list[dict[str, Any]], eligible: Any,
) -> dict[str, set[str]]:
    """The invariant core shared by generation and runtime load. Returns the per-partition id sets.

    Enforces, over stable IDs: no duplicate id within a partition; all five partitions pairwise
    disjoint; exclusions unique with valid status; assignments and exclusions disjoint; and COMPLETE
    ACCOUNTING (every eligible unit appears exactly once across assignments+exclusions).
    """
    assigned_sets: dict[str, set[str]] = {}
    for part in PARTITIONS:
        seen: set[str] = set()
        for sid in _as_str_list(partitions.get(part)):
            if sid in seen:
                raise SplitArtifactError(f"duplicate id {sid!r} within partition {part!r}")
            seen.add(sid)
        assigned_sets[part] = seen

    names = list(PARTITIONS)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            inter = assigned_sets[names[i]] & assigned_sets[names[j]]
            if inter:
                raise SplitArtifactError(
                    f"partitions {names[i]!r}/{names[j]!r} overlap on {len(inter)} id(s): {sorted(inter)[:5]}"
                )

    assigned: set[str] = set().union(*assigned_sets.values()) if assigned_sets else set()

    excl_ids: set[str] = set()
    for row in exclusion_rows:
        sid = str(row["stable_id"])
        if sid in excl_ids:
            raise SplitArtifactError(f"duplicate exclusion id {sid!r}")
        if str(row.get("status")) not in EXCLUSION_STATUSES:
            raise SplitArtifactError(f"exclusion {sid!r} has invalid status {row.get('status')!r}")
        excl_ids.add(sid)

    both = assigned & excl_ids
    if both:
        raise SplitArtifactError(f"{len(both)} id(s) appear in BOTH assignments and exclusions: {sorted(both)[:5]}")

    eligible_list = [str(e) for e in eligible]
    eligible_set = set(eligible_list)
    if len(eligible_set) != len(eligible_list):
        raise SplitArtifactError("eligible population contains duplicate ids")
    union = assigned | excl_ids
    if union != eligible_set:
        missing = sorted(eligible_set - union)
        extra = sorted(union - eligible_set)
        raise SplitArtifactError(
            "incomplete accounting: every eligible unit must appear exactly once across "
            f"assignments+exclusions -- {len(missing)} eligible id(s) missing (e.g. {missing[:5]}), "
            f"{len(extra)} unexpected id(s) (e.g. {extra[:5]})"
        )
    return assigned_sets


def validate_leaf(spec: LeafSpec, eligible_ids: Any) -> None:
    """Generation-time structural validation. Raises SplitArtifactError on any violation."""
    assigned_sets = _check_partitions_accounting(spec.partitions, spec.exclusions, eligible_ids)
    _check_domain_coverage(set().union(*assigned_sets.values()) if assigned_sets else set(), spec.domains)
    for part in PARTITIONS:
        stat_n = spec.partition_stats.get(part, {}).get("n")
        if stat_n is not None and int(stat_n) != len(assigned_sets[part]):
            raise SplitArtifactError(
                f"partition_stats[{part!r}].n={stat_n} != {len(assigned_sets[part])} assigned ids"
            )


# --------------------------------------------------------------------------- #
# Serialization
# --------------------------------------------------------------------------- #
def _sort_key(sid: str):
    return (0, int(sid)) if sid.isdigit() else (1, sid)


def _assignments_bytes(spec: LeafSpec) -> bytes:
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(_ASSIGN_HEADER)
    for part in PARTITIONS:
        for sid in sorted(_as_str_list(spec.partitions.get(part)), key=_sort_key):
            w.writerow([sid, part, str(spec.domains.get(sid, ""))])
    return buf.getvalue().encode()


def _exclusions_bytes(spec: LeafSpec) -> bytes:
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(_EXCL_HEADER)
    for row in sorted(spec.exclusions, key=lambda r: _sort_key(str(r["stable_id"]))):
        w.writerow([str(row["stable_id"]), str(row["reason"]), str(row["status"])])
    return buf.getvalue().encode()


def _reason_counts(exclusions: list[dict[str, Any]]) -> dict[str, int]:
    out: dict[str, int] = {}
    for row in exclusions:
        key = f"{row['reason']}:{row['status']}"
        out[key] = out.get(key, 0) + 1
    return out


def _manifest_dict(spec: LeafSpec) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "benchmark": spec.benchmark,
        "regime": spec.regime,
        "seed": int(spec.seed),
        "holdout": spec.holdout_label,  # EXACT original label
        "holdout_dirname": holdout_dirname(spec.holdout_label),
        "target_unit": spec.target_unit,
        "domain_basis": spec.domain_basis,
        "group_kind": spec.group_kind,
        "has_target": bool(spec.has_target),
        "params": spec.params,
        "partitions": {part: spec.partition_stats.get(part, {"n": 0}) for part in PARTITIONS},
        "n_excluded": len(spec.exclusions),
        "exclusion_reason_counts": _reason_counts(spec.exclusions),
        "audit": spec.audit,
        # No model identity. No hashes/fingerprints/checksums.
    }


def _manifest_bytes(spec: LeafSpec) -> bytes:
    return (json.dumps(_manifest_dict(spec), indent=2) + "\n").encode()


# --------------------------------------------------------------------------- #
# Atomic, overwrite-safe publication
# --------------------------------------------------------------------------- #
def _atomic_write(path: Path, data: bytes) -> None:
    tmp = path.parent / f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    with open(tmp, "wb") as fh:
        fh.write(data)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


@contextmanager
def _leaf_lock(ldir: Path):
    ldir.parent.mkdir(parents=True, exist_ok=True)
    lock_path = ldir.parent / (ldir.name + ".lock")
    fh = open(lock_path, "w")
    try:
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
        fh.close()


def publish_leaf(root: str | os.PathLike, spec: LeafSpec, eligible_ids: Any, *, overwrite: bool = True) -> Path:
    """Validate then atomically publish a leaf so a crash never yields a mixed, complete-looking leaf.

    1. Stage all three files and structurally validate BEFORE touching the canonical leaf.
    2. Acquire the per-leaf writer lock.
    3. If overwriting a complete leaf, remove its ``manifest.json`` FIRST (readers now see it as
       incomplete).
    4. Atomically replace ``assignments.csv`` and ``exclusions.csv`` from staging.
    5. Atomically publish ``manifest.json`` LAST (the completeness marker).
    6. Release the lock and clean staging.

    A crash at any intermediate point leaves either the old complete leaf or a manifest-less
    (incomplete) leaf.
    """
    validate_leaf(spec, eligible_ids)
    ldir = leaf_dir(root, spec.benchmark, spec.regime, spec.seed, spec.holdout_label)
    ldir.parent.mkdir(parents=True, exist_ok=True)

    staging = ldir.parent / f".staging-{ldir.name}-{os.getpid()}-{uuid.uuid4().hex}"
    staging.mkdir(parents=True, exist_ok=True)
    try:
        _atomic_write(staging / "assignments.csv", _assignments_bytes(spec))
        _atomic_write(staging / "exclusions.csv", _exclusions_bytes(spec))
        _atomic_write(staging / "manifest.json", _manifest_bytes(spec))
        with _leaf_lock(ldir):
            existing_manifest = ldir / "manifest.json"
            if existing_manifest.exists():
                if not overwrite:
                    raise SplitArtifactError(f"leaf already complete and overwrite=False: {ldir}")
                existing_manifest.unlink()  # from here the leaf reads as incomplete
            ldir.mkdir(parents=True, exist_ok=True)
            os.replace(staging / "assignments.csv", ldir / "assignments.csv")
            os.replace(staging / "exclusions.csv", ldir / "exclusions.csv")
            os.replace(staging / "manifest.json", ldir / "manifest.json")  # LAST
    finally:
        for leftover in list(staging.glob("*")):
            leftover.unlink()
        if staging.exists():
            staging.rmdir()
    return ldir


# --------------------------------------------------------------------------- #
# generation.json / index.json
# --------------------------------------------------------------------------- #
def write_generation(
    root: str | os.PathLike,
    benchmark: str,
    regime: str,
    seed: int,
    *,
    requested: list[str],
    yielded: list[str],
    dropped: list[dict[str, Any]],
    audit_events: list[dict[str, Any]],
    regime_problems: list[Any],
) -> Path:
    d = regime_seed_dir(root, benchmark, regime, seed)
    d.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "benchmark": benchmark,
        "regime": regime,
        "seed": int(seed),
        "requested_holdouts": list(requested),
        "yielded_holdouts": list(yielded),
        "dropped_holdouts": list(dropped),
        "audit_events": list(audit_events),
        "regime_problems": [list(p) if isinstance(p, tuple) else p for p in regime_problems],
    }
    path = d / "generation.json"
    _atomic_write(path, (json.dumps(payload, indent=2) + "\n").encode())
    return path


def write_index(root: str | os.PathLike, benchmark: str, leaves: list[dict[str, Any]]) -> Path:
    d = benchmark_dir(root, benchmark)
    d.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "benchmark": benchmark,
        "leaves": list(leaves),
    }
    path = d / "index.json"
    _atomic_write(path, (json.dumps(payload, indent=2) + "\n").encode())
    return path


# --------------------------------------------------------------------------- #
# Building a LeafSpec from realized (behavior-preserving) regime output
# --------------------------------------------------------------------------- #
def _ia(a: Any) -> np.ndarray:
    return np.asarray(a, dtype=np.int64)


def build_tabular_leaf(
    benchmark: str,
    regime: str,
    seed: int,
    *,
    label: str,
    train: Any,
    val: Any,
    test: Any,
    source_val: Any,
    source_test: Any,
    domains: Any,
    labels: Any,
    sample_ids: Any,
    has_target: bool,
    group_kind: str,
    params: dict[str, Any],
    audit_events: list[dict[str, Any]],
) -> tuple[LeafSpec, list[str]]:
    """Turn one realized tabular split (row-index arrays) into a canonical, model-free LeafSpec.

    Exclusions are the SET-COMPLEMENT of the five partitions within the eligible population, so a
    sample can never silently vanish. Reasons are proven, never guessed: a purge audit event proves
    ``purged_near_ood``; an ``unknown``/``nan`` domain proves ``unknown_domain``; otherwise
    ``unassigned_current_behavior`` with status ``unknown``.
    """
    from evals.regimes import base as regime_base  # local import: base never imports this module

    sample_ids = np.asarray(sample_ids, dtype=object)
    idx = {
        "train": _ia(train), "val": _ia(val), "test": _ia(test),
        "source_val": _ia(source_val), "source_test": _ia(source_test),
    }
    partitions = {p: [str(sample_ids[i]) for i in a.tolist()] for p, a in idx.items()}
    stats = {p: regime_base.partition_stats(domains, labels, a) for p, a in idx.items()}

    eligible = [str(s) for s in sample_ids.tolist()]
    assigned = set().union(*(set(v) for v in partitions.values())) if partitions else set()
    id_to_index = {str(s): i for i, s in enumerate(sample_ids.tolist())}
    domains_s = np.asarray(domains).astype(str)

    purged: set[str] = set()
    for ev in audit_events:
        if ev.get("kind") == "purge":
            for i in ev.get("purged_train_indices", []):
                if 0 <= int(i) < len(sample_ids):
                    purged.add(str(sample_ids[int(i)]))

    exclusions: list[dict[str, Any]] = []
    for sid in eligible:
        if sid in assigned:
            continue
        if sid in purged:
            reason, status = "purged_near_ood", "proven"
        elif domains_s[id_to_index[sid]] in ("unknown", "nan"):
            reason, status = "unknown_domain", "proven"
        else:
            reason, status = UNASSIGNED_REASON, "unknown"
        exclusions.append({"stable_id": sid, "reason": reason, "status": status})

    # Realized per-ID domain label for every ASSIGNED sample (persisted so runtime never recomputes).
    domain_by_id = {sid: str(domains_s[id_to_index[sid]]) for sid in assigned}

    spec = LeafSpec(
        benchmark=benchmark, regime=regime, seed=int(seed), holdout_label=str(label),
        target_unit="sample", domain_basis=str(group_kind), group_kind=str(group_kind),
        has_target=bool(has_target), params=params, partitions=partitions, partition_stats=stats,
        domains=domain_by_id, exclusions=exclusions,
        audit=[e for e in audit_events if e.get("kind") in ("purge", "stratification_fallback")],
    )
    return spec, eligible


def _dense_partition_stats(pids: list[int], fold_of: dict[int, int], class_sets: dict[int, set[int]]) -> dict[str, Any]:
    folds = [str(fold_of[int(p)]) for p in pids]
    class_counts: dict[str, int] = {}
    for p in pids:
        for c in class_sets.get(int(p), set()):
            class_counts[str(c)] = class_counts.get(str(c), 0) + 1
    return {
        "n": len(pids),
        "domains": sorted(set(folds), key=lambda s: (0, int(s)) if s.isdigit() else (1, s)),
        "domain_counts": dict(Counter(folds)),
        "class_counts": class_counts,
    }


def build_dense_leaf(
    benchmark: str,
    regime: str,
    seed: int,
    *,
    cfg: Any,
    bench: Any,
    params: dict[str, Any],
    audit_events: list[dict[str, Any]],
    all_patch_ids: list[int],
    fold_of: dict[int, int],
    class_sets: dict[int, set[int]],
    patch_latlon: dict[int, Any],
) -> tuple[LeafSpec, list[str]]:
    """Turn one realized dense (PASTIS) fold config into a canonical, patch-level LeafSpec.

    Split units are ORIGINAL PATCHES only. Dense purge operates in a per-config index space, so we
    do not assert ``purged`` for dense: a NaN-coordinate patch proves ``no_coords``; any other
    unassigned patch is ``unassigned_current_behavior`` with status ``unknown`` (never guessed).
    """
    def resolve(explicit: Any, folds: Any) -> list[int]:
        if explicit is not None:
            return sorted(int(p) for p in explicit)
        return sorted(int(p) for p in bench.patch_ids(set(folds)))

    parts = {
        "train": resolve(cfg.train_patches, cfg.train_folds),
        "val": resolve(cfg.val_patches, cfg.val_folds),
        "test": resolve(cfg.test_patches, cfg.test_folds),
        "source_val": sorted(int(p) for p in (cfg.source_val_patches or set())),
        "source_test": sorted(int(p) for p in (cfg.source_test_patches or set())),
    }
    partitions = {p: [str(pid) for pid in pids] for p, pids in parts.items()}
    stats = {p: _dense_partition_stats(pids, fold_of, class_sets) for p, pids in parts.items()}

    eligible = [str(p) for p in all_patch_ids]
    assigned = set().union(*(set(v) for v in partitions.values())) if partitions else set()

    exclusions: list[dict[str, Any]] = []
    for pid_s in eligible:
        if pid_s in assigned:
            continue
        ll = patch_latlon.get(int(pid_s))
        if ll is None or not np.all(np.isfinite(np.asarray(ll, dtype=float))):
            reason, status = "no_coords", "proven"
        else:
            reason, status = UNASSIGNED_REASON, "unknown"
        exclusions.append({"stable_id": pid_s, "reason": reason, "status": status})

    # Dense per-ID domain label = the patch's fold (persisted for audit + fold-consistency checks).
    domain_by_id = {pid_s: str(fold_of[int(pid_s)]) for pid_s in assigned}

    spec = LeafSpec(
        benchmark=benchmark, regime=regime, seed=int(seed), holdout_label=str(cfg.label),
        target_unit="patch", domain_basis=str(cfg.group_kind), group_kind=str(cfg.group_kind),
        has_target=bool(cfg.has_target), params=params, partitions=partitions, partition_stats=stats,
        domains=domain_by_id, exclusions=exclusions,
        audit=[e for e in audit_events if e.get("kind") in ("purge", "stratification_fallback")],
    )
    return spec, eligible


# --------------------------------------------------------------------------- #
# Reading / runtime mapping
# --------------------------------------------------------------------------- #
def is_complete(ldir: str | os.PathLike) -> bool:
    """A leaf is complete iff all three canonical files exist (manifest.json is written LAST)."""
    p = Path(ldir)
    return all((p / name).is_file() for name in ("manifest.json", "assignments.csv", "exclusions.csv"))


def read_manifest(ldir: str | os.PathLike) -> dict[str, Any]:
    path = Path(ldir) / "manifest.json"
    manifest = json.loads(path.read_text())
    _require_schema_version(manifest, f"leaf manifest {path}")  # parser-contract check on every read
    return manifest


def _read_assignment_rows(ldir: str | os.PathLike) -> list[tuple[str, str, str]]:
    path = Path(ldir) / "assignments.csv"
    rows: list[tuple[str, str, str]] = []
    with open(path, newline="") as fh:
        reader = csv.reader(fh)
        header = next(reader, None)
        if header != _ASSIGN_HEADER:
            raise SplitArtifactError(f"malformed assignments header in {path}: {header}")
        for row in reader:
            if len(row) != 3:
                raise SplitArtifactError(f"malformed assignments row in {path}: {row}")
            if not row[2].strip():  # a blank/whitespace-only domain is a malformed record, never valid
                raise SplitArtifactError(f"blank/whitespace-only domain for id {row[0]!r} in {path}")
            rows.append((row[0], row[1], row[2]))
    return rows


def read_assignments(ldir: str | os.PathLike) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {part: [] for part in PARTITIONS}
    for sid, part, _domain in _read_assignment_rows(ldir):
        if part not in out:
            raise SplitArtifactError(f"unknown partition {part!r} in {Path(ldir) / 'assignments.csv'}")
        out[part].append(sid)
    return out


def read_domains(ldir: str | os.PathLike) -> dict[str, str]:
    """Per-ID domain label from assignments.csv. Rejects duplicate/conflicting domain records."""
    out: dict[str, str] = {}
    for sid, _part, domain in _read_assignment_rows(ldir):
        if sid in out and out[sid] != domain:
            raise SplitArtifactError(f"conflicting domain records for id {sid!r}: {out[sid]!r} vs {domain!r}")
        if sid in out:
            raise SplitArtifactError(f"duplicate domain record for id {sid!r}")
        out[sid] = domain
    return out


def read_exclusions(ldir: str | os.PathLike) -> list[dict[str, str]]:
    path = Path(ldir) / "exclusions.csv"
    rows: list[dict[str, str]] = []
    with open(path, newline="") as fh:
        reader = csv.reader(fh)
        header = next(reader, None)
        if header != _EXCL_HEADER:
            raise SplitArtifactError(f"malformed exclusions header in {path}: {header}")
        for row in reader:
            if len(row) != 3:
                raise SplitArtifactError(f"malformed exclusions row in {path}: {row}")
            rows.append({"stable_id": row[0], "reason": row[1], "status": row[2]})
    return rows


def map_to_indices(assignments: dict[str, list[str]], id_map: dict[str, Any]) -> dict[str, np.ndarray]:
    """Resolve stable IDs to current row indices (tabular) or patch IDs (dense).

    ``id_map`` maps every current stable id -> its index/target. Refuses unknown ids and ids that
    appear in more than one partition. This is the structural runtime identity gate -- there is no
    hash comparison.
    """
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


def _check_leaf_identity(
    ldir: Path, manifest: dict[str, Any],
    benchmark: str | None, regime: str | None, seed: int | None, holdout: str | None,
) -> None:
    """The manifest must agree with the canonical path and with the requested leaf identity."""
    parts = ldir.parts
    if len(parts) < 4:
        raise SplitArtifactError(f"leaf path is not canonical (<benchmark>/<regime>/<seed>/<holdout>): {ldir}")
    p_benchmark, p_regime, p_seed, p_holdout_dir = parts[-4], parts[-3], parts[-2], parts[-1]
    for fld, mval, pval in (
        ("benchmark", manifest.get("benchmark"), p_benchmark),
        ("regime", manifest.get("regime"), p_regime),
        ("seed", manifest.get("seed"), p_seed),
        ("holdout_dirname", manifest.get("holdout_dirname"), p_holdout_dir),
    ):
        if str(mval) != str(pval):
            raise SplitArtifactError(f"manifest {fld}={mval!r} disagrees with canonical path component {pval!r}")
    if holdout_dirname(str(manifest.get("holdout"))) != p_holdout_dir:
        raise SplitArtifactError(
            f"manifest holdout {manifest.get('holdout')!r} does not encode to directory {p_holdout_dir!r}"
        )
    requested = (("benchmark", benchmark), ("regime", regime), ("seed", seed), ("holdout", holdout))
    for fld, want in requested:
        if want is None:
            continue
        got = manifest.get(fld)
        if str(got) != str(want):  # seed compared as string too, avoiding int(None)
            raise SplitArtifactError(f"requested {fld}={want!r} does not match leaf manifest {fld}={got!r}")


SPLIT_REF_FILE = "split_ref.json"

#: The single canonical, committed location of split artifacts, relative to the repo (never a
#: machine-specific absolute path).
SPLITS_LOCATION = "data/splits"

#: The scope of what split_ref.json / run_manifest.json reference.
SPLIT_SCOPE = "regime_partitions_only"

#: Explicit scope statement written into every split_ref.json.
SPLIT_REF_NOTE = (
    "These are the REGIME-LEVEL train/val/test/source_val/source_test partitions consumed for this "
    "(model, benchmark) pair, referenced by canonical relative path under data/splits/. Target-budget "
    "80/20 draws, nested few-shot ordering, and probe initialization are NOT stored here -- they "
    "remain runtime operations derived deterministically from the experiment seed."
)


def benchmark_splits_exist(root: str | os.PathLike, benchmark: str) -> bool:
    return benchmark_dir(root, benchmark).is_dir()


def list_leaves(root: str | os.PathLike, benchmark: str, regime: str, seed: int) -> list[str]:
    """Exact holdout labels this (benchmark, regime, seed) yielded, per its generation.json.

    ``generation.json`` is the authority. This validates it and hard-fails on:
      * a missing/malformed one -- splits were not generated for this cell;
      * an identity mismatch (its benchmark/regime/seed disagree with the requested cell);
      * a non-empty ``regime_problems`` record (a declared regime did not run cleanly);
      * duplicate yielded holdout labels;
      * a declared-yielded holdout whose leaf is incomplete/missing (never silently skipped).
    A generated regime that legitimately dropped some LODO targets but still yielded valid leaves
    (no regime_problem) is fine -- the dropped labels live in ``dropped_holdouts``, not here.
    """
    sdir = regime_seed_dir(root, benchmark, regime, seed)
    gen_path = sdir / "generation.json"
    if not gen_path.is_file():
        raise SplitArtifactError(
            f"no generation.json under {sdir} -- canonical splits were not generated for "
            f"{benchmark}/{regime}/seed={seed} (run tools/generate_splits.py)"
        )
    try:
        gen = json.loads(gen_path.read_text())
    except (ValueError, OSError) as exc:
        raise SplitArtifactError(f"malformed generation.json at {gen_path}: {exc}") from exc
    _require_schema_version(gen, f"generation.json at {gen_path}")
    for fld, want in (("benchmark", benchmark), ("regime", regime), ("seed", seed)):
        got = gen.get(fld)
        ok = (int(got) == int(want)) if fld == "seed" else (str(got) == str(want))
        if not ok:
            raise SplitArtifactError(f"generation.json {fld}={got!r} disagrees with requested {want!r} ({gen_path})")
    problems = gen.get("regime_problems") or []
    if problems:
        raise SplitArtifactError(
            f"generation.json for {benchmark}/{regime}/seed={seed} records {len(problems)} regime "
            f"problem(s): {problems[:3]} -- the regime did not run cleanly; refuse to consume"
        )
    yielded = [str(x) for x in gen.get("yielded_holdouts", [])]
    if len(set(yielded)) != len(yielded):
        raise SplitArtifactError(f"generation.json has duplicate yielded holdout labels: {yielded}")
    for label in yielded:
        ldir = sdir / holdout_dirname(label)
        if not is_complete(ldir):
            raise SplitArtifactError(
                f"generation.json declares holdout {label!r} yielded, but its leaf is incomplete or "
                f"missing (no complete manifest/assignments/exclusions): {ldir}"
            )
    return yielded


def write_split_ref(results_dir: str | os.PathLike, *, benchmark: str, consumed: list[str]) -> Path:
    """Record the canonical regime-level split leaves this pair consumed (relative paths only)."""
    payload = {
        "schema_version": SCHEMA_VERSION,
        "benchmark": benchmark,
        "splits_location": SPLITS_LOCATION,  # canonical committed location, relative -- never absolute
        "scope": SPLIT_SCOPE,
        "consumed_leaves": sorted(set(consumed)),  # e.g. "cropharvest/geographic_ood/0/kenya"
        "scope_note": SPLIT_REF_NOTE,
    }
    path = Path(results_dir) / SPLIT_REF_FILE
    _atomic_write(path, (json.dumps(payload, indent=2) + "\n").encode())
    return path


def load_tabular_splits(
    root: str | os.PathLike, benchmark: str, sample_ids: Any, bench: Any, bench_mod: Any,
    split_regimes: list[str], seeds: list[int],
) -> tuple[list[tuple], list[str]]:
    """Consume tabular splits from data/splits/ as the runtime's split_specs (no construction).

    Partitions AND the per-sample ``domains`` array (used only for worst-group scoring) come entirely
    from the canonical artifacts -- the runtime never calls assign_domains/KMeans. ``bench``/``bench_mod``
    are unused (kept for signature stability). Returns (split_specs, consumed_leaf_paths). Hard-fails
    on any missing/invalid artifact, a regime that recorded problems, or a requested regime that
    yielded zero leaves.
    """
    del bench, bench_mod  # domains now come from the artifact, not from re-labeling the benchmark

    if not benchmark_splits_exist(root, benchmark):
        raise SplitArtifactError(
            f"no canonical splits under {benchmark_dir(root, benchmark)} -- run tools/generate_splits.py first"
        )
    ids_list = [str(s) for s in np.asarray(sample_ids).tolist()]
    id_map = {s: i for i, s in enumerate(ids_list)}
    n = len(ids_list)
    split_specs: list[tuple] = []
    consumed: list[str] = []
    for seed in seeds:
        for regime in split_regimes:
            labels = list_leaves(root, benchmark, regime, seed)
            if not labels:
                raise SplitArtifactError(
                    f"requested regime {regime!r} yielded zero leaves for {benchmark}/seed={seed} -- refuse to run"
                )
            for label in labels:
                ldir = leaf_dir(root, benchmark, regime, seed, label)
                idx = load_split_indices(ldir, id_map, benchmark=benchmark, regime=regime, seed=seed, holdout=label)
                # Reconstruct the per-sample domains array from the artifact (assigned ids only; the
                # placeholder for unassigned rows is never indexed -- test/val/source are all assigned).
                domains = np.full(n, "__unassigned__", dtype=object)
                for sid, dom in read_domains(ldir).items():
                    domains[id_map[sid]] = dom
                m = read_manifest(ldir)
                split_specs.append((
                    seed, regime, label, idx["train"], idx["val"], idx["test"], domains,
                    bool(m["has_target"]), str(m["domain_basis"]), idx["source_val"], idx["source_test"],
                ))
                consumed.append(f"{benchmark}/{regime}/{seed}/{holdout_dirname(label)}")
    return split_specs, consumed


def load_dense_split(
    root: str | os.PathLike, benchmark: str, regime: str, seed: int, holdout: str,
    id_map: dict[str, Any], patch_fold: dict[int, int],
) -> dict[str, Any]:
    """Load one PASTIS dense leaf: folds from manifest params, patch sets from validated assignments.

    Also verifies every assigned patch's CURRENT fold (``patch_fold``) is compatible with the
    manifest's corresponding fold set -- so changed/inconsistent patch-fold membership is refused
    here rather than silently omitted by downstream stream filtering. source_val/source_test patches
    are drawn from the train pool, so they are checked against the train fold set.
    """
    ldir = leaf_dir(root, benchmark, regime, seed, holdout)
    idx = load_split_indices(ldir, id_map, benchmark=benchmark, regime=regime, seed=seed, holdout=holdout)
    m = read_manifest(ldir)
    params = m.get("params", {})

    def _folds(key: str) -> set[int]:
        return {int(f) for f in params.get(key, [])}

    def _patches(part: str) -> set[int]:
        return {int(p) for p in idx[part].tolist()}

    train_folds, val_folds, test_folds = _folds("train_folds"), _folds("val_folds"), _folds("test_folds")
    for part, fset in (("train", train_folds), ("val", val_folds), ("test", test_folds),
                       ("source_val", train_folds), ("source_test", train_folds)):
        for pid in idx[part].tolist():
            cur = patch_fold.get(int(pid))
            if cur is None:
                raise SplitArtifactError(f"patch {pid} (partition {part!r}) has no current fold in the benchmark")
            if cur not in fset:
                raise SplitArtifactError(
                    f"patch {pid} current fold {cur} is not in the manifest {part!r} fold set {sorted(fset)} "
                    f"({benchmark}/{regime}/{seed}/{holdout}) -- patch-fold membership changed"
                )
    return {
        "label": str(m["holdout"]),
        "train_folds": train_folds, "val_folds": val_folds, "test_folds": test_folds,
        "train_patches": _patches("train"), "val_patches": _patches("val"), "test_patches": _patches("test"),
        "source_val_patches": _patches("source_val") or None, "source_test_patches": _patches("source_test") or None,
        "has_target": bool(m["has_target"]), "group_kind": str(m["group_kind"]),
    }


def load_dense_splits(
    root: str | os.PathLike, benchmark: str, patch_fold: dict[int, int], split_regimes: list[str], seeds: list[int],
) -> tuple[dict[int, list[tuple[str, Any]]], list[str]]:
    """Consume PASTIS patch-level splits for all (regime, seed) as ready-to-run DenseSplit configs.

    ``patch_fold`` is the CURRENT benchmark patch->fold mapping; it defines the eligible patch set and
    drives the fold-consistency check. Returns ({seed: [(regime, DenseSplit)]}, consumed_leaf_paths).
    Hard-fails on any missing/invalid artifact, a regime that recorded problems, or a requested regime
    that yielded zero leaves.
    """
    from evals.regimes.base import DenseSplit  # local import: base never imports this module

    if not benchmark_splits_exist(root, benchmark):
        raise SplitArtifactError(
            f"no canonical splits under {benchmark_dir(root, benchmark)} -- run tools/generate_splits.py first"
        )
    id_map = {str(int(p)): int(p) for p in patch_fold}
    by_seed: dict[int, list[tuple[str, Any]]] = {}
    consumed: list[str] = []
    for seed in seeds:
        configs: list[tuple[str, Any]] = []
        for regime in split_regimes:
            labels = list_leaves(root, benchmark, regime, seed)
            if not labels:
                raise SplitArtifactError(
                    f"requested regime {regime!r} yielded zero leaves for {benchmark}/seed={seed} -- refuse to run"
                )
            for label in labels:
                d = load_dense_split(root, benchmark, regime, seed, label, id_map, patch_fold)
                cfg = DenseSplit(
                    d["label"], d["train_folds"], d["val_folds"], d["test_folds"],
                    train_patches=d["train_patches"], val_patches=d["val_patches"], test_patches=d["test_patches"],
                    source_val_patches=d["source_val_patches"], source_test_patches=d["source_test_patches"],
                    has_target=d["has_target"], group_kind=d["group_kind"],
                )
                configs.append((regime, cfg))
                consumed.append(f"{benchmark}/{regime}/{seed}/{holdout_dirname(label)}")
        by_seed[seed] = configs
    return by_seed, consumed


def load_split_indices(
    ldir: str | os.PathLike,
    id_map: dict[str, Any],
    *,
    benchmark: str | None = None,
    regime: str | None = None,
    seed: int | None = None,
    holdout: str | None = None,
) -> dict[str, np.ndarray]:
    """Full runtime consume path for one leaf.

    Refuses an incomplete leaf, validates the manifest against the canonical path (and the requested
    identity), reads all three files, and enforces the SAME invariants as generation over the CURRENT
    eligible population (five-partition disjointness, no duplicates, assignments/exclusions disjoint,
    complete accounting, valid schemas/statuses, no unknown ids) before mapping stable ids to indices.
    """
    ldir = Path(ldir)
    if not is_complete(ldir):
        raise SplitArtifactError(
            f"leaf is incomplete (missing manifest.json / assignments.csv / exclusions.csv) -- "
            f"ignored, regenerate: {ldir}"
        )
    manifest = read_manifest(ldir)
    _check_leaf_identity(ldir, manifest, benchmark, regime, seed, holdout)

    assignments = read_assignments(ldir)  # validates CSV schema/header
    exclusions = read_exclusions(ldir)    # validates CSV schema/header

    for part, ids in assignments.items():
        for sid in ids:
            if sid not in id_map:
                raise SplitArtifactError(f"unknown id {sid!r} in partition {part!r} not present in current benchmark")
    for row in exclusions:
        if str(row["stable_id"]) not in id_map:
            raise SplitArtifactError(f"unknown excluded id {row['stable_id']!r} not present in current benchmark")

    # Full structural + complete-accounting invariants over the CURRENT eligible ids (overlap/dup
    # reported here), THEN the per-ID domain records + coverage.
    assigned_sets = _check_partitions_accounting(assignments, exclusions, [str(k) for k in id_map])
    domain_by_id = read_domains(ldir)  # validates domain schema + duplicate/conflicting records
    _check_domain_coverage(set().union(*assigned_sets.values()) if assigned_sets else set(), domain_by_id)
    return map_to_indices(assignments, id_map)
