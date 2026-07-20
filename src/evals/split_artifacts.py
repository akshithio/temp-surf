"""Frozen split assignments under ``data/splits/`` plus one central log ``data/logs/splits.json``.

Per split leaf: ``assignments.csv`` (always) plus, for ``geographic_ood`` headline targets, a sibling
``label_access.csv`` (the frozen label-blind label-access order). One JSON log for the whole run:

    data/splits/<benchmark>/<regime>/<seed>/<holdout>/assignments.csv
    data/splits/<benchmark>/geographic_ood/<seed>/<target>/label_access.csv   # headline targets only
    data/logs/splits.json

``assignments.csv`` is the primary file in a leaf (columns ``stable_id, partition, status, domain,
reason``); ``geographic_ood`` headline-target leaves ALSO carry a sibling ``label_access.csv`` -- the
frozen, label-blind label-access order (see ``LABEL_ACCESS_HEADER``). ``assignments.csv`` carries
EVERY eligible stable id exactly once:

  * assigned -- ``partition`` is one of the five v2 partitions, ``status`` ``assigned``, blank reason;
  * purged   -- deliberately removed geography with a proven cause; blank ``partition``, ``status``
                ``purged``, ``reason`` either ``purged_near_ood`` (within the source<->target distance
                purge) or ``inside_buffered_target_footprint`` (inside the target's buffered
                convex-hull footprint). The two are counted separately in the log: one bounds distance
                to the nearest labelled target sample, the other establishes territorial exclusion;
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

#: Stable removal reasons. Both are ``STATUS_PURGED`` -- they are deliberate geographic removals with
#: a proven cause, NOT leftovers -- but they answer different scientific questions and must stay
#: distinguishable in the artifact: one bounds distance to the nearest labelled target sample, the
#: other establishes territorial exclusion from the target's buffered footprint.
REASON_PURGED_NEAR_OOD = "purged_near_ood"
REASON_INSIDE_FOOTPRINT = "inside_buffered_target_footprint"
PURGE_REASONS = (REASON_PURGED_NEAR_OOD, REASON_INSIDE_FOOTPRINT)

LOG_FILENAME = "splits.json"
_PART_RANK = {p: i for i, p in enumerate(PARTITIONS)}

#: Second per-leaf artifact, ``geographic_ood`` headline targets ONLY: the frozen, label-blind
#: label-access order the runtime loads instead of regenerating an ordering inside probing. One row
#: per unit in the source label pool, the target label pool, and the target test set. The source pool
#: carries ONE nested order -- ``source_rank`` -- so every fixed-budget allocation point takes its
#: source share as a PREFIX of that same order and the five points are strictly nested rather than
#: independent draws; ``target_rank`` orders the target pool (allocation + additive selection);
#: target_test units carry no rank. Integrity is BOTH structural (population-correct / complete /
#: contiguous, validated at load against the frozen split) AND cryptographic: the file's SHA-256 is
#: recorded in the central log beside the assignments checksum and re-verified at load, because a
#: DIFFERENT valid permutation over the same id sets passes every structural check while silently
#: changing every allocation and additive experiment.
LABEL_ACCESS_FILENAME = "label_access.csv"
LABEL_ACCESS_HEADER: list[str] = ["stable_id", "population", "source_rank", "target_rank"]
SOURCE_RANK_COLS = ("source_rank",)
LABEL_ACCESS_REGIME = "geographic_ood"
#: The ONE canonical label-access count set (target few-shot AND fixed-total source removal); every
#: module referencing these counts imports this, never a re-literal. A headline target that cannot
#: support every one of these is a hard preprocessing failure -- never clamped.
LABEL_ACCESS_COUNTS: tuple[int, ...] = (5, 10, 25, 50)
#: The label-budget UNIT contract. Tabular benchmarks allocate whole samples; dense PASTIS allocates
#: whole PATCHES (never a fraction of a patch). Rows carry the unit verbatim; the manifest contract and
#: the completion semantic-validator BOTH resolve the expected unit per benchmark via
#: ``label_access_unit`` -- never a hardcoded literal -- so PASTIS validates patch units, not samples.
LABEL_ACCESS_TABULAR_UNIT = "samples"
LABEL_ACCESS_DENSE_UNIT = "patches"
#: Benchmarks whose label-access unit is the whole PATCH (dense segmentation). Everything else is tabular
#: (whole samples). Kept explicit -- and tiny -- so the unit contract has one obvious home.
DENSE_LABEL_ACCESS_BENCHMARKS: frozenset[str] = frozenset({"pastis"})


def label_access_unit(benchmark: str) -> str:
    """The canonical label-access unit for a benchmark: ``patches`` for dense PASTIS, else ``samples``."""
    return LABEL_ACCESS_DENSE_UNIT if str(benchmark) in DENSE_LABEL_ACCESS_BENCHMARKS else LABEL_ACCESS_TABULAR_UNIT

#: Canonical label-access routes (single source of truth). Stage 2 fits these; Stage 5 contrasts them.
#: The HEADLINE experiment is the fixed-budget ALLOCATION curve: at fraction ``f`` the cell trains on
#: ``B - k`` source units plus ``k`` target units (``k = round_half_up(f * B)``), so TOTAL supervision is
#: pinned at the benchmark-common budget ``B_d`` and only its source/target COMPOSITION moves. f=0 and
#: f=1 are the two endpoints of that single curve -- they are NOT separate "matched-source" /
#: "matched-target" routes, which is why the old two-order draw is gone: every point slices a prefix of
#: the SAME frozen ``source_rank`` / ``target_rank``, so the curve is strictly nested.
#: The ADDITIVE routes are a separate operational question -- hold the COMPLETE source pool fixed and add
#: k target units on top -- and must never be pooled into the fixed-budget allocation claim.
ROUTE_FIXED_BUDGET_ALLOCATION = "fixed_budget_allocation"  # one per ALLOCATION_PERCENTS
ROUTE_SOURCE_PLUS_TARGET = "source_plus_target"            # one per LABEL_ACCESS_COUNTS (appendix)
ROUTE_TARGET_ONLY_FULL = "target_only_full"
ROUTE_SOURCE_PLUS_TARGET_FULL = "source_plus_target_full"
#: The 4 canonical route NAMES, in emission order (fixed_budget_allocation / source_plus_target expand
#: across their budget axes at runtime). The manifest contract and any route enumeration import this.
LABEL_ACCESS_ROUTES: tuple[str, ...] = (
    ROUTE_FIXED_BUDGET_ALLOCATION, ROUTE_SOURCE_PLUS_TARGET,
    ROUTE_TARGET_ONLY_FULL, ROUTE_SOURCE_PLUS_TARGET_FULL,
)
#: The frozen allocation fractions as INTEGER PERCENT -- the ``label_budget`` axis for allocation rows.
#: A percent and an additive COUNT can collide numerically (allocation@25 vs source_plus_target@25); they
#: stay distinct because the cell key carries ``label_access_route`` beside ``label_budget``, so no
#: widening of the result schema is needed to tell the two 25s apart.
ALLOCATION_PERCENTS: tuple[int, ...] = (0, 25, 50, 75, 100)

#: The in-distribution source reference (random_id source) -- a cross-regime contrast anchor resolved at
#: aggregation; never a label-access route/fit.
ANCHOR_SOURCE_ID_REFERENCE = "source_ID_reference"
#: The ordinary geographic full-source row (E1: geographic_ood / budget_type=source / label_budget=1.0 /
#: target_test). Label access does NOT refit it -- the completion check requires the existing row and the
#: contrasts read it -- so the complete-source probe is fit exactly once per cell, not twice.
ANCHOR_GEOGRAPHIC_FULL_SOURCE = "geographic_full_source"
#: Subtrahend sentinel: the f=0 endpoint of the allocation curve itself (B_d source units, no target).
#: Distinct from ANCHOR_GEOGRAPHIC_FULL_SOURCE, which trains on the COMPLETE source pool (N_S >= B_d).
ALLOCATION_BASELINE = "fixed_budget_allocation@0"

#: The scientific contrast contract: each a (name, minuend, subtrahend) triple and a DISTINCT deployment
#: question that must NOT be merged. Computed within (benchmark, model, target, seed, frozen target_test,
#: frozen orders) on the ``target_test`` evaluation ONLY. ``allocation_effect`` expands over the non-zero
#: ALLOCATION_PERCENTS (so f=1.0 vs f=0 -- the whole-budget reallocation -- is one of its points, not a
#: separate contrast); ``additive_target_label_gain`` expands over LABEL_ACCESS_COUNTS.
LABEL_ACCESS_CONTRASTS: tuple[tuple[str, str, str], ...] = (
    ("target_label_advantage", ROUTE_TARGET_ONLY_FULL, ANCHOR_GEOGRAPHIC_FULL_SOURCE),
    ("allocation_effect", ROUTE_FIXED_BUDGET_ALLOCATION, ALLOCATION_BASELINE),
    ("additive_target_label_gain", ROUTE_SOURCE_PLUS_TARGET, ANCHOR_GEOGRAPHIC_FULL_SOURCE),
    ("full_supervision_gain", ROUTE_SOURCE_PLUS_TARGET_FULL, ANCHOR_GEOGRAPHIC_FULL_SOURCE),
)

#: Evaluation-split identity for the label-access suite: every route is scored on the frozen
#: ``target_test`` and nothing else. ``complete_target`` remains defined for the LEGACY target-budget
#: sweep (spatial_cluster_ood) but is no longer a label-access row -- the canonical comparison population
#: is the frozen ``target_test``.
EVAL_TARGET_TEST = "target_test"
EVAL_COMPLETE_TARGET = "complete_target"
LABEL_ACCESS_EVAL_SPLITS: tuple[str, ...] = (EVAL_TARGET_TEST,)


def allocation_target_count(percent: int, budget: int) -> int:
    """``k = round_half_up(f * B)`` -- the target share of a fixed-budget allocation point. Explicit
    half-up rounding (never Python's banker's rounding), so k is reproducible across platforms and the
    source share is always exactly ``B - k``."""
    return int((int(percent) / 100.0) * int(budget) + 0.5)


def label_access_contract(
    *, enabled: bool, benchmark: str, percents: tuple[int, ...] = ALLOCATION_PERCENTS,
    counts: tuple[int, ...] = LABEL_ACCESS_COUNTS, full_target_reference: bool = True,
    full_combined_reference: bool = True, controlled_budget_cap: int | None = None,
) -> dict[str, Any]:
    """The readable label-access contract recorded in the run manifest: whether the suite is config-active
    (geographic_ood requested), the requested allocation fractions / additive counts / optional full
    references / controlled total-budget cap, the canonical routes and evaluation split, and the
    benchmark's label unit (``patches`` for dense PASTIS, else ``samples``). Derived entirely from the
    canonical constants and the caller's direct config so the manifest can never drift from the runtime --
    and so two machines running different caps or fractions are distinguishable from the manifest alone."""
    return {
        "enabled": bool(enabled),
        "allocation_percents": [int(f) for f in percents],
        "additive_counts": [int(k) for k in counts],
        "full_target_reference": bool(full_target_reference),
        "full_combined_reference": bool(full_combined_reference),
        "controlled_budget_cap": (None if controlled_budget_cap is None else int(controlled_budget_cap)),
        "routes": list(LABEL_ACCESS_ROUTES),
        "evaluation_splits": list(LABEL_ACCESS_EVAL_SPLITS),
        "unit": label_access_unit(benchmark),
    }


def label_access_expected_rows(
    percents: tuple[int, ...] = ALLOCATION_PERCENTS, counts: tuple[int, ...] = LABEL_ACCESS_COUNTS,
    *, full_target_reference: bool = True, full_combined_reference: bool = True,
) -> list[tuple[str, int, str]]:
    """The (label_access_route, label_budget, evaluation_split) rows one fully-eligible geographic_ood
    headline cell must emit, all scored on ``target_test``: one allocation fit per requested fraction,
    one additive fit per requested count, and the two optional full references. There is deliberately NO
    ``source_only`` row (the ordinary geographic full-source E1 row is reused instead) and NO
    complete-target diagnostic. Single source of truth for expected-key / completeness planning AND the
    sweep's emitted set (a test locks the two together)."""
    rows = [(ROUTE_FIXED_BUDGET_ALLOCATION, int(f), EVAL_TARGET_TEST) for f in percents]
    rows += [(ROUTE_SOURCE_PLUS_TARGET, int(k), EVAL_TARGET_TEST) for k in counts]
    if full_target_reference:
        rows.append((ROUTE_TARGET_ONLY_FULL, 0, EVAL_TARGET_TEST))
    if full_combined_reference:
        rows.append((ROUTE_SOURCE_PLUS_TARGET_FULL, 0, EVAL_TARGET_TEST))
    return rows
POP_SOURCE, POP_TARGET_POOL, POP_TARGET_TEST = "source", "target_pool", "target_test"
_LA_POP_RANK = {POP_SOURCE: 0, POP_TARGET_POOL: 1, POP_TARGET_TEST: 2}


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


def label_access_path(root: str | os.PathLike, benchmark: str, seed: int, holdout: str) -> Path:
    """The geographic_ood label-access CSV -- sibling of ``assignments.csv`` in the same leaf dir."""
    return leaf_dir(root, benchmark, LABEL_ACCESS_REGIME, seed, holdout) / LABEL_ACCESS_FILENAME


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
    footprint: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the central-log entry for one leaf (everything but the CSV's SHA-256, which the generator
    fills in after writing the file). Carries per-partition stratification stats, not stable ids.

    ``class_by_id`` maps each ASSIGNED id to its class label(s): a one-element list for a tabular
    sample, and the patch's sorted class-presence set for a PASTIS patch (so dense class_counts are
    patch-level presence within each partition, never pixel totals)."""
    status_counts = {s: 0 for s in STATUSES}
    exclusion_counts: dict[str, int] = {}
    purge_counts: dict[str, int] = {}
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
        elif r["status"] == STATUS_PURGED:
            purge_counts[r["reason"]] = purge_counts.get(r["reason"], 0) + 1
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
        # Broken out by cause: a distance purge and a territorial footprint exclusion are different
        # scientific claims and must stay countable apart in the central log.
        "purge_counts": dict(sorted(purge_counts.items())),
        # The realized footprint SPECIFICATION for this target -- recorded whenever the mask ran, even
        # if it excluded nothing, so a zero-exclusion target is provably masked rather than unmasked.
        "footprint": dict(footprint) if footprint else None,
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
    purged, footprint = _removed_by_reason(audit_events, sample_ids)
    footprint_spec = _footprint_spec(audit_events)

    rows: list[dict[str, str]] = []
    for sid in (str(s) for s in sample_ids.tolist()):
        dom = str(domains_s[id_index[sid]])
        if sid in partition_of:
            rows.append(_row(sid, partition_of[sid], STATUS_ASSIGNED, dom, ""))
        elif sid in purged:
            rows.append(_row(sid, "", STATUS_PURGED, dom, REASON_PURGED_NEAR_OOD))
        elif sid in footprint:
            rows.append(_row(sid, "", STATUS_PURGED, dom, REASON_INSIDE_FOOTPRINT))
        elif dom in ("unknown", "nan"):
            rows.append(_row(sid, "", STATUS_EXCLUDED, dom, "unknown_domain"))
        else:
            rows.append(_row(sid, "", STATUS_EXCLUDED, dom, "unassigned"))

    class_by_id = {sid: [str(labels_s[id_index[sid]])] for sid in partition_of}
    summary = _leaf_summary(
        benchmark, regime, seed, str(split.label), rows, target_unit="sample",
        group_kind=str(split.group_kind), has_target=split.has_target,
        supports_target_labels=split.supports_target_labels, target_role=str(split.target_role),
        purge_km=purge_km, class_by_id=class_by_id, footprint=footprint_spec,
    )
    return rows, summary


#: Footprint-specification fields lifted verbatim from the regime's audit event into the log entry.
#: Everything needed to RECONSTRUCT the decision boundary and re-verify it independently.
FOOTPRINT_SPEC_FIELDS = (
    "crs", "buffer_m", "quad_segs", "hull_policy", "hull_wkt", "footprint_sha256", "n_excluded",
)


def _footprint_spec(audit_events) -> dict[str, Any] | None:
    """The realized footprint specification for this leaf, or None if the mask never ran.

    Present even when nothing was excluded: "masked, zero hits" and "never masked" are different
    claims about the split and must be distinguishable in the artifact.
    """
    for ev in audit_events:
        if ev.get("kind") == "footprint_exclusion":
            return {k: ev[k] for k in FOOTPRINT_SPEC_FIELDS if k in ev}
    return None


def _removed_by_reason(audit_events, unit_ids) -> tuple[set[str], set[str]]:
    """``(purged_near_target, inside_target_footprint)`` id sets proven by the regime's audit events.

    Kept separate so a footprint exclusion can never collapse into ``excluded/unassigned``, which
    would erase the scientific reason the item was removed.
    """
    unit_ids = np.asarray(unit_ids, dtype=object)
    purged: set[str] = set()
    footprint: set[str] = set()
    for ev in audit_events:
        kind = ev.get("kind")
        if kind == "purge":
            sink, idx = purged, ev.get("purged_indices", ev.get("purged_train_indices", []))
        elif kind == "footprint_exclusion":
            sink, idx = footprint, ev.get("excluded_indices", [])
        else:
            continue
        for i in idx:
            if 0 <= int(i) < len(unit_ids):
                sink.add(str(unit_ids[int(i)]))
    return purged, footprint


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
    purged, footprint = _removed_by_reason(audit_events, [str(int(p)) for p in all_patch_ids])
    footprint_spec = _footprint_spec(audit_events)

    rows: list[dict[str, str]] = []
    for pid in all_patch_ids:
        sid = str(int(pid))
        dom = str(domain_of.get(int(pid), ""))
        if sid in partition_of:
            rows.append(_row(sid, partition_of[sid], STATUS_ASSIGNED, dom, ""))
        elif sid in purged:
            rows.append(_row(sid, "", STATUS_PURGED, dom, REASON_PURGED_NEAR_OOD))
        elif sid in footprint:
            rows.append(_row(sid, "", STATUS_PURGED, dom, REASON_INSIDE_FOOTPRINT))
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
        purge_km=purge_km, class_by_id=class_by_id, footprint=footprint_spec,
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
# Label-access order (geographic_ood only): frozen, label-blind selection ranks
# --------------------------------------------------------------------------- #
def _assert_disjoint_unique(source_ids: Any, target_pool_ids: Any, target_test_ids: Any, where: str) -> None:
    """Hard-fail on a duplicate stable id WITHIN any population, or an id shared ACROSS populations.
    The label-access populations must be a clean partition of distinct units; silently collapsing a
    duplicate (e.g. via ``set()``) would misalign ranks against the frozen split, so it is refused."""
    seen: dict[str, str] = {}
    for pop, ids in (("source", source_ids), ("target_pool", target_pool_ids), ("target_test", target_test_ids)):
        local: set[str] = set()
        for raw in ids:
            sid = str(raw)
            if sid in local:
                raise SplitArtifactError(f"{where}: duplicate stable_id {sid!r} within {pop}")
            local.add(sid)
            if sid in seen:
                raise SplitArtifactError(f"{where}: stable_id {sid!r} appears in both {seen[sid]} and {pop}")
            seen[sid] = pop


def _blind_order(ids: list[str], rng: np.random.Generator) -> dict[str, int]:
    """Label-blind rank map over ``ids`` (assumed already unique -- callers pre-check): a seeded
    permutation of the ids taken in canonical numeric-aware order, returning ``{stable_id: contiguous
    rank 0..N-1}``. Uses only ids, never labels -- the "label-blind" guarantee. Consecutive calls
    advance ``rng``, so two calls over the same id set yield two DISTINCT deterministic orders from the
    one run seed. It does NOT de-duplicate (no set collapse); duplicate detection is the caller's."""
    canon = sorted((str(s) for s in ids), key=_sid_key)
    order = [canon[int(p)] for p in rng.permutation(len(canon))]
    return {sid: rank for rank, sid in enumerate(order)}


#: A benchmark needs at least this many ELIGIBLE headline targets for its fixed-budget allocation
#: aggregate to be reported. Below it the benchmark is omitted from the headline allocation aggregate
#: and its remaining targets are recorded as supplementary stress -- a region-level bootstrap over one
#: or two regions has no usable uncertainty.
MIN_HEADLINE_TARGETS = 3


def label_access_eligibility(
    *, n_source: int, n_target_pool: int, counts: tuple[int, ...] = LABEL_ACCESS_COUNTS,
) -> list[str]:
    """PREDECLARED eligibility for a headline geographic_ood target, applied BEFORE any budget exists.
    Returns the reasons this target is ineligible (empty == eligible).

    This is a PREDICATE, not an assertion: an undersized region is demoted to supplementary stress
    rather than aborting generation, because ``B_d`` is a benchmark-common minimum and one tiny region
    must not be able to drag the whole benchmark's budget down (or kill the run). Sizing is the only
    question asked here -- realized allocation validity is audited AFTER construction, against the
    frozen order, by :func:`audit_allocation`."""
    m = max(counts) if counts else 0
    problems: list[str] = []
    if n_source < 1 or n_target_pool < 1:
        problems.append(f"empty pool (source={n_source}, target_label_pool={n_target_pool})")
        return problems
    if n_target_pool < m:
        problems.append(f"target_label_pool {n_target_pool} < max additive count {m}")
    if n_source < m:
        problems.append(f"source_train {n_source} < max additive count {m}")
    return problems


def benchmark_budget(cells: list[dict[str, Any]], max_label_budget: int | None = None) -> int:
    """``B_d = min(B_max,d, min-cell N_source, min-cell N_target)`` over the ELIGIBLE (target, seed)
    cells ONLY -- ineligible cells are excluded before this is called, so an undersized region cannot
    lower the budget. One budget per benchmark, shared by every target and seed, which is what makes
    the allocation curve comparable across regions. ``max_label_budget`` (``B_max,d``) is an optional
    frozen ceiling; unbounded when None."""
    if not cells:
        raise SplitArtifactError("benchmark_budget: no eligible cells to derive B_d from")
    b = min(
        min(int(c["n_source"]) for c in cells),
        min(int(c["n_target_pool"]) for c in cells),
    )
    if max_label_budget is not None:
        b = min(b, int(max_label_budget))
    if b < 1:
        raise SplitArtifactError(f"benchmark_budget: B_d resolved to {b} -- no allocation is constructible")
    return int(b)


def audit_allocation(
    *, source_classes: list[frozenset[str]], target_classes: list[frozenset[str]], budget: int,
    percents: tuple[int, ...] = ALLOCATION_PERCENTS, where: str = "allocation",
) -> list[str]:
    """Post-construction scientific validity of one target's REALIZED allocation sets. Returns the
    problems found (empty == valid); a target with any problem is demoted to supplementary stress.

    ``source_classes`` / ``target_classes`` are the per-unit class sets in FROZEN RANK ORDER -- one
    singleton per sample for tabular benchmarks, the patch's whole class set for dense PASTIS -- because
    the audit inspects the exact prefixes the runtime will train on (``source[:B-k] + target[:k]``), not
    the pools in the abstract. Catches what row counts cannot: a training set that collapses to a single
    class (unfittable) at any fraction, and a fraction whose realized set cannot be drawn at all."""
    problems: list[str] = []
    n_src, n_tgt = len(source_classes), len(target_classes)
    for f in percents:
        k = allocation_target_count(f, budget)
        s = int(budget) - k
        if s < 0 or k < 0:
            problems.append(f"{where}: f={f}% yields source={s}, target={k}")
            continue
        if s > n_src:
            problems.append(f"{where}: f={f}% needs {s} source units, pool holds {n_src}")
            continue
        if k > n_tgt:
            problems.append(f"{where}: f={f}% needs {k} target units, pool holds {n_tgt}")
            continue
        classes: set[str] = set()
        for cs in source_classes[:s]:
            classes |= set(cs)
        for cs in target_classes[:k]:
            classes |= set(cs)
        if len(classes) < 2:
            problems.append(f"{where}: f={f}% training set holds {len(classes)} class(es) -- not fittable")
    return problems


def ranked_ids(rows: list[dict[str, str]]) -> tuple[list[str], list[str]]:
    """The (source, target-pool) stable ids of a built label-access table in frozen rank order --
    exactly the prefixes the runtime slices. Used by the generator to audit realized allocation sets
    without re-deriving the draw."""
    n_src = sum(1 for r in rows if r["population"] == POP_SOURCE)
    n_pool = sum(1 for r in rows if r["population"] == POP_TARGET_POOL)
    src: list[str] = [""] * n_src
    tgt: list[str] = [""] * n_pool
    for r in rows:
        if r["population"] == POP_SOURCE:
            src[int(r["source_rank"])] = r["stable_id"]
        elif r["population"] == POP_TARGET_POOL:
            tgt[int(r["target_rank"])] = r["stable_id"]
    return src, tgt


def build_label_access_rows(
    *, seed: int, source_ids: list[str], target_pool_ids: list[str], target_test_ids: list[str],
    where: str = "label_access",
) -> list[dict[str, str]]:
    """Label-blind, deterministic, contiguous per-population label-access ranking for one
    geographic_ood target. The source pool carries ONE nested order -- ``source_rank`` -- so every
    fixed-budget allocation point takes its source share as a prefix of the same order and the five
    points are strictly nested. ``target_rank`` orders the target label pool (allocation + additive
    selection). target_test units are listed (population-complete) but never ranked. Both orders come
    from the run ``seed`` directly (drawn in sequence from one Generator) -- no derived or per-route
    seeds, no checksum/version. Duplicate/overlapping ids are a hard error (never silently
    de-duplicated)."""
    _assert_disjoint_unique(source_ids, target_pool_ids, target_test_ids, where)
    rng = np.random.default_rng(int(seed))
    src = _blind_order(list(source_ids), rng)
    tgt = _blind_order(list(target_pool_ids), rng)
    rows: list[dict[str, str]] = []
    rows += [
        {"stable_id": s, "population": POP_SOURCE, "source_rank": str(r), "target_rank": ""}
        for s, r in src.items()
    ]
    rows += [
        {"stable_id": s, "population": POP_TARGET_POOL, "source_rank": "", "target_rank": str(r)}
        for s, r in tgt.items()
    ]
    rows += [
        {"stable_id": str(s), "population": POP_TARGET_TEST, "source_rank": "", "target_rank": ""}
        for s in target_test_ids
    ]
    return rows


def _label_access_sort_key(row: dict[str, str]) -> tuple[int, int, tuple[int, Any]]:
    pop = row["population"]
    if pop == POP_SOURCE:
        rank_str = row["source_rank"]
    elif pop == POP_TARGET_POOL:
        rank_str = row["target_rank"]
    else:
        rank_str = ""
    rank = int(rank_str) if rank_str not in ("", None) else -1
    return (_LA_POP_RANK.get(pop, 99), rank, _sid_key(row["stable_id"]))


def _label_access_bytes(rows: list[dict[str, str]]) -> bytes:
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(LABEL_ACCESS_HEADER)
    for r in sorted(rows, key=_label_access_sort_key):
        w.writerow([r[c] for c in LABEL_ACCESS_HEADER])
    return buf.getvalue().encode()


def write_label_access(
    root: str | os.PathLike, benchmark: str, seed: int, holdout: str, rows: list[dict[str, str]],
) -> tuple[Path, str]:
    """Write one geographic_ood target's ``label_access.csv`` (deterministic order) and return
    ``(path, sha256)``.

    The checksum is recorded beside the assignments checksum in the central log and re-verified at
    load. Structural validation alone cannot detect a DIFFERENT valid permutation over the same id
    sets -- and such a permutation silently changes every matched-label and fixed-total experiment.
    """
    label = str(holdout)
    if "/" in label or label in ("", ".", ".."):
        raise SplitArtifactError(f"unsafe holdout label for a leaf directory: {holdout!r}")
    data = _label_access_bytes(rows)
    path = label_access_path(root, benchmark, seed, holdout)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path, sha256_bytes(data)


def read_label_access_csv(path: str | os.PathLike) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with open(path, newline="") as fh:
        reader = csv.reader(fh)
        header = next(reader, None)
        if header != LABEL_ACCESS_HEADER:
            raise SplitArtifactError(f"malformed label_access header in {path}: {header}")
        for row in reader:
            if len(row) != len(LABEL_ACCESS_HEADER):
                raise SplitArtifactError(f"malformed label_access row in {path}: {row}")
            rows.append(dict(zip(LABEL_ACCESS_HEADER, row, strict=True)))
    return rows


def _assert_contiguous_rank(pop_rows: list[dict[str, str]], rank_col: str, blank_cols, where: str) -> None:
    if isinstance(blank_cols, str):
        blank_cols = (blank_cols,)
    ranks: list[int] = []
    for r in pop_rows:
        for bc in blank_cols:
            if r[bc] != "":
                raise SplitArtifactError(
                    f"{where}: id {r['stable_id']!r} carries a {bc} but is ranked by {rank_col}"
                )
        if r[rank_col] == "":
            raise SplitArtifactError(f"{where}: id {r['stable_id']!r} missing {rank_col}")
        try:
            ranks.append(int(r[rank_col]))
        except ValueError as exc:
            raise SplitArtifactError(f"{where}: non-integer {rank_col} {r[rank_col]!r}") from exc
    if sorted(ranks) != list(range(len(ranks))):
        raise SplitArtifactError(f"{where}: {rank_col} is not contiguous 0..{len(ranks) - 1}")


def validate_label_access_rows(
    rows: list[dict[str, str]], *, source_ids: Any, target_pool_ids: Any, target_test_ids: Any,
    where: str = "label_access",
) -> str:
    """Structural integrity of a ``label_access.csv`` against the frozen split: unique ids, known
    populations, population-correct + complete id sets, and contiguous 0..N-1 ranks in the correct
    column (blank in the other). Raises :class:`SplitArtifactError` on any violation; returns
    ``"passed"``. (Label-blindness is a generation property and cannot be re-derived here.)"""
    _assert_disjoint_unique(source_ids, target_pool_ids, target_test_ids, where)  # never silently dedup
    by_pop: dict[str, list[dict[str, str]]] = {POP_SOURCE: [], POP_TARGET_POOL: [], POP_TARGET_TEST: []}
    seen: set[str] = set()
    for r in rows:
        sid, pop = r["stable_id"], r["population"]
        if pop not in _LA_POP_RANK:
            raise SplitArtifactError(f"{where}: unknown population {pop!r} for id {sid!r}")
        if sid in seen:
            raise SplitArtifactError(f"{where}: duplicate stable_id {sid!r}")
        seen.add(sid)
        by_pop[pop].append(r)
    for pop, want in ((POP_SOURCE, source_ids), (POP_TARGET_POOL, target_pool_ids), (POP_TARGET_TEST, target_test_ids)):
        got = {r["stable_id"] for r in by_pop[pop]}
        want_set = {str(s) for s in want}
        if got != want_set:
            missing = sorted(want_set - got)
            extra = sorted(got - want_set)
            raise SplitArtifactError(
                f"{where}: population {pop!r} does not match the frozen split -- {len(missing)} missing "
                f"(e.g. {missing[:5]}), {len(extra)} unexpected (e.g. {extra[:5]})"
            )
    # source pool: the ONE nested order contiguous 0..S-1; no source unit carries a target_rank.
    _assert_contiguous_rank(by_pop[POP_SOURCE], "source_rank", ("target_rank",), where)
    # target pool: target_rank contiguous 0..P-1; not the source order.
    _assert_contiguous_rank(by_pop[POP_TARGET_POOL], "target_rank", SOURCE_RANK_COLS, where)
    for r in by_pop[POP_TARGET_TEST]:
        if any(r[c] != "" for c in (*SOURCE_RANK_COLS, "target_rank")):
            raise SplitArtifactError(f"{where}: target_test id {r['stable_id']!r} must carry no rank")
    return "passed"


class LoadedLabelAccess(NamedTuple):
    """One geographic_ood target's frozen label-access order, resolved to CURRENT row indices, in
    ascending rank order (rank 0 first). The source pool exposes ONE nested order --
    ``source_ranked_idx`` -- and ``target_ranked_idx`` orders the target pool (allocation +
    additive). The routes slice prefixes of these."""

    holdout: str
    source_ranked_idx: np.ndarray
    target_ranked_idx: np.ndarray
    #: B_d, the benchmark-common allocation budget frozen at generation.
    benchmark_budget: int = 0


def _verify_label_access_checksum(data: bytes, expected_sha256: str | None, where: str) -> None:
    """Integrity gate, run BEFORE structural validation.

    Structural validation accepts ANY contiguous permutation over the correct id sets, so it cannot
    distinguish a regenerated or hand-edited draw from the frozen one -- and a different valid draw
    silently changes every allocation and additive experiment. Only the checksum catches that.
    """
    if expected_sha256 is None:
        raise SplitArtifactError(
            f"{where}: the split log records no label_access sha256 -- the frozen label draw is "
            f"unverifiable; regenerate the splits with the current generator"
        )
    got = sha256_bytes(data)
    if got != str(expected_sha256):
        raise SplitArtifactError(
            f"checksum mismatch for {where}: log records {str(expected_sha256)[:12]}..., file hashes "
            f"to {got[:12]}... -- the frozen label draw changed; refuse to consume"
        )


def load_label_access(
    root: str | os.PathLike, benchmark: str, seed: int, split: SourceTargetSplit, id_map: dict[str, Any],
    expected_sha256: str | None = None, benchmark_budget: int | None = None,
) -> LoadedLabelAccess:
    """Load + structurally validate one geographic_ood target's frozen ``label_access.csv`` and
    resolve its ranked stable ids to CURRENT row indices. Missing or malformed => hard error. The
    validation checks the file against ``split`` (the already-loaded frozen partitions), so a stale
    order that no longer matches the current population is refused rather than mis-consumed."""
    holdout = str(split.label)
    where = f"{benchmark}/{LABEL_ACCESS_REGIME}/{int(seed)}/{holdout}/{LABEL_ACCESS_FILENAME}"
    path = label_access_path(root, benchmark, seed, holdout)
    if not path.is_file():
        raise SplitArtifactError(f"missing label_access.csv at {path} -- run tools/generate_splits.py first")
    _verify_label_access_checksum(path.read_bytes(), expected_sha256, where)
    rows = read_label_access_csv(path)
    inv = {int(v): k for k, v in id_map.items()}
    src_ids = [inv[int(i)] for i in np.asarray(split.source_train, dtype=np.int64).tolist()]
    pool_ids = [inv[int(i)] for i in np.asarray(split.target_label_pool, dtype=np.int64).tolist()]
    test_ids = [inv[int(i)] for i in np.asarray(split.target_test, dtype=np.int64).tolist()]
    validate_label_access_rows(
        rows, source_ids=src_ids, target_pool_ids=pool_ids, target_test_ids=test_ids, where=where,
    )
    n_src = sum(1 for r in rows if r["population"] == POP_SOURCE)
    n_pool = sum(1 for r in rows if r["population"] == POP_TARGET_POOL)
    src: list[int] = [0] * n_src
    tgt: list[int] = [0] * n_pool
    for r in rows:
        if r["population"] == POP_SOURCE:
            src[int(r["source_rank"])] = id_map[r["stable_id"]]
        elif r["population"] == POP_TARGET_POOL:
            tgt[int(r["target_rank"])] = id_map[r["stable_id"]]
    if benchmark_budget is None:
        raise SplitArtifactError(
            f"{where}: the split log records no label_access benchmark_budget (B_d) -- the allocation "
            f"curve is undefined; regenerate the label-access artifacts with the current generator"
        )
    return LoadedLabelAccess(
        holdout=holdout,
        source_ranked_idx=np.asarray(src, dtype=np.int64),
        target_ranked_idx=np.asarray(tgt, dtype=np.int64),
        benchmark_budget=int(benchmark_budget),
    )


class LoadedDenseLabelAccess(NamedTuple):
    """One geographic_ood dense (PASTIS) target's frozen label-access order, resolved to STABLE PATCH IDs
    in ascending rank order (rank 0 first). Unlike the tabular loader there is no row-index remap: the
    patch id IS the stable unit. The source pool exposes ONE nested order --
    ``source_ranked_patches`` -- and ``target_ranked_patches`` orders the target label pool. Every
    selection/removal is a WHOLE patch."""

    holdout: str
    source_ranked_patches: np.ndarray
    target_ranked_patches: np.ndarray
    #: B_d in PATCH units, frozen at generation.
    benchmark_budget: int = 0


def load_dense_label_access(
    root: str | os.PathLike, benchmark: str, seed: int, dense_split: DenseSourceTargetSplit,
    expected_sha256: str | None = None, benchmark_budget: int | None = None,
) -> LoadedDenseLabelAccess:
    """Load + structurally validate one geographic_ood PASTIS target's frozen ``label_access.csv`` and
    resolve its ranked stable patch ids to PATCH IDs in rank order. Missing/malformed => hard error. The
    validation checks the file against ``dense_split`` (the already-loaded frozen patch partitions), so a
    stale order that no longer matches the current patch population is refused rather than mis-consumed."""
    holdout = str(dense_split.label)
    where = f"{benchmark}/{LABEL_ACCESS_REGIME}/{int(seed)}/{holdout}/{LABEL_ACCESS_FILENAME}"
    path = label_access_path(root, benchmark, seed, holdout)
    if not path.is_file():
        raise SplitArtifactError(f"missing label_access.csv at {path} -- run tools/generate_splits.py first")
    _verify_label_access_checksum(path.read_bytes(), expected_sha256, where)
    rows = read_label_access_csv(path)
    src_ids = [str(int(p)) for p in sorted(dense_split.source_train_patches)]
    pool_ids = [str(int(p)) for p in sorted(dense_split.target_label_pool_patches)]
    test_ids = [str(int(p)) for p in sorted(dense_split.target_test_patches)]
    validate_label_access_rows(
        rows, source_ids=src_ids, target_pool_ids=pool_ids, target_test_ids=test_ids, where=where,
    )
    n_src = sum(1 for r in rows if r["population"] == POP_SOURCE)
    n_pool = sum(1 for r in rows if r["population"] == POP_TARGET_POOL)
    src: list[int] = [0] * n_src
    tgt: list[int] = [0] * n_pool
    for r in rows:
        if r["population"] == POP_SOURCE:
            src[int(r["source_rank"])] = int(r["stable_id"])
        elif r["population"] == POP_TARGET_POOL:
            tgt[int(r["target_rank"])] = int(r["stable_id"])
    if benchmark_budget is None:
        raise SplitArtifactError(
            f"{where}: the split log records no label_access benchmark_budget (B_d) -- the allocation "
            f"curve is undefined; regenerate the label-access artifacts with the current generator"
        )
    return LoadedDenseLabelAccess(
        holdout=holdout,
        source_ranked_patches=np.asarray(src, dtype=np.int64),
        target_ranked_patches=np.asarray(tgt, dtype=np.int64),
        benchmark_budget=int(benchmark_budget),
    )


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
    #: sha256 of this leaf's label_access.csv as recorded in the central log (None where the leaf
    #: carries no label-access order). Threaded to load_label_access so the frozen draw is verified.
    label_access_sha256: str | None = None
    #: the benchmark-common fixed-budget allocation budget B_d frozen at generation (None where absent).
    label_access_budget: int | None = None


class LoadedDenseSplit(NamedTuple):
    """One consumed dense leaf: a :class:`DenseSourceTargetSplit` (patch sets) + its seed/regime."""

    seed: int
    regime: str
    split: DenseSourceTargetSplit
    #: sha256 of this leaf's label_access.csv as recorded in the central log (None where absent).
    label_access_sha256: str | None = None
    #: the benchmark-common fixed-budget allocation budget B_d frozen at generation (None where absent).
    label_access_budget: int | None = None


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
                loaded.append(LoadedTabularSplit(
                    seed=int(seed), regime=str(regime), domains=domains, split=split,
                    label_access_sha256=entry.get("label_access_sha256"),
                    label_access_budget=(entry.get("label_access") or {}).get("benchmark_budget"),
                ))
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
                leaves.append(LoadedDenseSplit(
                    seed=int(seed), regime=str(regime), split=dsplit,
                    label_access_sha256=entry.get("label_access_sha256"),
                    label_access_budget=(entry.get("label_access") or {}).get("benchmark_budget"),
                ))
        by_seed[int(seed)] = leaves
    return by_seed
