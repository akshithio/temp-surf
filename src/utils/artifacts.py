"""Run provenance and completion artifacts.

Two gaps this closes.

**Provenance.** Result artifacts do not record the environment that produced them. The canonical
``output-erm-full-20260711`` numbers were made under an exactly standardized numerical core
(numpy 1.26.4 / scipy 1.17.1 / scikit-learn 1.9.0 / torch 2.7.1), but that fact survives only in
the launch record and in the still-installed environments -- both on scratch filesystems with a
60-day purge. scikit-learn drives probe numerics and is not part of the run signature, so a drift
is invisible. ``environment.json`` puts the versions next to the numbers they produced, and a
resume that would append rows from a DIFFERENT numerical core is refused rather than silently
mixing two environments into one results table.

**Completion.** ``run_manifest.json`` is published BEFORE any work starts, so it marks *started*,
not *finished*. The derived artifacts (``summary.csv``, ``deltas.csv``, ``probe_results.csv``) are
written only at the very end, so a pair killed mid-probe-loop leaves STALE derived files beside a
now-larger ``probe_results.jsonl``, with nothing signalling that they disagree -- and those files
are read directly by the figure scripts. ``run_complete.json`` is written last, atomically, and
only when the pair is provably complete: every required artifact present, every planned cell
realized exactly once, and no declared regime dropped.

This module is deliberately NOT part of the run manifest's identity: recording what an environment
was, or that a run finished, must never change the identity of the run it describes. Fields can be
added here without invalidating a single existing results directory.
"""

from __future__ import annotations

import collections
import hashlib
import json
import platform
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from utils import ioutils as IOU

ENVIRONMENT_FILE = "environment.json"
RUN_COMPLETE_FILE = "run_complete.json"
RUN_MANIFEST_FILE = "run_manifest.json"
SCHEMA_VERSION = 1

#: Versions pinned for COMPARABILITY with the canonical run -- a drift in any of these silently
#: changes probe numbers, and none of them are in the run signature. A resume across a change here
#: is refused.
NUMERICAL_CORE = ("numpy", "scipy", "scikit-learn", "torch")

#: Encoder-side packages whose version can move an embedding. Recorded, but NOT gated on: they
#: affect cached embeddings (which carry their own checkpoint fingerprint), not probe arithmetic.
ENCODER_PACKAGES = (
    "presto", "olmoearth-pretrain", "timm", "mmengine", "thop", "einops",
    "torchvision", "breizhcrops", "h5py",
)

#: Every artifact a finished pair must have. Hashed into the completion marker so a later reader
#: can tell that the derived CSVs belong to the probe_results.jsonl sitting next to them -- and so
#: that a tampered or half-rewritten directory is detectable rather than merely unlikely.
REQUIRED_ARTIFACTS = (
    "probe_results.jsonl",
    "probe_results.csv",
    "summary.csv",
    "deltas.csv",
    ENVIRONMENT_FILE,
    # PHASE B: canonical splits live under data/splits/; each pair records the regime-level leaves it
    # consumed in split_ref.json (replacing the retired per-model split_manifest.json).
    "split_ref.json",
)

#: The identity of one evaluation cell. Identical for the tabular and dense paths -- both write
#: these eight fields on every row -- so completeness is checkable the same way for both.
CELL_KEY_FIELDS = (
    "seed", "split_regime", "holdout", "method",
    "probe_family", "budget_type", "label_budget", "evaluation_split",
)


class IncompleteRunError(RuntimeError):
    """A pair cannot be certified complete. Raised instead of silently skipping the marker, so
    the pair is recorded as FAILED and the shard exits non-zero."""


class EnvironmentMismatchError(RuntimeError):
    """A resume would append rows produced by a different numerical core."""


class EnvironmentProvenanceError(RuntimeError):
    """Existing rows have no readable environment record, so resuming would have to invent one."""


def cell_key(row: dict[str, Any]) -> tuple:
    return tuple(row.get(k) for k in CELL_KEY_FIELDS)


def sha256_file(path: Path | str) -> str | None:
    """Streaming SHA-256; None if absent. Chunked -- predictions.jsonl reaches ~20 GB."""
    path = Path(path)
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_jsonl_rows(path: Path) -> list[dict[str, Any]]:
    """Parse a JSONL file, raising on any malformed row.

    Validation must not count non-blank lines: a torn or corrupt row is exactly the condition
    worth catching, and it is indistinguishable from a healthy one by line count alone.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    rows = []
    with path.open() as fh:
        for lineno, line in enumerate(fh, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{lineno}: malformed JSONL row: {exc}") from exc
    return rows


def _version(package: str) -> str | None:
    try:
        import importlib.metadata as md

        return md.version(package)
    except Exception:
        return None


def _git(repo: Path, *args: str) -> str | None:
    """Raw stdout, NOT stripped.

    Stripping here silently corrupted `status --porcelain`, whose format is two status columns
    then a space: the leading space of the FIRST line is significant (' M path'), so a strip made
    line[3:] eat a character off that one path and no other. Whitespace is likewise significant to
    `diff`, which is hashed. Callers that want a single token use _git_line.
    """
    try:
        out = subprocess.run(
            ["git", *args], cwd=str(repo), capture_output=True, text=True, timeout=15, check=False
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout if out.returncode == 0 else None


def _git_line(repo: Path, *args: str) -> str | None:
    """A single-token git result (a hash, a branch name)."""
    out = _git(repo, *args)
    return out.strip() if out is not None else None


#: Untracked files that count as CODE for identity purposes: anything that can change what a run
#: computes or how it is launched. Sources, config, dependency locks, and launchers -- a run
#: submitted by a different sbatch script is a different run even if every .py is identical.
#:
#: Datasets and generated artifacts are excluded two ways: `--exclude-standard` drops everything
#: gitignored (viz/data's 79 MB snapshot, data/cache, data/output), and the allowlist below admits
#: no data extension (.jsonl/.csv/.npy/.png/.h5/.pkl are all absent), so a results file dropped in
#: the tree cannot perturb the identity of the code.
_SOURCE_SUFFIXES = frozenset({
    ".py", ".pyi",                       # source
    ".sh", ".bash", ".zsh",              # shell
    ".sbatch", ".slurm",                 # cluster launchers
    ".toml", ".yml", ".yaml", ".json", ".ini", ".cfg", ".conf",  # config
    ".lock",                             # dependency locks
})
#: Extensionless files that are code by name.
_SOURCE_NAMES = frozenset({"Dockerfile", "Makefile", "makefile", "Containerfile", "Justfile"})


def _is_source_file(rel: str) -> bool:
    path = Path(rel)
    if path.suffix in _SOURCE_SUFFIXES or path.name in _SOURCE_NAMES:
        return True
    # requirements text (requirements.txt, requirements-dev.txt, ...) but not arbitrary .txt,
    # which is far more often a note or a log than a dependency declaration.
    return path.suffix == ".txt" and path.name.lower().startswith("requirements")


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _untracked_source_digest(repo: Path) -> tuple[str | None, list[str]]:
    """Deterministic digest over untracked SOURCE files (content, not just names)."""
    listing = _git(repo, "ls-files", "--others", "--exclude-standard")
    if listing is None:
        return None, []
    paths = sorted(p for p in listing.splitlines() if p.strip() and _is_source_file(p))
    digest = hashlib.sha256()
    kept: list[str] = []
    for rel in paths:
        full = repo / rel
        try:
            body = full.read_bytes()
        except OSError:
            continue
        digest.update(rel.encode())
        digest.update(b"\0")
        digest.update(_sha256_bytes(body).encode())
        digest.update(b"\n")
        kept.append(rel)
    return digest.hexdigest(), kept


def git_identity(repo: Path) -> dict[str, Any]:
    """Content-addressed identity of the code that produced a run.

    Three of the machines that produce results are rsync copies with no ``.git`` at all, so
    ``commit`` is legitimately null there -- record that honestly rather than implying a
    provenance that does not exist.

    A commit alone does not identify this code: the working trees routinely carry tens of
    modified files plus whole untracked modules, and a file LIST cannot tell two different edits
    to the same filenames apart. So the identity hashes CONTENT -- the tracked diff against HEAD,
    and every untracked source file -- and combines them into ``tree_identity``. Two runs whose
    ``tree_identity`` matches were produced by the same code; two that differ were not, however
    similar their dirty-file lists look. The human-readable list is retained alongside, because a
    hash tells you THAT the code differed and the list starts telling you where.
    """
    commit = _git_line(repo, "rev-parse", "HEAD")
    if commit is None:
        return {
            "commit": None, "branch": None, "dirty": None, "dirty_files": None,
            "n_dirty_files": None, "tracked_diff_sha256": None,
            "untracked_source_sha256": None, "tree_identity": None,
            "note": "no git repository at the code root -- provenance is not recoverable",
        }
    status = _git(repo, "status", "--porcelain")
    dirty_files = sorted(line[3:] for line in (status or "").splitlines() if line.strip())
    tracked_diff = _git(repo, "diff", "HEAD")
    tracked_sha = _sha256_bytes((tracked_diff or "").encode())
    untracked_sha, untracked_files = _untracked_source_digest(repo)
    tree_identity = _sha256_bytes(
        f"{commit}|{tracked_sha}|{untracked_sha}".encode()
    )
    return {
        "commit": commit,
        "branch": _git_line(repo, "rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": bool(dirty_files),
        "dirty_files": dirty_files[:200],
        "n_dirty_files": len(dirty_files),
        "tracked_diff_sha256": tracked_sha,
        "untracked_source_sha256": untracked_sha,
        "n_untracked_source_files": len(untracked_files),
        "tree_identity": tree_identity,
    }


def capture_environment(repo: Path | None = None) -> dict[str, Any]:
    """Everything needed to decide whether two result trees are numerically comparable."""
    from utils import cacheutils

    repo = Path(repo) if repo is not None else cacheutils.REPO
    core = {name: _version(name) for name in NUMERICAL_CORE}
    try:
        import torch

        core["torch"] = torch.__version__  # carries the +cu128 / cpu build suffix
        cuda = {"available": bool(torch.cuda.is_available()), "version": torch.version.cuda}
    except Exception:
        cuda = {"available": None, "version": None}
    return {
        "schema": SCHEMA_VERSION,
        "captured_at": datetime.now(UTC).isoformat(),
        "python": sys.version.split()[0],
        "python_full": sys.version,
        "platform": platform.platform(),
        "machine": platform.node(),
        "numerical_core": core,
        "encoder_packages": {name: _version(name) for name in ENCODER_PACKAGES},
        "cuda": cuda,
        "git": git_identity(repo),
    }


def environment_state(results_dir: Path) -> tuple[str, dict[str, Any] | None]:
    """('absent' | 'malformed' | 'present', record).

    Absent and malformed are NOT the same fault. Absent means the rows predate provenance
    recording; malformed means the record was damaged. Both are unsafe to silently overwrite when
    rows exist, but they call for different remedies, so the caller is told which it is.
    """
    path = Path(results_dir) / ENVIRONMENT_FILE
    if not path.exists():
        return "absent", None
    try:
        loaded = json.loads(path.read_text())
    except json.JSONDecodeError:
        return "malformed", None
    if not isinstance(loaded, dict) or "numerical_core" not in loaded:
        return "malformed", None
    return "present", loaded


def has_result_rows(results_dir: Path) -> bool:
    """Does this directory already hold probe rows that an environment record would describe?"""
    path = Path(results_dir) / "probe_results.jsonl"
    if not path.exists():
        return False
    with path.open() as fh:
        return any(line.strip() for line in fh)


def environment_mismatches(a: dict[str, Any], b: dict[str, Any]) -> list[str]:
    """Numerical-core + Python differences between two environment records.

    These are the reasons two sets of probe rows may not be pooled. Encoder packages, platform and
    git are recorded but not compared: they move embeddings (already covered by the checkpoint
    fingerprint) rather than the probe arithmetic, and gating on them would refuse legitimate
    cross-machine resumes.
    """
    out = []
    for name in NUMERICAL_CORE:
        va, vb = a.get("numerical_core", {}).get(name), b.get("numerical_core", {}).get(name)
        if va != vb:
            out.append(f"{name}: {va} != {vb}")
    if a.get("python") != b.get("python"):
        out.append(f"python: {a.get('python')} != {b.get('python')}")
    return out


def write_environment(
    results_dir: Path, repo: Path | None = None, *, overwrite_mode: bool = False
) -> dict[str, Any]:
    """Record the producing environment, or verify a resume against the recorded one.

    The governing rule: **never attach the current environment to rows it did not produce.**
    Writing environment.json beside existing rows asserts "this environment made these numbers",
    so it is only allowed when that assertion is true or the rows are gone:

      * no rows yet            -> record freely; there is nothing to mislabel.
      * rows + compatible      -> preserve the ORIGINAL record untouched (captured_at included)
                                  and return it. Overwriting would erase the only evidence of the
                                  environment that made most of the table.
      * rows + absent record   -> REFUSE. The rows predate provenance recording; stamping the
                                  current environment on them would fabricate provenance. Use
                                  backfill_environment() if it can actually be attributed.
      * rows + malformed       -> REFUSE. The record was damaged; the rows' real provenance is
                                  unknown, and unknown must not silently become "current".
      * rows + incompatible    -> REFUSE unless overwrite_mode, since the appended rows would not
                                  be comparable with the ones already there.

    OVERWRITE_MODE is sound only because the caller discards the rows before calling this.
    """
    results_dir = Path(results_dir)
    current = capture_environment(repo)
    state, existing = environment_state(results_dir)
    rows_exist = has_result_rows(results_dir)

    if overwrite_mode and rows_exist:
        # OVERWRITE_MODE does not license relabeling: it licenses DISCARDING. Replacing the
        # record while its rows are still on disk would attach this environment to numbers it did
        # not produce -- the same fabrication the no-overwrite path refuses, merely authorized by
        # a flag. The callers unlink probe_results.jsonl before reaching here; this enforces that
        # ordering rather than trusting it.
        raise EnvironmentProvenanceError(
            f"Refusing to replace the environment record in {results_dir}: OVERWRITE_MODE is set "
            f"but probe_results.jsonl still holds rows. The rows must be removed BEFORE their "
            f"provenance is replaced, or the new record would relabel them."
        )

    if not rows_exist:
        # Nothing to mislabel. A fresh record is correct; note what it supersedes, if anything.
        if state == "present" and not environment_mismatches(existing, current):
            return existing  # idempotent re-entry: keep the original captured_at
        payload = dict(current)
        if state == "present":
            problems = environment_mismatches(existing, current)
            payload["superseded_environment"] = {
                "reason": f"replaced an incompatible record ({'; '.join(problems)})",
                "previous": existing,
            }
        elif state == "malformed":
            payload["superseded_environment"] = {
                "reason": "replaced an unreadable record", "previous": None,
            }
        IOU.write_json(results_dir / ENVIRONMENT_FILE, payload)
        return payload

    # Rows exist and we are not discarding them.
    if state == "absent":
        raise EnvironmentProvenanceError(
            f"Refusing to resume {results_dir}: it holds probe rows but has no "
            f"{ENVIRONMENT_FILE}. Writing one now would assert that THIS environment produced "
            f"those rows, which is not known to be true -- scikit-learn/scipy/numpy are not in "
            f"the run signature, so nothing else would catch the fabrication. Either "
            f"backfill_environment() with the attributable producing environment, or set "
            f"OVERWRITE_MODE=True to discard the rows and re-run this pair."
        )
    if state == "malformed":
        raise EnvironmentProvenanceError(
            f"Refusing to resume {results_dir}: its {ENVIRONMENT_FILE} is unreadable, so the "
            f"provenance of the rows already there is unknown. Replacing it with the current "
            f"environment would silently convert 'unknown' into 'this one'. Restore the record, "
            f"backfill_environment() it, or set OVERWRITE_MODE=True to discard the rows."
        )

    problems = environment_mismatches(existing, current)
    if problems:
        raise EnvironmentMismatchError(
            f"Refusing to resume {results_dir}: it holds rows produced by a different "
            f"environment ({'; '.join(problems)}). scikit-learn/scipy/numpy drive probe numerics "
            f"and are NOT part of the run signature, so appending here would silently mix two "
            f"numerical environments into one results table. Rebuild this machine's env to match "
            f"(`uv sync --locked --all-extras`), or set OVERWRITE_MODE=True to discard the "
            f"existing rows and re-run this pair under the current environment."
        )
    return existing  # compatible: preserve the original record and its captured_at


def environment_schema_problems(env: Any) -> list[str]:
    """Is this a COMPLETE environment record, or merely a well-formed JSON object?

    A record missing python or any NUMERICAL_CORE version answers nothing about comparability,
    which is the only reason the record exists. Hashing such a record certifies its bytes while
    saying nothing about whether the rows beside it are poolable -- so completeness validation
    checks the schema too, not just the hash.
    """
    if not isinstance(env, dict):
        return ["environment record is not a JSON object"]
    problems: list[str] = []
    if not str(env.get("python") or "").strip():
        problems.append("missing python version")
    core = env.get("numerical_core")
    if not isinstance(core, dict):
        return [*problems, "missing numerical_core"]
    for name in NUMERICAL_CORE:
        if not str(core.get(name) or "").strip():
            problems.append(f"numerical_core.{name} is missing or empty")
    return problems


def backfill_environment(
    results_dir: Path, *, verified_by: str, note: str, environment: dict[str, Any]
) -> dict[str, Any]:
    """Attribute a producing environment to rows that predate environment.json.

    The ONLY sanctioned way to put a record beside rows this process did not produce. The
    environment must be SUPPLIED (from the launch record, or read off the machine that ran it) --
    it is deliberately not captured here, because capturing it would be exactly the fabrication
    write_environment refuses. Marked backfilled + attributable so a reader can weigh it as a
    human's assertion rather than an observation.

    Refuses unless it is genuinely filling a HOLE:
      * rows must exist -- there is nothing to attribute otherwise, and a fresh run should record
        its own environment by observation rather than by assertion;
      * the record must be absent or malformed -- a readable record is the run's own evidence, and
        a human assertion must never quietly replace it;
      * the supplied record must be complete, or it asserts nothing useful.
    """
    results_dir = Path(results_dir)
    if not verified_by or not note:
        raise ValueError("backfill_environment requires verified_by and note: it records a "
                         "human's assertion about provenance and must be attributable")
    if not has_result_rows(results_dir):
        raise ValueError(
            f"refusing to backfill {results_dir}: it holds no probe rows, so there is no "
            f"provenance to attribute. A run that is about to produce rows records its own "
            f"environment by observation."
        )
    state, existing = environment_state(results_dir)
    if state == "present":
        raise ValueError(
            f"refusing to backfill {results_dir}: it already has a readable {ENVIRONMENT_FILE} "
            f"(python={existing.get('python')}, "
            f"scikit-learn={existing.get('numerical_core', {}).get('scikit-learn')}). That record "
            f"is the run's own evidence; a human assertion must not overwrite it. Backfill fills "
            f"a hole -- an absent or malformed record -- and nothing else."
        )
    problems = environment_schema_problems(environment)
    if problems:
        raise ValueError(
            f"backfill_environment requires a COMPLETE attributed record; it is not captured from "
            f"this process. Problems: {problems}"
        )
    payload = {
        **environment,
        "backfilled": True,
        "verified_by": verified_by,
        "note": note,
        "backfilled_at": datetime.now(UTC).isoformat(),
        "backfilled_over": state,  # 'absent' or 'malformed' -- what hole this filled
    }
    IOU.write_json(results_dir / ENVIRONMENT_FILE, payload)
    return payload


# --- completeness -----------------------------------------------------------


def completeness(expected_keys: set[tuple], rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Compare the planned cell-key set against what actually landed on disk.

    Three distinct faults, all of which otherwise leave a table that reads as finished:
      missing    -- a planned cell produced no row (crashed, skipped, or dropped)
      unexpected -- a row nobody planned (e.g. left over from a superseded config)
      duplicate  -- a cell written twice (a resume that re-ran without pruning)
    """
    actual = [cell_key(r) for r in rows]
    counts = collections.Counter(actual)
    missing = set(expected_keys) - set(actual)
    unexpected = set(actual) - set(expected_keys)
    duplicate = {k for k, c in counts.items() if c > 1}
    return {
        "expected": len(set(expected_keys)),
        "actual_rows": len(actual),
        "actual_cells": len(counts),
        "missing": sorted((list(k) for k in missing), key=str),
        "unexpected": sorted((list(k) for k in unexpected), key=str),
        "duplicate": sorted((list(k) for k in duplicate), key=str),
        "ok": not (missing or unexpected or duplicate),
    }


def _summarize(label: str, items: list, limit: int = 5) -> str:
    shown = ", ".join(str(i) for i in items[:limit])
    more = f" (+{len(items) - limit} more)" if len(items) > limit else ""
    return f"{len(items)} {label}: {shown}{more}"


# --- completion marker ------------------------------------------------------


def invalidate_run_complete(results_dir: Path) -> bool:
    """Drop the completion marker at the top of a resume.

    A resume is about to make the directory incomplete again, so the marker must not survive it:
    a stale marker is worse than none, because it asserts a finished state that is being undone.
    """
    path = Path(results_dir) / RUN_COMPLETE_FILE
    existed = path.exists()
    path.unlink(missing_ok=True)
    return existed


def write_run_complete(
    results_dir: Path,
    *,
    run_manifest_sha256: str,
    expected_keys: set[tuple],
    rows: list[dict[str, Any]],
    regime_problems: Any = (),
    cell_failures: Any = (),
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Publish the completion marker. MUST be the pair's last write.

    Raises IncompleteRunError rather than quietly declining to publish: an incomplete pair has to
    surface as a FAILURE (non-zero exit), because a missing marker on its own is indistinguishable
    from an old run that predates markers entirely.
    """
    results_dir = Path(results_dir)
    problems: list[str] = []

    missing_artifacts = [n for n in REQUIRED_ARTIFACTS if not (results_dir / n).exists()]
    if missing_artifacts:
        problems.append(f"required artifact(s) absent: {missing_artifacts}")

    if regime_problems:
        problems.append(
            f"{len(regime_problems)} declared regime(s) did not run -- the table is incomplete "
            f"by construction: {[f'{b}/{r}' for b, r, _ in regime_problems]}"
        )

    if cell_failures:
        problems.append(_summarize(
            "probe cell(s) skipped after a degenerate fit",
            [f"{c.get('method')}/{c.get('holdout')}@{c.get('label_budget')}: {c.get('reason')}"
             for c in cell_failures],
        ))

    comp = completeness(expected_keys, rows)
    if comp["missing"]:
        problems.append(_summarize("planned cell(s) never produced a row", comp["missing"]))
    if comp["unexpected"]:
        problems.append(_summarize("row(s) for cells that were never planned", comp["unexpected"]))
    if comp["duplicate"]:
        problems.append(_summarize("cell(s) written more than once", comp["duplicate"]))

    if problems:
        raise IncompleteRunError(
            f"refusing to mark {results_dir} complete:\n  - " + "\n  - ".join(problems)
        )

    payload: dict[str, Any] = {
        "schema": SCHEMA_VERSION,
        "run_manifest_sha256": run_manifest_sha256,
        "expected_cells": comp["expected"],
        "actual_rows": comp["actual_rows"],
        "completed_at": datetime.now(UTC).isoformat(),
        "backfilled": False,
        "artifacts": {},
    }
    for name in REQUIRED_ARTIFACTS:
        path = results_dir / name
        payload["artifacts"][name] = {"sha256": sha256_file(path), "bytes": path.stat().st_size}
    if extra:
        payload.update(extra)
    IOU.write_json(results_dir / RUN_COMPLETE_FILE, payload)
    return payload


def read_run_complete(results_dir: Path) -> dict[str, Any] | None:
    path = Path(results_dir) / RUN_COMPLETE_FILE
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return None


def validate_run_complete(
    results_dir: Path, *, expected_signature: str | None = None, check_hashes: bool = True
) -> tuple[bool, list[str]]:
    """Is this directory a trustworthy finished run?

    Returns (ok, problems). Callers that read ``summary.csv`` / ``deltas.csv`` should refuse a
    directory that fails, because the failure mode is a stale derived table beside a newer
    ``probe_results.jsonl`` -- silently wrong rather than absent.
    """
    results_dir = Path(results_dir)
    marker = read_run_complete(results_dir)
    if marker is None:
        return False, [f"no readable {RUN_COMPLETE_FILE} -- the run is started, not known finished"]

    problems: list[str] = []
    marker_sig = marker.get("run_manifest_sha256", marker.get("signature"))  # legacy markers used "signature"
    if expected_signature is not None and marker_sig != expected_signature:
        problems.append(f"run_manifest_sha256 mismatch: marker={marker_sig} expected={expected_signature}")

    rows_path = results_dir / "probe_results.jsonl"
    rows: list[dict[str, Any]] | None
    try:
        rows = parse_jsonl_rows(rows_path)
    except FileNotFoundError:
        problems.append("probe_results.jsonl is missing")
        rows = None
    except ValueError as exc:
        problems.append(f"probe_results.jsonl is corrupt: {exc}")
        rows = None
    if rows is not None:
        if len(rows) != marker.get("actual_rows"):
            problems.append(
                f"probe_results.jsonl parses to {len(rows)} rows, marker recorded {marker.get('actual_rows')}"
            )
        dupes = [k for k, c in collections.Counter(cell_key(r) for r in rows).items() if c > 1]
        if dupes:
            problems.append(_summarize("duplicate cell key(s) in probe_results.jsonl", [list(k) for k in dupes]))

    if marker.get("expected_cells") is not None and marker.get("actual_rows") is not None:
        if int(marker["expected_cells"]) != int(marker["actual_rows"]):
            problems.append(
                f"expected {marker['expected_cells']} cells but {marker['actual_rows']} rows were written"
            )

    # The environment record's BYTES being unchanged says nothing about whether it answers the
    # question it exists for. Validate its schema too.
    env_state, env = environment_state(results_dir)
    if env_state == "absent":
        problems.append(f"{ENVIRONMENT_FILE} is missing -- the rows' provenance is unrecorded")
    elif env_state == "malformed":
        problems.append(f"{ENVIRONMENT_FILE} is unreadable -- the rows' provenance is unrecoverable")
    else:
        for problem in environment_schema_problems(env):
            problems.append(f"{ENVIRONMENT_FILE}: {problem}")

    recorded = marker.get("artifacts") or {}
    for name in REQUIRED_ARTIFACTS:
        if name not in recorded:
            problems.append(f"{name}: required but not recorded in the marker")
            continue
        if not check_hashes:
            continue
        want = recorded[name].get("sha256")
        if want is None:
            # A null hash is a hole, not a pass: it means the artifact was absent when the marker
            # was written, so the marker certifies nothing about it.
            problems.append(f"{name}: marker records a null sha256 -- the artifact was absent at completion")
            continue
        got = sha256_file(results_dir / name)
        if got is None:
            problems.append(f"{name}: recorded in the marker but missing on disk")
        elif got != want:
            problems.append(f"{name}: sha256 changed since completion (stale or edited)")
    return (not problems), problems


def backfill_run_complete(
    results_dir: Path,
    *,
    verified_by: str,
    note: str,
    expected_cells: int,
    signature: str | None = None,
) -> dict[str, Any]:
    """Write a marker for a run that predates the marker's existence.

    Every canonical results directory was produced before ``run_complete.json`` existed, so a
    validator that simply required it would reject all of them at once. This is the escape hatch --
    but a backfilled marker asserts only that a HUMAN verified the directory, never that the
    harness observed it finish, and it says so in the artifact.

    ``expected_cells`` must be established INDEPENDENTLY (from the run's config: seeds x regimes x
    holdouts x probes x budgets x scopes) and passed in. It is deliberately not defaulted to the
    observed row count: that would make expected == actual a tautology and certify a truncated
    directory as complete, which is the precise failure this whole mechanism exists to catch.
    ``verified_by`` and ``note`` are mandatory so a backfill cannot be anonymous or swept over a
    tree in a loop.

    TEMPORARY -- this backfill path (and its ``run_signature.txt`` read, the only remaining reference
    to that retired file) is pre-release cleanup support for the pre-marker canonical directories. It
    is removed once those directories are re-certified or retired; nothing in the frozen run writes
    ``run_signature.txt`` any more.
    """
    if not verified_by or not note:
        raise ValueError("backfill_run_complete requires verified_by and note: a backfilled "
                         "marker records a human's assertion and must be attributable")
    if not isinstance(expected_cells, int) or isinstance(expected_cells, bool) or expected_cells <= 0:
        raise ValueError("backfill_run_complete requires a positive, independently-derived "
                         "expected_cells; it must not be inferred from the rows on disk")

    results_dir = Path(results_dir)
    missing_artifacts = [n for n in REQUIRED_ARTIFACTS if not (results_dir / n).exists()]
    if missing_artifacts:
        raise FileNotFoundError(f"refusing to backfill {results_dir}: absent artifact(s) {missing_artifacts}")

    rows = parse_jsonl_rows(results_dir / "probe_results.jsonl")  # raises on corrupt
    dupes = [k for k, c in collections.Counter(cell_key(r) for r in rows).items() if c > 1]
    if dupes:
        raise ValueError(
            f"refusing to backfill {results_dir}: "
            + _summarize("duplicate cell key(s)", [list(k) for k in dupes])
        )
    if len(rows) != expected_cells:
        raise ValueError(
            f"refusing to backfill {results_dir}: {len(rows)} rows on disk but {expected_cells} "
            f"cells were asserted -- the directory is not the run it is claimed to be"
        )

    sig = signature
    if sig is None:
        sig_path = results_dir / "run_signature.txt"
        sig = sig_path.read_text().strip() if sig_path.exists() else None

    payload: dict[str, Any] = {
        "schema": SCHEMA_VERSION,
        "signature": sig,
        "expected_cells": int(expected_cells),
        "actual_rows": len(rows),
        "completed_at": datetime.now(UTC).isoformat(),
        "backfilled": True,
        "verified_by": verified_by,
        "note": note,
        "artifacts": {},
    }
    for name in REQUIRED_ARTIFACTS:
        path = results_dir / name
        payload["artifacts"][name] = {"sha256": sha256_file(path), "bytes": path.stat().st_size}
    IOU.write_json(results_dir / RUN_COMPLETE_FILE, payload)
    return payload
