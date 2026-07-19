"""One-time preflight: portable content digest of each benchmark's consumed inputs.

TEMPORARY UTILITY -- part of the freeze-and-run migration, to be removed before the repository's
final-release cleanup (see the pre-release TODO). No command-line arguments: edit the CONFIG block
below and run it. On Gilbreth this is a CPU Slurm job (it hashes tens of GB); never run it on the
frontend.

What it does, per benchmark in CONFIG["benchmarks"]:
  * enumerate EXACTLY the files the loader consumes (all inputs, minus PASTIS's unused products);
  * hash each file's full content and combine via the canonical aggregate digest -- independent of
    mtimes, directory sizes, inodes, absolute paths, and traversal order;
  * if CONFIG["reference"] has an expected digest for that benchmark, FAIL when they differ -- this
    is the cross-machine equality check. A failure to read or hash any input is a hard error, never
    a downgrade to a weaker check.

Publication is transactional: ALL digests are computed and EVERY configured reference validated
before anything is written. On any mismatch the run returns failure and leaves cache.json
unchanged; on full success every digest is merged into cache.json in ONE atomic replacement (so
concurrently-written embedding records are preserved). ``write=False`` performs ZERO filesystem
writes.

Cross-machine use: run on each assigned machine, then paste each machine's printed digest into
CONFIG["reference"] and re-run everywhere; a mismatch means the data differs and the run must stop.
"""

from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import hashlib  # noqa: E402
import json  # noqa: E402

from utils import artifacts  # noqa: E402
from utils import cacheutils as C  # noqa: E402


def _aggregate_sha256(root, rels: list[str]) -> str:
    """Portable content digest: sorted relative POSIX paths + each file's full SHA-256."""
    h = hashlib.sha256()
    for rel in sorted(rels):
        digest = artifacts.sha256_file(root / rel)
        if digest is None:
            raise FileNotFoundError(f"cannot hash consumed input: {root / rel}")
        h.update(rel.encode())
        h.update(b"\0")
        h.update(digest.encode())
        h.update(b"\n")
    return h.hexdigest()


# ===================== CONFIG (edit me; no CLI) =============================
CONFIG = {
    # Benchmarks to hash on THIS machine.
    "benchmarks": ["cropharvest", "eurocropsml", "breizhcrops", "pastis"],
    # Known-good digests to cross-check against (benchmark -> digest). Leave a benchmark out to
    # only compute+record it; include it to FAIL on any difference. Populate from other machines.
    "reference": {
        # "cropharvest": "….",
    },
    # Write the digest files (set False for a pure dry-run comparison).
    "write": True,
}
# ===========================================================================


def _consumed_files(benchmark: str, root: Path) -> list[str]:
    """Sorted portable relative POSIX paths of exactly the files a benchmark's loader reads."""
    if benchmark == "pastis":
        # The release also ships large UNUSED products (descending-orbit S1D, instance
        # annotations). Hash only what the loader consumes: metadata + S2 + S1A + TARGET.
        meta = root / "metadata.geojson"
        if not meta.exists():
            raise FileNotFoundError(f"PASTIS metadata missing: {meta}")
        geo = json.loads(meta.read_text())
        patch_ids = sorted({int(f["properties"]["ID_PATCH"]) for f in geo["features"]})
        rels = ["metadata.geojson"]
        for pid in patch_ids:
            for subdir, tmpl in (("DATA_S2", "S2_{}.npy"), ("DATA_S1A", "S1A_{}.npy"), ("ANNOTATIONS", "TARGET_{}.npy")):
                rel = f"{subdir}/{tmpl.format(pid)}"
                if not (root / rel).exists():
                    raise FileNotFoundError(f"PASTIS input missing: {root / rel}")
                rels.append(rel)
        return sorted(rels)
    # Tabular benchmarks: every regular file under the benchmark input dir is loader input.
    if not root.is_dir():
        raise FileNotFoundError(f"benchmark input dir missing: {root}")
    return sorted(p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file())


def main() -> int:
    # Transactional: compute EVERY digest and validate EVERY configured reference FIRST, writing
    # nothing. Only on full success (and write=True) do we publish -- so a later benchmark's mismatch
    # can never leave an earlier benchmark's file written, and a dry run touches the filesystem zero
    # times (not even directory creation).
    computed: dict[str, str] = {}
    failures = []
    for benchmark in CONFIG["benchmarks"]:
        root = C.INPUT_ROOT / benchmark
        rels = _consumed_files(benchmark, root)
        digest = _aggregate_sha256(root, rels)
        computed[benchmark] = digest
        print(f"[preflight] {benchmark}: {len(rels)} files -> {digest}")
        ref = CONFIG["reference"].get(benchmark)
        if ref is not None and ref != digest:
            failures.append(f"{benchmark}: {digest} != reference {ref}")
    if failures:
        print("[preflight] DATASET DIGEST MISMATCH -- data differs across machines; STOP (nothing written):")
        for f in failures:
            print(f"  - {f}")
        return 1
    if CONFIG["write"]:
        C.update_cache(datasets=computed)  # one atomic merge; concurrent embedding records preserved
        print("[preflight] all benchmarks hashed and recorded in cache.json")
    else:
        print("[preflight] all benchmarks hashed (dry run; no files written)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
