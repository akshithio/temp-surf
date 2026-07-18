"""P0-1 validation: check each re-scored output against the original canonical artifact.

For every (box, benchmark, model) we assert, per the P0-1 acceptance criteria:

  1. identical experimental-cell keys  - the re-score covers exactly the original's cells
  2. identical n_test                  - no prediction rows dropped or double counted
  3. old NLL/Brier reproduced          - re-scoring the saved predictions with the PRE-FIX
                                         convention reproduces the number the original run
                                         wrote, proving we read the same data and that the
                                         only difference downstream is the correction itself
  4. no dropped rows                   - (enforced inside the scorer's reconciliation too)

Usage:
    python3 viz/validate_rescore.py <rescored_dir>
"""

from __future__ import annotations

import glob
import json
import os
import sys

ORIG_ROOT = os.path.join(os.path.dirname(__file__), "data")

CELL_KEYS = (
    "model", "benchmark", "probe_family", "split_regime", "evaluation_split",
    "label_budget", "holdout", "seed", "method", "budget_type",
)
TOL = 1e-6


def _key(r: dict) -> tuple:
    return tuple(r.get(k) for k in CELL_KEYS)


def _load_original(box: str, bench: str, model: str) -> dict[tuple, dict]:
    p = f"{ORIG_ROOT}/{box}/output-erm-full-20260711/results/{model}/{bench}/probe_results.jsonl"
    out = {}
    if not os.path.exists(p):
        return out
    with open(p) as fh:
        for line in fh:
            r = json.loads(line)
            if r.get("probe_family") is None:
                r["probe_family"] = "logistic"
            out[_key(r)] = r
    return out


def validate(rescored_dir: str) -> int:
    files = sorted(glob.glob(os.path.join(rescored_dir, "*__*__*.jsonl")))
    if not files:
        print(f"no re-scored files in {rescored_dir}", file=sys.stderr)
        return 2
    failures: list[str] = []
    print(f"{'file':34} {'cells':>6} {'n_test':>12} {'keys':>6} {'n_test':>7} {'old_nll':>8} {'old_brier':>9}")
    print("-" * 92)
    for f in files:
        tag = os.path.basename(f)[: -len(".jsonl")]
        box, bench, model = tag.split("__")
        rescored = [json.loads(line) for line in open(f)]
        orig = _load_original(box, bench, model)
        if not orig:
            failures.append(f"{tag}: original probe_results.jsonl not found")
            continue

        by_key = {_key(r): r for r in rescored}
        # 1. cell keys: the re-score must cover exactly the original's multiclass cells.
        #    (originals may hold cells whose probe wrote no probs; compare on the intersection
        #    and report any original cell the re-score failed to produce.)
        missing = set(orig) - set(by_key)
        extra = set(by_key) - set(orig)
        keys_ok = not missing and not extra

        n_ok = nll_ok = brier_ok = True
        n_cells = 0
        n_rows = 0
        for k, rs in by_key.items():
            o = orig.get(k)
            if o is None:
                continue
            n_cells += 1
            n_rows += rs["n_test"]
            if o.get("n_test") is not None and int(o["n_test"]) != int(rs["n_test"]):
                n_ok = False
            # 3. the pre-fix convention must reproduce the original's written value
            if o.get("nll") is not None and abs(float(o["nll"]) - rs["old_nll"]) > TOL:
                nll_ok = False
            if o.get("brier") is not None and abs(float(o["brier"]) - rs["old_brier"]) > TOL:
                brier_ok = False

        def mark(b: bool) -> str:
            return "ok" if b else "FAIL"

        print(f"{tag:34} {n_cells:>6} {n_rows:>12,} {mark(keys_ok):>6} {mark(n_ok):>7} "
              f"{mark(nll_ok):>8} {mark(brier_ok):>9}")
        if not keys_ok:
            failures.append(f"{tag}: cell-key mismatch (missing={len(missing)} extra={len(extra)})")
        if not n_ok:
            failures.append(f"{tag}: n_test mismatch vs original")
        if not nll_ok:
            failures.append(f"{tag}: old_nll does not reproduce original nll")
        if not brier_ok:
            failures.append(f"{tag}: old_brier does not reproduce original brier")

    print()
    if failures:
        print("VALIDATION FAILED:")
        for x in failures:
            print("  -", x)
        return 1
    print(f"VALIDATION PASSED for all {len(files)} files "
          f"(cell keys, n_test, old NLL/Brier reproduction).")
    return 0


if __name__ == "__main__":
    raise SystemExit(validate(sys.argv[1] if len(sys.argv) > 1 else "/tmp/rescored"))
