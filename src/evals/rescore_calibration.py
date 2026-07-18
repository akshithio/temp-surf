"""P0-1 re-scorer: recompute calibration from saved per-sample predictions.

The canonical erm-full run wrote per-sample rows (y_true, full `probs` vector, `classes`) to
`predictions.jsonl`, so the corrected calibration can be recomputed WITHOUT re-probing and
without re-encoding -- a pure streaming pass over artifacts that already exist.

Memory-safe: predictions.jsonl is ~10 GB per (model, benchmark), so we accumulate per-cell
SUFFICIENT STATISTICS (ECE bins + NLL/Brier sums + support counts) rather than raw arrays,
exactly as score_segmentation_streamed does. Pinned to evals.metrics.multiclass_calibration by
tests/test_calibration_support.py so the two implementations cannot drift.

Fails loudly: a malformed row, or one missing y_true/probs/classes, raises with file:line
context rather than being silently skipped -- silently dropping rows would invalidate the
n_test reconciliation that proves nothing was lost.

Output columns per evaluation cell:
  old_nll / old_brier       the pre-fix convention, reproduced so the delta is auditable
  nll                       +inf when any true class had no column (honest, not smoothed)
  brier / union_brier       union-space: unsupported rows pay the unavoidable (0-1)^2 = 1
  shared_*                  ordinary calibration restricted to supported examples
  unseen_prevalence         the label-support diagnostic that explains the rest
  nll_eps_* / shared_nll_eps_*   epsilon sensitivity (1e-6/1e-9/1e-12/1e-15). If these move
                            materially, the NLL is an artefact of the floor, not the model.
                            NOTE old_nll == nll_eps_1e-12 by construction (the old clip).

Usage:
    python3 -m evals.rescore_calibration <predictions.jsonl> [...] > rescored.jsonl
"""

from __future__ import annotations

import collections
import json
import sys

import numpy as np

_NUM_EPS = 1e-15
_OLD_CLIP = 1e-12  # the pre-fix constant, reproduced only to report the delta
N_BINS = 10
_EDGES = np.linspace(0.0, 1.0, N_BINS + 1)

#: Epsilon sweep for the NLL floor sensitivity.
EPS_SWEEP: tuple[float, ...] = (1e-6, 1e-9, 1e-12, 1e-15)
_NEG_LOG_EPS = -np.log(np.asarray(EPS_SWEEP))

#: Fields identifying one evaluation cell.
CELL_KEYS = (
    "model",
    "benchmark",
    "probe_family",
    "split_regime",
    "evaluation_split",
    "label_budget",
    "holdout",
    "seed",
    "method",
    "budget_type",
)


class _Acc:
    """Streaming sufficient statistics for one evaluation cell."""

    __slots__ = (
        "bin_conf", "bin_correct", "bin_conf_s", "bin_correct_s",
        "nll_sum_seen", "sq_sum", "sq_sum_seen", "pt_sum_seen",
        "eps_sum_seen", "old_brier_sum", "n", "n_seen", "classes",
    )

    def __init__(self) -> None:
        self.bin_conf = np.zeros(N_BINS)
        self.bin_correct = np.zeros(N_BINS)
        self.bin_conf_s = np.zeros(N_BINS)
        self.bin_correct_s = np.zeros(N_BINS)
        self.nll_sum_seen = 0.0
        self.sq_sum = 0.0
        self.sq_sum_seen = 0.0
        self.pt_sum_seen = 0.0
        self.eps_sum_seen = np.zeros(len(EPS_SWEEP))  # per-eps -log(max(pt,eps)) over SEEN rows
        self.old_brier_sum = 0.0
        self.n = 0
        self.n_seen = 0
        self.classes: list | None = None

    def add(self, y_true: int, probs: list[float], classes: list[int]) -> None:
        if self.classes is None:
            self.classes = list(classes)
        p = np.asarray(probs, dtype=np.float64)
        cls = self.classes
        conf = float(p.max())
        pred = cls[int(p.argmax())]
        correct = 1.0 if pred == y_true else 0.0
        b = int(np.clip(np.digitize(conf, _EDGES[1:-1]), 0, N_BINS - 1))
        self.bin_conf[b] += conf
        self.bin_correct[b] += correct

        try:
            j = cls.index(y_true)
        except ValueError:
            j = -1
        sq = float((p**2).sum())
        self.sq_sum += sq
        self.n += 1
        if j >= 0:
            pt = float(p[j])
            self.n_seen += 1
            self.pt_sum_seen += pt
            self.sq_sum_seen += sq
            self.nll_sum_seen += -float(np.log(max(pt, _NUM_EPS)))
            # -log(max(pt, eps)) == min(-log(pt), -log(eps)); vectorised over the sweep.
            nl = -np.log(pt) if pt > 0.0 else np.inf
            self.eps_sum_seen += np.minimum(nl, _NEG_LOG_EPS)
            self.bin_conf_s[b] += conf
            self.bin_correct_s[b] += correct
            self.old_brier_sum += sq - 2.0 * pt + 1.0
        else:
            # old convention: Brier omitted the unit penalty for unsupported rows
            self.old_brier_sum += sq

    def finalize(self) -> dict[str, float]:
        n, ns = self.n, self.n_seen
        if n == 0:
            return {}
        n_unseen = n - ns
        ece_all = float(np.abs(self.bin_correct - self.bin_conf).sum() / n)
        union_brier = float((self.sq_sum - 2.0 * self.pt_sum_seen + n) / n)
        out: dict[str, float] = {
            "n_test": n,
            "ece": ece_all,
            "top_label_ece_all": ece_all,
            "nll": float(self.nll_sum_seen / n) if ns == n else float("inf"),
            "brier": union_brier,
            "union_brier": union_brier,
            "unseen_prevalence": float(1.0 - ns / n),
            "old_brier": float(self.old_brier_sum / n),
        }
        if ns > 0:
            out["shared_ece"] = float(np.abs(self.bin_correct_s - self.bin_conf_s).sum() / ns)
            out["shared_nll"] = float(self.nll_sum_seen / ns)
            out["shared_brier"] = float((self.sq_sum_seen - 2.0 * self.pt_sum_seen + ns) / ns)
        else:
            out["shared_ece"] = out["shared_nll"] = out["shared_brier"] = float("nan")
        # epsilon sensitivity: full-label smooths unsupported rows to -log(eps); shared floors
        # only genuinely-tiny supported probabilities.
        for i, e in enumerate(EPS_SWEEP):
            tag = f"{e:.0e}"
            out[f"nll_eps_{tag}"] = float((self.eps_sum_seen[i] + n_unseen * _NEG_LOG_EPS[i]) / n)
            out[f"shared_nll_eps_{tag}"] = float(self.eps_sum_seen[i] / ns) if ns > 0 else float("nan")
        # The pre-fix NLL clipped every row at 1e-12, so it is exactly the 1e-12 sweep point.
        out["old_nll"] = out[f"nll_eps_{_OLD_CLIP:.0e}"]
        return out


def rescore(paths: list[str], *, progress: bool = False):
    cells: dict[tuple, _Acc] = collections.defaultdict(_Acc)
    total_rows = 0
    for path in paths:
        rows_here = 0
        with open(path) as fh:
            for lineno, line in enumerate(fh, 1):
                if not line.strip():
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}:{lineno}: malformed JSON prediction row: {exc}") from exc
                missing = [k for k in ("y_true", "probs", "classes") if r.get(k) is None]
                if missing:
                    raise ValueError(
                        f"{path}:{lineno}: prediction row missing {missing}. This scorer only "
                        f"accepts multiclass rows carrying a full `probs` vector; binary rows "
                        f"(scalar `prob`) are unaffected by P0-1 and must not be passed here."
                    )
                key = tuple(r.get(k) for k in CELL_KEYS)
                cells[key].add(int(r["y_true"]), r["probs"], r["classes"])
                rows_here += 1
        total_rows += rows_here
        if progress:
            print(f"  read {rows_here} rows from {path}", file=sys.stderr, flush=True)

    emitted = 0
    for key, acc in cells.items():
        fin = acc.finalize()
        if not fin:
            continue
        # Indexed rather than zipped: `key` is built from CELL_KEYS above so the alignment is
        # guaranteed by construction, and this scorer has to run on the cluster's stock
        # python3 (3.9), which predates zip(strict=...).
        row = {name: key[i] for i, name in enumerate(CELL_KEYS)}
        row.update(fin)
        emitted += fin["n_test"]
        yield row
    if emitted != total_rows:  # every input row must land in exactly one cell
        raise ValueError(f"row reconciliation failed: read {total_rows} but accounted {emitted}")
    if progress:
        print(f"  reconciled {total_rows} rows across {len(cells)} cells", file=sys.stderr, flush=True)


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__, file=sys.stderr)
        return 2
    for row in rescore(argv, progress=True):
        print(json.dumps(row), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
