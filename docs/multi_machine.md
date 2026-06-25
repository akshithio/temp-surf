# Running the probe pipeline across multiple machines

The eventual plan is to run this on **several machines at once** to split the work. This doc explains
how the work is partitioned, how to launch it on each machine, and — most importantly — **the
invariants every machine must agree on** so the results are comparable and never corrupt each other.

## The unit of work

`main.py`'s unit of work is a `(model, benchmark)` pair (the compatibility matrix in
`evals/compat.py` decides which models run on each benchmark — currently ~20 pairs). Each pair is
**independent** only when each machine has its own writable `data/cache` and `data/output` tree.
Do not point multiple machines at the same `data/output`, even when shards are meant to be disjoint:
reruns, logs, partial files, and accidental shard/config overlap become hard to audit.

`gputils.take_shard` round-robins the pair list by `(RB_SHARD, RB_NUM_SHARDS)`, so shard `s` of `n`
owns pairs `s, s+n, s+2n, …`. Sharding is by **whole pairs**, so a given `(model, benchmark)` is
owned by exactly one shard → no two processes ever append to the same `probe_results.jsonl`.

## How to split the work

Pick **one** of these. The GPU-shard approach is preferred (balances automatically).

### A. Global GPU sharding (preferred)

Treat every GPU across all machines as one global shard. `total = sum of GPU counts`. Set
`LAUNCH_GPU_SHARDS = True` in `src/main.py`. On each machine set its starting offset and the global
total, then run `main.py`:

```bash
# Example: machine X has 2 GPUs, machine Y has 2 GPUs  -> 4 global shards.
# On machine X (owns global shards 0,1):
RB_SHARD_BASE=0 RB_NUM_SHARDS=4  # then, in the repo on the box:
cd src && RB_SHARD_BASE=0 RB_NUM_SHARDS=4 python main.py

# On machine Y (owns global shards 2,3):
cd src && RB_SHARD_BASE=2 RB_NUM_SHARDS=4 python main.py
```

`fan_out` gives each local GPU a unique **global** shard index `RB_SHARD_BASE + i` out of
`RB_NUM_SHARDS`. A single-box run needs neither variable (base 0, total = local GPU count) and is
unchanged. Per-shard logs land in `data/output/logs/shard_<globalindex>.log`.

### B. Split the benchmark/model list by hand

Simplest, no env math: give each machine a disjoint slice of the config in `main.py`.

```python
# machine X
BENCHMARKS = ["cropharvest", "eurocropsml"]
# machine Y
BENCHMARKS = ["breizhcrops", "pastis"]
```

(Or run a single process per box with one GPU via `python main.py`.) Just make sure the slices are
disjoint and together cover everything.

## What MUST match across every machine

The results are only comparable — and the resume guards only behave — if all machines agree on the
things that **define the numbers**. The pipeline enforces most of this for you, but you have to set
it up correctly:

| Must match | Why | How it's enforced / checked |
|---|---|---|
| **Code commit** (`src/`) | Different probe/loader/model code → different numbers | `_run_signature` hashes `main.py`, the probe/eval/confound/perf utilities, cache/run-state utilities, the model wrapper and its helper files, and every regime file. A results dir with a *different* signature is **refused** on resume (unless `OVERWRITE_MODE=True`). |
| **Staged input data** | Different data → different embeddings | `bench_tag` folds a recursive content fingerprint of `data/input/benchmarks/<bench>` into every cache key. If a machine has stale/different data, its caches won't collide with the others' (different key). |
| **Model weights** | Different checkpoint → different embeddings | `_checkpoint_fingerprint` folds the resolved checkpoint's identity into the embedding cache key (and the run signature). |
| **Config block in `main.py`** | `SEEDS`, `ACTIVE_PROBES`, `SPLIT_REGIMES`, `MAX_SAMPLES`, `MAX_DENSE_PIXELS` all change results | Folded into `_run_signature`; a mismatch is refused on resume. |
| **Python env / deps** | sklearn / torch / numpy versions shift numbers | Not auto-checked — build with the same `pyproject.toml` (`uv pip install -e .`) on every machine. |
| **Output destination** | Where results land | Use **per-machine** `data/output` trees and merge/read them afterward. Cranberry and dewberry share `/local/scratch/a`, so a single shared output tree is not an isolation boundary. |

### Current machine layout

The repositories intentionally point `data/` at host-specific run roots:

| Machine | Repo `data/` | `data/input` | `data/cache` + `data/output` |
|---|---|---|---|
| cranberry | `/local/scratch/a/agarapat/robustness-run-data/cranberry` | symlink to `/local/scratch/a/agarapat/robustness-data/input` | cranberry-only |
| dewberry | `/local/scratch/a/agarapat/robustness-run-data/dewberry` | symlink to `/local/scratch/a/agarapat/robustness-data/input` | dewberry-only |
| digital-ag | `/var/tmp/agarapat/robustness-run-data/digital-ag` | symlink to `/var/tmp/agarapat/robustness-data/input` | digital-ag-only |

The input trees must match; the cache/output trees must not be shared.

### Caches are content-keyed, but still isolate them per machine
`data/cache/` (benchmark pickles + embeddings) is keyed by code+data+checkpoint hashes and written
atomically (unique temp path + `os.replace`). Two shards that happen to touch the same model/benchmark cache
should not corrupt each other, but shared cache contention makes crashes and cleanup harder to reason about.
Keep `data/cache` per-machine and merge only final outputs.

## Per-machine pre-flight checklist

Run through this on **each** machine before kicking off its shard:

1. **Code is the same commit** as the others (`git rev-parse HEAD` matches).
2. **Env built**: `uv pip install -e .` from the project conda env; `python -c "import torch, sklearn"`
   works; `ruff check src` and `pytest src/tests -q` pass (catches a broken sync).
3. **Data staged & identical**: `data/input/benchmarks/<bench>` present for every benchmark this shard
   will touch, and byte-identical to the other machines (same source). If you intentionally run with
   `DATA_FINGERPRINT=top`, bump `DATA_VERSION` **identically everywhere** after any staged-data edit.
4. **Weights present** for every model this shard will run (`data/input/models/...`, or the
   `*_WEIGHTS` env overrides) — set the same way on every machine.
5. **GPU visible**: `nvidia-smi` shows the expected GPUs; `python -c "import torch; print(torch.cuda.device_count())"`.
6. **Shard env correct** (approach A): `RB_SHARD_BASE` = sum of GPU counts on machines ordered before
   this one; `RB_NUM_SHARDS` = global GPU total — the **same** total on every machine. Sanity check:
   the union of `[RB_SHARD_BASE .. RB_SHARD_BASE+local_gpus)` across machines must be exactly
   `0..RB_NUM_SHARDS-1` with no gaps or overlaps.
7. **Output destination isolated**: `data/output` must resolve to that machine's run root, not a shared root.
8. **Disk**: dense embeddings (PASTIS especially) are large; confirm scratch has room.

## Merging per-machine outputs

Each machine writes disjoint `data/output/results/<model>/<benchmark>/` trees, so merge by copying:

```bash
rsync -a machineY:.../data/output/results/  ./data/output/results/   # disjoint pairs -> no conflicts
```

`probe_results.jsonl` / `predictions.jsonl` / `perf.jsonl` are append-only per pair; `summary.csv`,
`deltas.csv`, `metric_roles.json`, `split_manifest.json`, `run_signature.txt` are per-pair too. Nothing
needs a real merge — just gather the per-pair directories into one tree.

## A note on resume / partial runs
Within a single `(model, benchmark)` the pipeline resumes per-row (classification) / per-family
(segmentation), so a crashed shard re-run continues where it left off. The process exits **nonzero**
if any pair failed, so a scheduler can tell a partial run from a clean one. Don't point two shards at
the same pair concurrently — `take_shard` already prevents this, so don't override `RB_SHARD` by hand
in a way that double-assigns a pair.
