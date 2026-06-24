# Running the probe pipeline across multiple machines

The eventual plan is to run this on **several machines at once** to split the work. This doc explains
how the work is partitioned, how to launch it on each machine, and — most importantly — **the
invariants every machine must agree on** so the results are comparable and never corrupt each other.

## The unit of work

`main.py`'s unit of work is a `(model, benchmark)` pair (the compatibility matrix in
`evals/compat.py` decides which models run on each benchmark — currently ~20 pairs). Each pair is
**independent**: it builds its own benchmark cache, its own embedding cache, and writes to its own
results directory `data/output/results/<model>/<benchmark>/`. Two different pairs never write to the
same file. This is what makes splitting across machines safe.

`gputils.take_shard` round-robins the pair list by `(RB_SHARD, RB_NUM_SHARDS)`, so shard `s` of `n`
owns pairs `s, s+n, s+2n, …`. Sharding is by **whole pairs**, so a given `(model, benchmark)` is
owned by exactly one shard → no two processes ever append to the same `probe_results.jsonl`.

## How to split the work

Pick **one** of these. The GPU-shard approach is preferred (balances automatically).

### A. Global GPU sharding (preferred)

Treat every GPU across all machines as one global shard. `total = sum of GPU counts`. On each
machine set its starting offset and the global total, then run the fan-out launcher:

```bash
# Example: machine X has 2 GPUs, machine Y has 2 GPUs  -> 4 global shards.
# On machine X (owns global shards 0,1):
RB_SHARD_BASE=0 RB_NUM_SHARDS=4 ./sync.sh-equivalent  # then, in the repo on the box:
cd src && RB_SHARD_BASE=0 RB_NUM_SHARDS=4 python utils/gputils.py

# On machine Y (owns global shards 2,3):
cd src && RB_SHARD_BASE=2 RB_NUM_SHARDS=4 python utils/gputils.py
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
BENCHMARKS = ["breizhcrops", "pastis_r"]
```

(Or run a single process per box with one GPU via `python main.py`.) Just make sure the slices are
disjoint and together cover everything.

## What MUST match across every machine

The results are only comparable — and the resume guards only behave — if all machines agree on the
things that **define the numbers**. The pipeline enforces most of this for you, but you have to set
it up correctly:

| Must match | Why | How it's enforced / checked |
|---|---|---|
| **Code commit** (`src/`) | Different probe/loader/model code → different numbers | `_run_signature` hashes `probes.py`, `evals.py`, `main.py`, `ioutils.py`, `cacheutils.py`, the model wrapper, and every regime file. A results dir with a *different* signature is **refused** on resume (unless `OVERWRITE_MODE=True`). |
| **Staged input data** | Different data → different embeddings | `bench_tag` folds a recursive content fingerprint of `data/input/benchmarks/<bench>` into every cache key. If a machine has stale/different data, its caches won't collide with the others' (different key). |
| **Model weights** | Different checkpoint → different embeddings | `_checkpoint_fingerprint` folds the resolved checkpoint's identity into the embedding cache key (and the run signature). |
| **Config block in `main.py`** | `SEEDS`, `ACTIVE_PROBES`, `SPLIT_REGIMES`, `MAX_SAMPLES`, `MAX_DENSE_PIXELS` all change results | Folded into `_run_signature`; a mismatch is refused on resume. |
| **Python env / deps** | sklearn / torch / numpy versions shift numbers | Not auto-checked — build with the same `pyproject.toml` (`uv pip install -e .`) on every machine. |
| **Output destination** | Where results land | Either a **shared** filesystem (sshfs/NFS) so all shards write into one tree, or **per-machine** trees that you merge afterward (they're disjoint by pair, so a plain copy merges them). |

### Caches are content-keyed → safe to share
`data/cache/` (benchmark pickles + embeddings) is keyed by code+data+checkpoint hashes and written
atomically (`tmp` + `os.replace`). Two shards that happen to touch the same model/benchmark cache
are safe — at worst one embedding is computed twice. So a **shared** `data/` (e.g. the cranberry
scratch over sshfs) is fine and avoids re-encoding the same pair twice.

## Per-machine pre-flight checklist

Run through this on **each** machine before kicking off its shard:

1. **Code is the same commit** as the others (`git rev-parse HEAD` matches; push via `./sync.sh push`).
2. **Env built**: `uv pip install -e .` from the project conda env; `python -c "import torch, sklearn"`
   works; `ruff check src` and `pytest src/tests -q` pass (catches a broken sync).
3. **Data staged & identical**: `data/input/benchmarks/<bench>` present for every benchmark this shard
   will touch, and byte-identical to the other machines (same source). If you changed staged data in
   a way the cheap fingerprint can't see, bump `DATA_VERSION` **identically everywhere**.
4. **Weights present** for every model this shard will run (`data/input/models/...`, or the
   `*_WEIGHTS` env overrides) — set the same way on every machine.
5. **GPU visible**: `nvidia-smi` shows the expected GPUs; `python -c "import torch; print(torch.cuda.device_count())"`.
6. **Shard env correct** (approach A): `RB_SHARD_BASE` = sum of GPU counts on machines ordered before
   this one; `RB_NUM_SHARDS` = global GPU total — the **same** total on every machine. Sanity check:
   the union of `[RB_SHARD_BASE .. RB_SHARD_BASE+local_gpus)` across machines must be exactly
   `0..RB_NUM_SHARDS-1` with no gaps or overlaps.
7. **Output destination decided**: shared FS (preferred) or per-machine dirs to merge later.
8. **Disk**: dense embeddings (PASTIS especially) are large; confirm scratch has room.

## Merging per-machine outputs (only if NOT on a shared FS)

Each machine writes disjoint `data/output/results/<model>/<benchmark>/` trees, so merge by copying:

```bash
rsync -a machineY:.../data/output/results/  ./data/output/results/   # disjoint pairs -> no conflicts
```

`probe_results.jsonl` / `predictions.jsonl` / `perf.jsonl` are append-only per pair; `summary.csv`,
`deltas.csv`, `metric_roles.json`, `run_signature.txt` are per-pair too. Nothing needs a real merge —
just gather the per-pair directories into one tree.

## A note on resume / partial runs
Within a single `(model, benchmark)` the pipeline resumes per-row (classification) / per-family
(segmentation), so a crashed shard re-run continues where it left off. The process exits **nonzero**
if any pair failed, so a scheduler can tell a partial run from a clean one. Don't point two shards at
the same pair concurrently — `take_shard` already prevents this, so don't override `RB_SHARD` by hand
in a way that double-assigns a pair.
