"""GPU work-splitting + launch helpers for ``src/main.py``.

main.py's unit of work is the list of ``(encoder, task)`` pairs. To use both GPUs
on digital-ag (2x RTX 3090), run two **sharded** processes: each is pinned to one
GPU via ``CUDA_VISIBLE_DEVICES`` and handed a disjoint, round-robin subset of the
pairs. The fan-out launcher does this for you::

    cd src && python utils/gputils.py        # one process per detected GPU, in parallel
    # or choose a count:  python utils/gputils.py 2

Inside main.py only two hooks are needed (already wired)::

    work = gputils.take_shard(work)   # keep only this process's (encoder, task) pairs
    device = gputils.device()         # "cuda" (the one visible GPU) or "cpu"

Because shards are split by (encoder, task) and the embedding cache is written
atomically (tmp + os.replace), two shards that happen to touch the same
benchmark/encoder cache are safe -- at worst one embedding is computed twice.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

try:
    import torch
except ImportError:
    torch = None  # type: ignore

REPO = Path(__file__).resolve().parents[2]


SHARD_ENV = "RB_SHARD"
NUM_SHARDS_ENV = "RB_NUM_SHARDS"


def gpu_count() -> int:
    if torch is None:
        return 0
    try:
        return int(torch.cuda.device_count())
    except Exception:
        return 0


def device() -> str:
    """The device main.py should hand to encoders: 'cuda' (the one visible GPU) or 'cpu'."""
    if torch is None:
        return "cpu"
    try:
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def shard_indices() -> tuple[int, int]:
    return int(os.environ.get(SHARD_ENV, 0)), int(os.environ.get(NUM_SHARDS_ENV, 1))


def take_shard(items: list) -> list:
    """Round-robin subset of ``items`` for this process's shard (identity if unsharded)."""
    idx, n = shard_indices()
    if n <= 1:
        return list(items)
    return [x for i, x in enumerate(items) if i % n == idx]


def fan_out(num_shards: int | None = None) -> int:
    """Launch one sharded ``main.py`` per GPU in parallel; tee per-shard logs.

    Each child gets ``CUDA_VISIBLE_DEVICES=i`` (so it sees exactly one GPU as
    cuda:0) plus the shard env, and runs the disjoint subset of (encoder, task)
    pairs. Returns the max child exit code. Falls back to a single process if no
    GPUs are visible.
    """
    n = num_shards or max(1, gpu_count())
    scratch = Path(os.environ.get("ROBUSTNESS_SCRATCH", REPO / "data"))
    log_dir = scratch / "output" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    per_shard_cores = max(1, (os.cpu_count() or 2) // n)
    procs = []
    for i in range(n):
        env = {
            **os.environ,
            "CUDA_VISIBLE_DEVICES": str(i),
            SHARD_ENV: str(i),
            NUM_SHARDS_ENV: str(n),
            "LOKY_MAX_CPU_COUNT": str(per_shard_cores),
        }
        log = open(log_dir / f"shard_{i}.log", "w")
        proc = subprocess.Popen(
            [sys.executable, "-u", "main.py"],
            cwd=str(REPO / "src"),
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        print(f"[gputils] shard {i + 1}/{n} -> GPU {i} | pid {proc.pid} | log {log.name}", flush=True)
        procs.append((proc, log))
    code = 0
    for proc, log in procs:
        code = max(code, proc.wait())
        log.close()
    print(f"[gputils] all {n} shard(s) done (max exit code {code})", flush=True)
    return code


if __name__ == "__main__":
    raise SystemExit(fan_out(int(sys.argv[1]) if len(sys.argv) > 1 else None))
