"""Performance tracking instrument for the robustness pipeline.

Records wall-clock, user-CPU, system-CPU time, MAC estimates, data
dimensions, and GPU utilization at every stage.  Use as a context manager::

    with measure("encode/clean", n_samples=128, n_features=128):
        ...

Every event is tagged with a thread-local **identity** (seed, holdout,
condition, method, budget, budget_type) so nested ``perf.measure`` calls
inside fit/score/sweep functions automatically know which cell they belong
to.  Callers set identity at the sweep boundary::

    perf.set_identity({"seed": 42, "holdout": "togo", ...})
    # all nested measure() calls inherit this identity

The logger is thread-safe so parallel probe workers can each record their
own timings.  All events are accumulated per-(encoder, task) and flushed
to a JSONL file at the end of each run.
"""

from __future__ import annotations

import json
import resource
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, asdict
from pathlib import Path
from threading import Lock
from typing import Any


try:
    import pynvml

    pynvml.nvmlInit()
    _NVML_OK = True
except Exception:
    _NVML_OK = False


@dataclass
class PerfEvent:
    name: str
    wall_s: float
    user_s: float
    sys_s: float
    macs: int | None = None
    n_samples: int | None = None
    n_features: int | None = None
    n_classes: int | None = None
    identity: dict[str, Any] | None = None
    gpu_util: float | None = None
    gpu_mem_mb: float | None = None
    extras: dict[str, Any] | None = None


_EVENTS: list[PerfEvent] = []
_LOCK = Lock()


# --------------------------------------------------------------------------- #
# Thread-local identity context — all measure() calls in this thread
# automatically get tagged, avoiding plumbing through every closure.
# --------------------------------------------------------------------------- #

_tls = threading.local()


def set_identity(identity: dict[str, Any] | None) -> None:
    _tls.identity = identity


def get_identity() -> dict[str, Any] | None:
    return getattr(_tls, "identity", None)


# --------------------------------------------------------------------------- #
# GPU snapshot (best-effort, returns None when unavailable)
# --------------------------------------------------------------------------- #


def _gpu_snapshot() -> tuple[float | None, float | None]:
    if not _NVML_OK:
        return None, None
    try:
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
        mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
        return float(util.gpu), float(mem.used / 1024**2)
    except Exception:
        return None, None


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #


def reset() -> None:
    with _LOCK:
        _EVENTS.clear()


@contextmanager
def measure(name: str, identity: dict[str, Any] | None = None, **extras: Any):
    start_wall = time.perf_counter()
    start_ru = resource.getrusage(resource.RUSAGE_SELF)
    gpu_util_start, gpu_mem_start = _gpu_snapshot()
    try:
        yield
    finally:
        wall = time.perf_counter() - start_wall
        end_ru = resource.getrusage(resource.RUSAGE_SELF)
        gpu_util_end, gpu_mem_end = _gpu_snapshot()
        with _LOCK:
            _EVENTS.append(PerfEvent(
                name=name,
                wall_s=round(wall, 4),
                user_s=round(end_ru.ru_utime - start_ru.ru_utime, 4),
                sys_s=round(end_ru.ru_stime - start_ru.ru_stime, 4),
                identity=identity or get_identity(),
                gpu_util=_avg(gpu_util_start, gpu_util_end),
                gpu_mem_mb=_avg(gpu_mem_start, gpu_mem_end),
                extras=extras or None,
            ))


def _avg(a: float | None, b: float | None) -> float | None:
    if a is None or b is None:
        return None
    return round((a + b) / 2, 1)


def log_static(
    name: str,
    *,
    macs: int | None = None,
    n_samples: int | None = None,
    n_features: int | None = None,
    n_classes: int | None = None,
    identity: dict[str, Any] | None = None,
    **extras: Any,
) -> None:
    """Record a dimension / MAC annotation (zero-duration event)."""
    with _LOCK:
        _EVENTS.append(PerfEvent(
            name=name,
            wall_s=0.0, user_s=0.0, sys_s=0.0,
            macs=macs, n_samples=n_samples, n_features=n_features,
            n_classes=n_classes,
            identity=identity or get_identity(),
            extras=extras or None,
        ))


def write_log(path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    with _LOCK:
        events = list(_EVENTS)
    with open(path, "w") as f:
        for ev in events:
            f.write(json.dumps(asdict(ev), default=str) + "\n")
    return len(events)
