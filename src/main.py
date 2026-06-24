"""Orchestrator for the frozen-embedding robustness pipeline.

Edit the config block below, then run:

    cd src && python main.py
    cd src && python utils/gputils.py
"""

from __future__ import annotations

import os
import sys

from evals import compat
from evals import evals as EV
from evals.regimes import base as regime_base
from utils import cacheutils, gputils

# === Configuration ===========================================================
BENCHMARKS = ["cropharvest", "eurocropsml", "breizhcrops", "pastis_r"]
RUN_STAGES = ["gen_embeddings", "probing"]
SPLIT_REGIMES = ["random_id", "geographic_ood"]
ACTIVE_PROBES = ["logistic"]
BUDGET_REGIMES = {
    "source": [0.05, 0.10, 0.25, 1.0],
    "target": [0, 5, 10, 25, 50, EV.TARGET_ID_UPPER_BOUND],
}
MAX_SAMPLES = None
MAX_DENSE_PIXELS = 50_000  # sampled pixels per PASTIS fold partition
OVERWRITE_MODE = False
SEEDS = [0]
# =============================================================================

# Downstream loaders use this to decide whether partial/corrupt inputs warn or fail.
os.environ["OVERWRITE_MODE"] = "1" if OVERWRITE_MODE else ""


def main() -> int:
    enc_kwargs = {"device": gputils.device()}

    all_pairs = [(mod, bm) for bm in BENCHMARKS for mod in compat.eligible_models(bm)]
    work = gputils.take_shard(all_pairs)
    shard, nshards = gputils.shard_indices()
    failures: list[tuple[str, str, str]] = []

    for mod, bm in work:
        print(f"\n========== [shard {shard}/{nshards}] {mod} / {bm} ==========", flush=True)
        print(f"  split_regimes={SPLIT_REGIMES}", flush=True)
        print(f"  run_stages={RUN_STAGES}", flush=True)
        try:
            EV.run_pair(
                benchmark_name=bm,
                model_name=mod,
                seeds=SEEDS,
                max_samples=MAX_SAMPLES,
                max_dense_pixels=MAX_DENSE_PIXELS,
                split_regimes=SPLIT_REGIMES,
                run_stages=RUN_STAGES,
                active_probes=ACTIVE_PROBES,
                budget_regimes=BUDGET_REGIMES,
                overwrite_mode=OVERWRITE_MODE,
                enc_kwargs=enc_kwargs,
            )
        except NotImplementedError as exc:
            print(f"   [shard {shard}] {mod}/{bm} skipped (not implemented): {exc}", flush=True)
        except cacheutils.MissingEmbeddingCache:
            raise
        except Exception as exc:
            import traceback

            failures.append((mod, bm, f"{type(exc).__name__}: {exc}"))
            print(
                f"!! [shard {shard}] {mod}/{bm} FAILED: {type(exc).__name__}: {exc} (continuing; re-run to resume)",
                flush=True,
            )
            traceback.print_exc()

    regime_base.report_regime_problems()

    if failures:
        bar = "!" * 78
        print(f"\n{bar}\n[shard {shard}/{nshards}] {len(failures)} (model, benchmark) pair(s) FAILED:", flush=True)
        for mod, bm, reason in failures:
            print(f"  - {mod}/{bm}: {reason}", flush=True)
        print(f"{bar}", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
