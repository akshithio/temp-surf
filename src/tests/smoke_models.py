"""GPU smoke test for every active frozen model.

Configuration is intentionally kept in this file so the run command stays
short and reproducible.
"""

from __future__ import annotations

import numpy as np
import torch

from dataio.get_input import get_input
from utils import cacheutils

BENCHMARK = "cropharvest"
MAX_SAMPLES = 2
EXPECTED_DIMS = {
    "presto": 128,
    "olmoearth": 768,
    "galileo": 768,
    "agrifm": 1024,
    "tessera": 128,
}


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA device is required for the model smoke test")
    benchmark = get_input(
        BENCHMARK,
        root=cacheutils.INPUT_ROOT,
        max_samples=MAX_SAMPLES,
        shuffle=False,
    )
    for name, expected_dim in EXPECTED_DIMS.items():
        model = cacheutils.build_model(name, device="cuda:0")
        embeddings = model.encode(benchmark)
        expected_shape = (benchmark.n_samples, expected_dim)
        if embeddings.shape != expected_shape:
            raise AssertionError(f"{name}: expected {expected_shape}, received {embeddings.shape}")
        if not np.isfinite(embeddings).all():
            raise AssertionError(f"{name}: embeddings contain non-finite values")
        print(f"{name}: shape={embeddings.shape}, finite=True", flush=True)


if __name__ == "__main__":
    main()
