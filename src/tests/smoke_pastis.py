"""Single-tile PASTIS-R dense-encoder smoke test."""

from __future__ import annotations

import numpy as np
import torch

from dataio.get_input import get_input
from utils import cacheutils

MAX_PATCHES = 1
EXPECTED_DIMS = {
    "presto": 128,
    "olmoearth": 768,
    "galileo": 768,
    "agrifm": 1024,
    "tessera": 128,
}


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA device is required for the PASTIS-R smoke test")
    benchmark = get_input(
        "pastis",
        root=cacheutils.INPUT_ROOT,
        max_samples=MAX_PATCHES,
        shuffle=False,
    )
    tile_id, _fold, tile, labels = next(benchmark.iter_tiles())
    for name, expected_dim in EXPECTED_DIMS.items():
        encoder = cacheutils.build_encoder(name, device="cuda:0")
        features = (
            encoder.encode_dense(tile)
            if hasattr(encoder, "encode_dense")
            else encoder.encode(tile.pixel_benchmark())
        )
        expected_shape = (len(labels), expected_dim)
        if features.shape != expected_shape:
            raise AssertionError(f"{name}: expected {expected_shape}, received {features.shape}")
        if not np.isfinite(features).all():
            raise AssertionError(f"{name}: dense features contain non-finite values")
        print(f"{name}/{tile_id}: shape={features.shape}, finite=True", flush=True)


if __name__ == "__main__":
    main()
