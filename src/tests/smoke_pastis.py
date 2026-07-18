"""Single-tile PASTIS-R dense-model smoke test."""

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


# Under the common-input S2-only contract, models that consume S1 (and/or coordinates) MUST change,
# while models already S2-only MUST be identical --- guards the v4 "S1 zeroed but still present" bug.
S2ONLY_CHANGES = {"galileo", "presto", "tessera", "raw"}
S2ONLY_SAME = {"olmoearth", "agrifm"}


def _dense(model, tile) -> np.ndarray:
    return model.encode_dense(tile) if hasattr(model, "encode_dense") else model.encode(tile.pixel_benchmark())


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
    _, _, s2o_tile, _ = next(benchmark.s2_only().iter_tiles())  # same patch, S2-only contract
    for name, expected_dim in EXPECTED_DIMS.items():
        model = cacheutils.build_model(name, device="cuda:0")
        features = _dense(model, tile)
        expected_shape = (len(labels), expected_dim)
        if features.shape != expected_shape:
            raise AssertionError(f"{name}: expected {expected_shape}, received {features.shape}")
        if not np.isfinite(features).all():
            raise AssertionError(f"{name}: dense features contain non-finite values")

        # S2-only contract: recompute on the S2-only tile and compare.
        s2o_features = _dense(model, s2o_tile)
        identical = np.allclose(features, s2o_features)
        if name in S2ONLY_SAME and not identical:
            raise AssertionError(f"{name}: s2_only changed an already-S2-only model (unexpected)")
        if name in S2ONLY_CHANGES and identical:
            raise AssertionError(
                f"{name}: s2_only did NOT change the embedding --- S1/coordinates were not actually "
                f"removed (the v4 bug). Contract not enforced for this model."
            )
        verdict = "== full" if identical else "!= full"
        print(f"{name}/{tile_id}: shape={features.shape}, finite=True, s2_only {verdict}", flush=True)


if __name__ == "__main__":
    main()
