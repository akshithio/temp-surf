from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from models.agrifm import AGRIFM_S2_BANDS, AgriFMEncoder
from utils import cacheutils


class FakeAgriFM(torch.nn.Module):
    def forward(self, x):
        pooled = x.mean(dim=(1, 3, 4))
        return {"encoder_features": pooled[:, :, None, None]}


def _agrifm_bench() -> SimpleNamespace:
    bands = ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12", "NDVI"]
    s2 = np.zeros((2, 4, len(bands)), dtype=np.float32)
    for idx, band in enumerate(bands):
        s2[:, :, idx] = idx + 1
    s2_mask = np.ones((2, 4), dtype=np.float32)
    s2_mask[1, 2:] = 0.0
    return SimpleNamespace(s2=s2, s2_mask=s2_mask, s2_bands=bands)


def test_agrifm_series_uses_official_s2_band_order_and_masks_missing_frames() -> None:
    enc = AgriFMEncoder(load_model=lambda repo, weights, device: FakeAgriFM())

    series = enc.to_agrifm_series(_agrifm_bench())

    assert series.shape == (2, 32, 10)
    assert AGRIFM_S2_BANDS == ["B2", "B3", "B4", "B8", "B5", "B6", "B7", "B8A", "B11", "B12"]
    assert np.all(series[1, -1] == 0.0)
    assert np.any(series[0, -1] != 0.0)


def test_agrifm_encode_pools_mocked_encoder_features() -> None:
    enc = AgriFMEncoder(
        load_model=lambda repo, weights, device: FakeAgriFM(),
        batch_size=2,
        tile_size=2,
        device="cpu",
    )

    emb = enc.encode(_agrifm_bench())

    assert emb.shape == (2, 10)
    assert enc.embedding_dim == 10
    np.testing.assert_allclose(emb[1], enc.to_agrifm_series(_agrifm_bench())[1].mean(axis=0), atol=1e-6)


def test_agrifm_is_registered_without_loading_external_weights() -> None:
    enc = cacheutils.build_encoder(
        "agrifm",
        repo_path=Path("/tmp/no-load"),
        weights_path=Path("/tmp/no-load.pth"),
        load_model=lambda repo, weights, device: FakeAgriFM(),
    )

    assert isinstance(enc, AgriFMEncoder)
