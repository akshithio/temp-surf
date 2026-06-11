"""TESSERA frozen-encoder wrapper running the OPEN-SOURCE model on the time series.

TESSERA (Cambridge, ucam-eo) is a multimodal Sentinel-1 + Sentinel-2 pixel-timeseries
SSL model that produces a 128-d per-pixel embedding. The model weights are released
publicly (CC0). We load those weights and run the actual forward pass on each sample's
(possibly corrupted) S2 / S1 time series. Because the embedding is computed FROM the
input series, our sensor-off / temporal-drop corruptions change it: unlike the old
geotessera precomputed-product path, this wrapper is condition-SENSITIVE and can be
stress-tested the same way Presto / OlmoEarth are.

Architecture (reproduced verbatim from ucam-eo/tessera so the published checkpoint
loads strict=True):

    S2 backbone : TransformerEncoder(band_num=10, latent_dim=128, nhead=8,
                                     num_encoder_layers=8, dim_feedforward=4096)
    S1 backbone : TransformerEncoder(band_num=2,  latent_dim=128, nhead=8,
                                     num_encoder_layers=8, dim_feedforward=4096)
    fusion      : concat(s2_repr[512], s1_repr[512]) -> dim_reducer Linear(1024, 128)

Each backbone embeds bands to latent_dim*4 (=512), adds a sinusoidal day-of-year
positional encoding, runs an 8-layer transformer, and temporal-attention-pools to a
single 512-d vector. The SSL projection head is dropped at inference; the 128-d
``dim_reducer`` output is the embedding.

Weights:
    The checkpoint is NOT redistributed here. Download it once (e.g. v1.0 float32
    ``best_model_fsdp_20250427_084307.pt`` from the TESSERA release) and point the
    wrapper at it via ``weights_path=`` or the ``TESSERA_WEIGHTS`` env var. On a miss
    the wrapper raises with the expected location rather than silently degrading.

Caveats (deployment, deliberately not reconciled here):
  * S1 scale. TESSERA was trained on its own preprocessed S1 (linear-amplitude DN,
    band means ~[5484, 3003]); our Benchmark carries S1 in dB. Feeding dB through the
    TESSERA S1 normalization is a domain mismatch. The S2 path matches (raw reflectance,
    same 10 bands). Until the S1 preprocessing is reconciled, read S1-driven TESSERA
    numbers with that caveat -- the S2 channel still responds correctly to stress.
  * DOY. We use ``bench.doy`` (real day-of-year where the dataset has it; CropHarvest's
    is synthetic monthly).
  * Sequence length. TESSERA's sampler caps at 40 steps; the modules are length-agnostic
    (the positional encoding is computed analytically from DOY), so we feed the native
    benchmark length directly.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import thop
import torch
import torch.nn as nn

if TYPE_CHECKING:
    from dataio.get_input import Benchmark

TESSERA_EMBEDDING_DIM = 128

# TESSERA's 10 Sentinel-2 bands (our Benchmark lists these first, then a trailing NDVI
# we drop) and 2 Sentinel-1 bands, with the published per-band normalization constants.
TESSERA_S2_BANDS = ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"]
TESSERA_S1_BANDS = ["VV", "VH"]
S2_BAND_MEAN = np.array([1711.0938, 1308.8511, 1546.4543, 3010.1293, 3106.5083,
                         2068.3044, 2685.0845, 2931.5889, 2514.6928, 1899.4922], dtype=np.float32)
S2_BAND_STD = np.array([1926.1026, 1862.9751, 1803.1792, 1741.7837, 1677.4543,
                        1888.7862, 1736.3090, 1715.8104, 1514.5199, 1398.4779], dtype=np.float32)
S1_BAND_MEAN = np.array([5484.0407, 3003.7812], dtype=np.float32)
S1_BAND_STD = np.array([1871.2334, 1726.0670], dtype=np.float32)

# Derived data (weights cache) goes on ROBUSTNESS_SCRATCH when set (a big/fast disk on
# crowded boxes), else the repo's data/cache. Mirrors src/main.py.
_SCRATCH = Path(os.environ.get("ROBUSTNESS_SCRATCH", Path(__file__).resolve().parents[2] / "data"))
DEFAULT_WEIGHTS = _SCRATCH / "cache" / "tessera" / "best_model_fsdp_20250427_084307.pt"


# --------------------------------------------------------------------------- #
# Model definition (verbatim structure from ucam-eo/tessera, for strict load)
# --------------------------------------------------------------------------- #
def _build_torch_modules():
    """Define the TESSERA modules (torch is imported at module level)."""

    class TemporalPositionalEncoder(nn.Module):
        def __init__(self, d_model):
            super().__init__()
            self.d_model = d_model

        def forward(self, doy):  # doy: (B, T)
            position = doy.unsqueeze(-1).float()
            div_term = torch.exp(torch.arange(0, self.d_model, 2, dtype=torch.float)
                                 * -(math.log(10000.0) / self.d_model)).to(doy.device)
            pe = torch.zeros(doy.shape[0], doy.shape[1], self.d_model, device=doy.device)
            pe[:, :, 0::2] = torch.sin(position * div_term)
            pe[:, :, 1::2] = torch.cos(position * div_term)
            return pe

    class TemporalAwarePooling(nn.Module):
        def __init__(self, input_dim):
            super().__init__()
            self.query = nn.Linear(input_dim, 1)
            self.temporal_context = nn.GRU(input_dim, input_dim, batch_first=True)

        def forward(self, x):
            x_context, _ = self.temporal_context(x)
            w = torch.softmax(self.query(x_context), dim=1)
            return (w * x).sum(dim=1)

    class TransformerEncoder(nn.Module):
        def __init__(self, band_num, latent_dim, nhead=8, num_encoder_layers=8,
                     dim_feedforward=4096, dropout=0.1):
            super().__init__()
            self.embedding = nn.Sequential(
                nn.Linear(band_num, latent_dim * 4), nn.ReLU(),
                nn.Linear(latent_dim * 4, latent_dim * 4))
            self.temporal_encoder = TemporalPositionalEncoder(d_model=latent_dim * 4)
            layer = nn.TransformerEncoderLayer(
                d_model=latent_dim * 4, nhead=nhead, dim_feedforward=dim_feedforward,
                dropout=dropout, activation="relu", batch_first=True)
            self.transformer_encoder = nn.TransformerEncoder(layer, num_layers=num_encoder_layers)
            self.attn_pool = TemporalAwarePooling(latent_dim * 4)

        def forward(self, x):  # x: (B, T, band_num + 1); last column is DOY
            bands, doy = x[:, :, :-1], x[:, :, -1]
            x = self.embedding(bands) + self.temporal_encoder(doy)
            x = self.transformer_encoder(x)
            return self.attn_pool(x)

    class ProjectionHead(nn.Module):
        # Present only so the SSL checkpoint loads strict=True; unused at inference.
        def __init__(self, input_dim, hidden_dim, output_dim):
            super().__init__()
            blocks = []
            d = input_dim
            for _ in range(5):
                blocks += [nn.Linear(d, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU(inplace=False)]
                d = hidden_dim
            blocks += [nn.Linear(hidden_dim, output_dim)]
            self.net = nn.Sequential(*blocks)

        def forward(self, x):
            return self.net(x)

    class MultimodalBTModel(nn.Module):
        def __init__(self, s2_backbone, s1_backbone, projector, fusion_method="concat", latent_dim=128):
            super().__init__()
            self.s2_backbone = s2_backbone
            self.s1_backbone = s1_backbone
            self.projector = projector
            self.fusion_method = fusion_method
            in_dim = 8 * latent_dim if fusion_method == "concat" else 4 * latent_dim
            self.dim_reducer = nn.Sequential(nn.Linear(in_dim, latent_dim))

        def forward(self, s2_x, s1_x):
            s2_repr, s1_repr = self.s2_backbone(s2_x), self.s1_backbone(s1_x)
            fused = (torch.cat([s2_repr, s1_repr], dim=-1)
                     if self.fusion_method == "concat" else s2_repr + s1_repr)
            return self.dim_reducer(fused)

    return MultimodalBTModel, TransformerEncoder, ProjectionHead


@dataclass
class TesseraEncoder:
    """Frozen TESSERA encoder running the published model on each sample's time series."""

    name: str = "tessera"
    embedding_dim: int = TESSERA_EMBEDDING_DIM
    weights_path: str | Path | None = None  # None -> TESSERA_WEIGHTS env or DEFAULT_WEIGHTS
    device: str = "cpu"
    batch_size: int = 4096
    latent_dim: int = 128
    fusion_method: str = "concat"
    condition_invariant: bool = False  # runs the model on the (corrupted) input -> stress-testable
    _model: Any = field(default=None, repr=False)

    def compute_macs(self) -> int:
        self._ensure_loaded()
        B, T = 1, 12
        s2_b = torch.randn(B, T, len(TESSERA_S2_BANDS) + 1, device=self.device)
        s1_b = torch.randn(B, T, len(TESSERA_S1_BANDS) + 1, device=self.device)
        macs, _ = thop.profile(self._model, inputs=(s2_b, s1_b), verbose=False)
        return int(macs)

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        path = Path(self.weights_path or os.environ.get("TESSERA_WEIGHTS") or DEFAULT_WEIGHTS)
        if not path.exists():
            raise FileNotFoundError(
                f"TESSERA weights not found at {path}. Download the published checkpoint "
                f"(e.g. best_model_fsdp_20250427_084307.pt) and set weights_path= or the "
                f"TESSERA_WEIGHTS env var."
            )
        MultimodalBTModel, TransformerEncoder, ProjectionHead = _build_torch_modules()
        s2 = TransformerEncoder(band_num=10, latent_dim=self.latent_dim)
        s1 = TransformerEncoder(band_num=2, latent_dim=self.latent_dim)
        proj = ProjectionHead(self.latent_dim, 2048, 2048)
        model = MultimodalBTModel(s2, s1, proj, fusion_method=self.fusion_method, latent_dim=self.latent_dim)

        ckpt = torch.load(path, map_location=self.device)
        state = ckpt.get("model_state", ckpt.get("model_state_dict", ckpt))
        state = {(k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k): v for k, v in state.items()}
        model.load_state_dict(state, strict=True)  # exact match validates the reproduced architecture
        model.to(self.device).eval()
        for p in model.parameters():
            p.requires_grad = False
        self._model = model

    def _modality_input(self, modality, bands, src_bands, mean, std, doy):
        """Select TESSERA's bands by name, normalize, append DOY -> (N, T, len(bands)+1)."""
        col = {b: i for i, b in enumerate(src_bands)}
        idx = [col[b] for b in bands]  # all three benchmarks list these exact names
        x = modality[:, :, idx].astype(np.float32)
        x = (x - mean) / (std + 1e-9)
        return np.concatenate([x, doy[:, :, None].astype(np.float32)], axis=2)

    def encode(self, bench: "Benchmark") -> np.ndarray:
        self._ensure_loaded()
        doy = np.asarray(bench.doy, dtype=np.float32)  # (N, T) day-of-year
        s2 = self._modality_input(bench.s2, TESSERA_S2_BANDS, bench.s2_bands, S2_BAND_MEAN, S2_BAND_STD, doy)
        s1 = self._modality_input(bench.s1, TESSERA_S1_BANDS, bench.s1_bands, S1_BAND_MEAN, S1_BAND_STD, doy)

        out = []
        with torch.no_grad():
            for i in range(0, s2.shape[0], self.batch_size):
                s2_b = torch.from_numpy(s2[i:i + self.batch_size]).to(self.device)
                s1_b = torch.from_numpy(s1[i:i + self.batch_size]).to(self.device)
                out.append(self._model(s2_b, s1_b).cpu().numpy())
        return np.concatenate(out, axis=0).astype(np.float32)
