"""TESSERA v1.1 frozen model for benchmark pixel time series.

This wrapper matches the released v1.1 MPC model-only checkpoint. TESSERA v1.1
uses separate S2 and merged-S1 temporal backbones, produces a 192-dimensional
representation, and publishes the first 128 dimensions for downstream use.
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
TESSERA_REPRESENTATION_DIM = 192
TESSERA_LATENT_DIM = 192

# Upstream source: ucam-eo/tessera `tessera_infer_QAT/src/datasets/v1_1_norm_stats.py`.
# The released checkpoints are bound to this non-wavelength-ordered S2 layout.
TESSERA_S2_BANDS = ["B4", "B2", "B3", "B8", "B8A", "B5", "B6", "B7", "B11", "B12"]
TESSERA_S1_BANDS = ["VV", "VH"]

S2_BAND_MEAN = np.array(
    [2683.4553, 2223.3630, 2432.0950, 3633.1970, 3602.1755, 3006.4324, 3400.2710, 3515.6392, 2456.9163, 1983.8783],
    dtype=np.float32,
)
S2_BAND_STD = np.array(
    [2739.5217, 2846.2993, 2690.8250, 2290.0439, 2088.8970, 2673.1106, 2381.4521, 2229.5225, 1601.0942, 1495.3545],
    dtype=np.float32,
)
S1_ASC_MEAN = np.array([5588.3291, 3025.6270], dtype=np.float32)
S1_ASC_STD = np.array([1713.4646, 1693.0471], dtype=np.float32)
S1_DESC_MEAN = np.array([5552.9683, 2955.0520], dtype=np.float32)
S1_DESC_STD = np.array([1685.5857, 1677.6414], dtype=np.float32)
S1_BAND_MEAN = (S1_ASC_MEAN + S1_DESC_MEAN) / 2
S1_BAND_STD = (S1_ASC_STD + S1_DESC_STD) / 2

OBSERVATION_BUCKETS = tuple(range(8, 257, 8))

_REPO = Path(__file__).resolve().parents[2]
_INPUT = Path(os.environ.get("ROBUSTNESS_INPUT", _REPO / "data" / "input"))
DEFAULT_WEIGHTS = _INPUT / "models" / "tessera" / "tessera_v1_1_mpc_encoder.pt"


class CustomGRUCell(nn.Module):
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.W_ir = nn.Linear(input_size, hidden_size, bias=False)
        self.W_iz = nn.Linear(input_size, hidden_size, bias=False)
        self.W_ih = nn.Linear(input_size, hidden_size, bias=False)
        self.W_hr = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W_hz = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W_hh = nn.Linear(hidden_size, hidden_size, bias=False)
        self.b_r = nn.Parameter(torch.zeros(hidden_size))
        self.b_z = nn.Parameter(torch.zeros(hidden_size))
        self.b_h = nn.Parameter(torch.zeros(hidden_size))

    def forward(self, x_t: torch.Tensor, h_prev: torch.Tensor) -> torch.Tensor:
        r_t = torch.sigmoid(self.W_ir(x_t) + self.W_hr(h_prev) + self.b_r)
        z_t = torch.sigmoid(self.W_iz(x_t) + self.W_hz(h_prev) + self.b_z)
        h_tilde = torch.tanh(self.W_ih(x_t) + self.W_hh(r_t * h_prev) + self.b_h)
        return (1 - z_t) * h_prev + z_t * h_tilde


class CustomGRU(nn.Module):
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.gru_cell = CustomGRUCell(input_size, hidden_size)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h_t = torch.zeros(x.shape[0], self.hidden_size, device=x.device, dtype=x.dtype)
        outputs = []
        for step in range(x.shape[1]):
            h_t = self.gru_cell(x[:, step], h_t)
            outputs.append(h_t)
        return torch.stack(outputs, dim=1), h_t


class CustomTemporalAwarePooling(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.temporal_context = CustomGRU(input_dim, input_dim)
        self.query = nn.Linear(input_dim, 1)
        self.layer_norm = nn.LayerNorm(input_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] == 1:
            return x[:, 0]
        context, _ = self.temporal_context(x)
        weights = torch.softmax(self.query(self.layer_norm(context)), dim=1)
        return (weights * x).sum(dim=1)


class TemporalPositionalModel(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model

    def forward(self, doy: torch.Tensor) -> torch.Tensor:
        position = doy.unsqueeze(-1).float()
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2, dtype=torch.float32, device=doy.device)
            * -(math.log(10000.0) / self.d_model)
        )
        out = torch.zeros(*doy.shape, self.d_model, device=doy.device)
        out[:, :, 0::2] = torch.sin(position * div_term)
        out[:, :, 1::2] = torch.cos(position * div_term)
        return out


class TransformerModel(nn.Module):
    def __init__(self, band_num: int):
        super().__init__()
        width = TESSERA_LATENT_DIM * 4
        self.embedding = nn.Sequential(nn.Linear(band_num, width), nn.ReLU(), nn.Linear(width, width))
        self.temporal_model = TemporalPositionalModel(width)
        layer = nn.TransformerEncoderLayer(
            d_model=width,
            nhead=4,
            dim_feedforward=2048,
            dropout=0.1,
            activation="relu",
            batch_first=True,
        )
        self.transformer_model = nn.TransformerEncoder(layer, num_layers=4)
        self.attn_pool = CustomTemporalAwarePooling(width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bands, doy = x[:, :, :-1], x[:, :, -1]
        encoded = self.embedding(bands) + self.temporal_model(doy)
        return self.attn_pool(self.transformer_model(encoded))


class TesseraV11Model(nn.Module):
    def __init__(self):
        super().__init__()
        width = TESSERA_LATENT_DIM * 4
        self.s2_backbone = TransformerModel(10)
        self.s1_backbone = TransformerModel(2)
        self.dim_reducer = nn.Sequential(
            nn.Linear(width * 2, width * 4),
            nn.LayerNorm(width * 4),
            nn.ReLU(inplace=False),
            nn.Dropout(0.2),
            nn.Linear(width * 4, TESSERA_REPRESENTATION_DIM),
        )

    def forward(self, s2_x: torch.Tensor, s1_x: torch.Tensor) -> torch.Tensor:
        return self.dim_reducer(torch.cat([self.s2_backbone(s2_x), self.s1_backbone(s1_x)], dim=-1))


def _resample_indices(valid_len: int, target_size: int) -> np.ndarray:
    if valid_len == target_size:
        return np.arange(valid_len, dtype=np.int64)
    if target_size < valid_len:
        chunks = np.array_split(np.arange(valid_len), target_size)
        return np.asarray([chunk[len(chunk) // 2] for chunk in chunks], dtype=np.int64)
    extras = np.rint(np.linspace(0, valid_len - 1, target_size - valid_len + 2)[1:-1]).astype(np.int64)
    return np.concatenate([np.arange(valid_len, dtype=np.int64), extras])


def _bucket_size(valid_len: int) -> int:
    return next((size for size in OBSERVATION_BUCKETS if valid_len <= size), OBSERVATION_BUCKETS[-1])


@dataclass
class TesseraModel:
    """Frozen TESSERA v1.1 MPC model."""

    name: str = "tessera"
    embedding_dim: int = TESSERA_EMBEDDING_DIM
    weights_path: str | Path | None = None
    device: str = "cuda" if torch.cuda.is_available() else "cpu"  # align with the other models' default
    batch_size: int = 256
    condition_invariant: bool = False
    _model: Any = field(default=None, repr=False)

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        path = Path(self.weights_path or os.environ.get("TESSERA_WEIGHTS") or DEFAULT_WEIGHTS)
        if not path.exists():
            raise FileNotFoundError(
                f"TESSERA v1.1 MPC weights not found at {path}. Download "
                "tessera_v1_1_mpc_encoder.pt and set TESSERA_WEIGHTS or weights_path."
            )
        model = TesseraV11Model()
        # weights_only=True blocks arbitrary code execution during unpickling (the checkpoint is a
        # plain dict of tensors). Gate any future full-pickle checkpoint behind a verified checksum.
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        state = checkpoint.get("model_state", checkpoint.get("model_state_dict", checkpoint))
        cleaned = {}
        for key, value in state.items():
            key = key.removeprefix("_orig_mod.")
            if key.startswith(("projector.", "segmented_matryoshka_projector.")):
                continue
            # The released v1.1 encoder checkpoint uses ``transformer_encoder`` while this
            # self-contained inference implementation names the equivalent module
            # ``transformer_model``.
            key = key.replace(".transformer_encoder.", ".transformer_model.")
            cleaned[key] = value
        missing, unexpected = model.load_state_dict(cleaned, strict=False)
        if missing or unexpected:
            raise RuntimeError(f"TESSERA v1.1 checkpoint mismatch: missing={missing}, unexpected={unexpected}")
        model.to(self.device).eval()
        for parameter in model.parameters():
            parameter.requires_grad = False
        self._model = model

    @staticmethod
    def _select_bands(modality: np.ndarray, wanted: list[str], available: list[str]) -> np.ndarray:
        columns = {band: index for index, band in enumerate(available)}
        missing = [band for band in wanted if band not in columns]
        if missing:
            raise ValueError(f"Benchmark is missing TESSERA bands: {missing}")
        return modality[:, :, [columns[band] for band in wanted]].astype(np.float32)

    @staticmethod
    def _scale_s1(values: np.ndarray) -> np.ndarray:
        finite = values[np.isfinite(values)]
        if finite.size and np.nanpercentile(np.abs(finite), 95) < 100:
            return np.clip((values + 50.0) * 200.0, 0.0, 32767.0)
        return values

    def _prepare_streams(self, bench: Benchmark) -> dict[tuple[int, int], tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Per-sample (S2, merged-S1) streams from the benchmark's NATIVE acquisition series.

        TESSERA's ``to_native``: it takes the FULL per-sample series with real day-of-year (bucketed
        up to OBSERVATION_BUCKETS) and z-scores it -- not a fixed monthly composite. S1 is used when
        the benchmark provides it, else an empty stream (S2-only benchmarks). Samples are grouped by
        (s2_len, s1_len) so each group batches together.
        """
        s2_values, s2_doy, _s2_months, s2_bands = bench.native_series("s2")
        s1_values, s1_doy, _s1_months, s1_bands = bench.native_series("s1")
        s2_cols = [s2_bands.index(b) for b in TESSERA_S2_BANDS]  # raises if a required S2 band is absent
        has_s1 = all(b in s1_bands for b in TESSERA_S1_BANDS)
        s1_cols = [s1_bands.index(b) for b in TESSERA_S1_BANDS] if has_s1 else []
        n_s1 = len(TESSERA_S1_BANDS)
        groups: dict[tuple[int, int], list[tuple[int, np.ndarray, np.ndarray]]] = {}
        for index in range(bench.n_samples):
            s2v = np.asarray(s2_values[index], dtype=np.float32)[:, s2_cols]
            s2_stream = self._stream(
                s2v, np.ones(len(s2v), np.float32), np.asarray(s2_doy[index], np.float32),
                S2_BAND_MEAN, S2_BAND_STD,
            )
            if has_s1 and len(s1_values[index]):
                s1v = self._scale_s1(np.asarray(s1_values[index], dtype=np.float32)[:, s1_cols])
                s1_stream = self._stream(
                    s1v, np.ones(len(s1v), np.float32), np.asarray(s1_doy[index], np.float32),
                    S1_BAND_MEAN, S1_BAND_STD,
                )
            else:  # benchmark has no native radar -> an all-missing S1 stream
                s1_stream = self._stream(
                    np.zeros((0, n_s1), np.float32), np.zeros(0, np.float32), np.zeros(0, np.float32),
                    S1_BAND_MEAN, S1_BAND_STD,
                )
            key = (s2_stream.shape[0], s1_stream.shape[0])
            groups.setdefault(key, []).append((index, s2_stream, s1_stream))
        return {
            key: (
                np.asarray([item[0] for item in items], dtype=np.int64),
                np.stack([item[1] for item in items]),
                np.stack([item[2] for item in items]),
            )
            for key, items in groups.items()
        }

    @staticmethod
    def _stream(values: np.ndarray, mask: np.ndarray, doy: np.ndarray, mean, std) -> np.ndarray:
        """One ``(T_bucket, C+1)`` stream: keep valid steps, resample to a bucket size, z-score, append doy."""
        valid = np.flatnonzero(np.asarray(mask) > 0)
        target = _bucket_size(max(1, len(valid)))
        if len(valid) == 0:
            return np.zeros((target, values.shape[1] + 1), dtype=np.float32)
        take = valid[_resample_indices(len(valid), target)]
        normalized = (values[take] - mean) / (std + 1e-9)
        return np.concatenate([normalized, np.asarray(doy)[take, None]], axis=1).astype(np.float32)

    def compute_macs(self) -> int:
        self._ensure_loaded()
        s2 = torch.randn(1, 16, 11, device=self.device)
        s1 = torch.randn(1, 16, 3, device=self.device)
        macs, _ = thop.profile(self._model, inputs=(s2, s1), verbose=False)
        return int(macs)

    @torch.no_grad()
    def encode(self, bench: Benchmark) -> np.ndarray:
        self._ensure_loaded()
        output = np.empty((bench.n_samples, self.embedding_dim), dtype=np.float32)
        for indices, s2, s1 in self._prepare_streams(bench).values():
            for start in range(0, len(indices), self.batch_size):
                sl = slice(start, start + self.batch_size)
                s2_tensor = torch.from_numpy(s2[sl]).to(self.device)
                s1_tensor = torch.from_numpy(s1[sl]).to(self.device)
                embedding = self._model(s2_tensor, s1_tensor)[:, : self.embedding_dim]
                output[indices[sl]] = embedding.cpu().numpy().astype(np.float32)
        return output
