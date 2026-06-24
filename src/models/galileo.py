"""Galileo v1 (NASA Harvest) frozen-model wrapper.

Galileo is a family of multimodal Vision Transformer models pretrained on Earth
observation data (S2, S1, ERA5, SRTM, etc.). We use the **Base** variant (768-dim,
12-layer model) frozen: construct a Galileo MaskedOutput from a Benchmark,
run ``model.forward(...)``, pool the tokens, and get ``(N, 768)`` embeddings.

Installation
------------
No pip package needed. We vendor ``single_file_galileo.py`` from the Galileo
repo. The project environment provides the torch/einops dependencies.

Weights are downloaded from HuggingFace ``nasaharvest/galileo`` on first use.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import thop
import torch
import torch.nn.functional as F

from utils.models.galileoutil import (
    SPACE_BAND_GROUPS_IDX,
    SPACE_BANDS,
    SPACE_TIME_BANDS,
    SPACE_TIME_BANDS_GROUPS_IDX,
    STATIC_BAND_GROUPS_IDX,
    STATIC_BANDS,
    TIME_BAND_GROUPS_IDX,
    TIME_BANDS,
    GalileoNativeModel,
)

if TYPE_CHECKING:
    from dataio.get_input import Benchmark

# --------------------------------------------------------------------------- #
# Model-size registry  (embedding_dim, hf_subdir)
# --------------------------------------------------------------------------- #
GALILEO_VARIANTS: dict[str, tuple[int, str]] = {
    "nano": (128, "models/nano"),
    "tiny": (192, "models/tiny"),
    "base": (768, "models/base"),
}

# --------------------------------------------------------------------------- #
# Galileo pretraining normalization statistics  (z-score per tensor sub-dim).
# Extracted from galileo/config/normalization.json.
# Key = dimensionality of the tensor being normalized.
# --------------------------------------------------------------------------- #
GALILEO_NORM_13_MEAN = np.array(
    [
        -11.7287,
        -18.8556,
        1395.3409,
        1338.4027,
        1343.0988,
        1543.8608,
        2186.2022,
        2525.0933,
        2410.3377,
        2750.2855,
        2234.9111,
        1474.5311,
        0.2892,
    ],
    dtype=np.float32,
)
GALILEO_NORM_13_STD = np.array(
    [
        4.8871,
        5.7303,
        917.7041,
        913.2988,
        1092.6787,
        1047.2206,
        1048.0102,
        1143.6903,
        1098.9792,
        1204.4728,
        1145.9774,
        980.2430,
        0.2721,
    ],
    dtype=np.float32,
)

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
_REPO = Path(__file__).resolve().parents[2]
_INPUT = Path(os.environ.get("ROBUSTNESS_INPUT", _REPO / "data" / "input"))
GALILEO_HF_REPO = "nasaharvest/galileo"
GALILEO_HF_REVISION = "f039dd5dde966a931baeda47eb680fa89b253e4e"  # immutable commit (reproducible weights)

# band indexing helpers
_GALILEO_S2_NAMES = ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"]
# indices of S2 bands within the 13-band SPACE_TIME_BANDS list:
_GALILEO_S2_IDX = np.array([SPACE_TIME_BANDS.index(b) for b in _GALILEO_S2_NAMES], dtype=int)
_NDVI_IDX = SPACE_TIME_BANDS.index("NDVI")
# indices of the S2 band groups in the groups mask (S2_RGB, S2_Red_Edge, ...)
_GALILEO_S2_GROUP_IDS = [i for i, k in enumerate(SPACE_TIME_BANDS_GROUPS_IDX) if k.startswith("S2") or k == "NDVI"]
# all non-S2 group IDs (S1, etc.) are masked
_GALILEO_NON_S2_GROUP_IDS = [i for i in range(len(SPACE_TIME_BANDS_GROUPS_IDX)) if i not in _GALILEO_S2_GROUP_IDS]

# Map benchmark band names -> SPACE_TIME_BANDS index (S1 + S2 + NDVI)
_BENCH_TO_GALILEO_ST_IDX: dict[str, int] = {
    "VV": 0,
    "VH": 1,
    "B2": 2,
    "B3": 3,
    "B4": 4,
    "B5": 5,
    "B6": 6,
    "B7": 7,
    "B8": 8,
    "B8A": 9,
    "B11": 10,
    "B12": 11,
    "NDVI": 12,
}


def _default_load_model(weights_path: str | Path | None, model_size: str) -> Any:
    """Download Galileo weights from HuggingFace and load the model.

    Returns the Galileo native model module.
    """
    dim, subdir = GALILEO_VARIANTS[model_size]
    if weights_path is not None:
        p = Path(weights_path).expanduser()
    else:
        p = _INPUT / "models" / "galileo" / model_size

    if not (p / "config.json").exists() or not (p / "model.pt").exists():
        from utils.ioutils import hf_download_to

        p.mkdir(parents=True, exist_ok=True)
        hf_download_to(GALILEO_HF_REPO, f"{subdir}/config.json", p / "config.json", revision=GALILEO_HF_REVISION)
        # The repo ships the encoder as encoder.pt (there is no model.pt); save it to the
        # local model.pt destination the loader (galileoutil) expects.
        hf_download_to(GALILEO_HF_REPO, f"{subdir}/encoder.pt", p / "model.pt", revision=GALILEO_HF_REVISION)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = GalileoNativeModel.load_from_folder(p, device=torch.device(device))
    model.eval()
    return model


@dataclass
class GalileoModel:
    """Frozen Galileo model: Benchmark -> (N, D) embeddings.

    ``encode`` expects a :class:`Benchmark` instance.
    """

    name: str = "galileo"
    model_size: str = "base"
    patch_size: int = 1  # spatial grid H=W=1 (point-level); patch_size=1 keeps it 1x1
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size: int = 64
    weights_path: str | Path | None = None
    load_model: Any = field(default=_default_load_model)
    _model: Any = field(default=None, repr=False)

    def __post_init__(self):
        self.embedding_dim = GALILEO_VARIANTS[self.model_size][0]

    # ---- model loading -----------------------------------------------------

    def _ensure_loaded(self) -> None:
        if self._model is None:
            self._model = self.load_model(self.weights_path, self.model_size)
            self._model.to(self.device)
            self._model.eval()

    # ---- input assembly ----------------------------------------------------

    def _bench_to_galileo(self, bench: Benchmark) -> tuple:
        """Convert a Benchmark to Galileo's expected inputs (numpy).

        Fills S1 + S2 + NDVI bands where available; marks unavailable band groups
        as masked so the model ignores them.

        Returns
        -------
        s_t_x : (N, 1, 1, T, 13) float32 — S1 + S2 + NDVI, z-score normalized
        sp_x  : (N, 1, 1, 16)     float32 — zeros
        t_x   : (N, T, 6)         float32 — zeros
        st_x  : (N, 18)           float32 — zeros
        s_t_m : (N, 1, 1, T, 7)   float32 — 0=available group, 1=masked (per-timestep s2/s1 mask)
        sp_m  : (N, 1, 1, 3)      float32 — all 1 (masked)
        t_m   : (N, T, 3)         float32 — all 1 (masked)
        st_m  : (N, 4)            float32 — all 1 (masked)
        months: (N, T)            int64   — calendar month (0-indexed) per timestep from doy
        """
        s2_vals, s2_msk, doy_arr, s2_bands = bench.monthly("s2")
        s1_vals, s1_msk, _, s1_bands = bench.monthly("s1")
        n, t = s2_vals.shape[0], s2_vals.shape[1]

        # Build per-modality band-to-column lookups (full native band sets)
        s2_idx = {b: i for i, b in enumerate(s2_bands)}
        s1_idx = {b: i for i, b in enumerate(s1_bands)}

        # -- space_time_x: (N, 1, 1, T, 13) ---------------------------------
        x_st = np.zeros((n, 1, 1, t, len(SPACE_TIME_BANDS)), dtype=np.float32)
        for bname, gidx in _BENCH_TO_GALILEO_ST_IDX.items():
            if bname in s2_idx:
                x_st[:, 0, 0, :, gidx] = s2_vals[:, :, s2_idx[bname]]
            elif bname in s1_idx:
                x_st[:, 0, 0, :, gidx] = s1_vals[:, :, s1_idx[bname]]

        # Normalize space_time_x (z-score per band)
        x_st = (x_st - GALILEO_NORM_13_MEAN) / GALILEO_NORM_13_STD

        # -- masks -----------------------------------------------------------
        # s_t_m: (N, 1, 1, T, 7) — 1 = masked, 0 = available. A group is available at a timestep
        # ONLY where that modality is actually observed there (the monthly view's per-month mask),
        # so empty months and the placeholder S1 on S2-only benchmarks stay masked.
        s_t_m = np.ones((n, 1, 1, t, len(SPACE_TIME_BANDS_GROUPS_IDX)), dtype=np.float32)
        if s2_bands:
            s2_masked = (s2_msk <= 0).astype(np.float32)  # (n, t): 1 where NOT observed
            for gid in _GALILEO_S2_GROUP_IDS:
                s_t_m[:, 0, 0, :, gid] = s2_masked
        if s1_bands:
            s_t_m[:, 0, 0, :, 0] = (s1_msk <= 0).astype(np.float32)  # S1 is the first group

        # -- empty modalities (all masked) -----------------------------------
        sp_x = np.zeros((n, 1, 1, len(SPACE_BANDS)), dtype=np.float32)
        sp_m = np.ones((n, 1, 1, len(SPACE_BAND_GROUPS_IDX)), dtype=np.float32)
        t_x = np.zeros((n, t, len(TIME_BANDS)), dtype=np.float32)
        t_m = np.ones((n, t, len(TIME_BAND_GROUPS_IDX)), dtype=np.float32)
        st_x = np.zeros((n, len(STATIC_BANDS)), dtype=np.float32)
        st_m = np.ones((n, len(STATIC_BAND_GROUPS_IDX)), dtype=np.float32)

        # -- months ----------------------------------------------------------
        # Real calendar month (0-indexed) per timestep from the monthly view's day-of-year.
        doy = np.clip(np.asarray(doy_arr, dtype=np.int64), 1, 365)
        months = (
            (np.datetime64("2001-01-01") + (doy - 1).astype("timedelta64[D]"))
            .astype("datetime64[M]")
            .astype(np.int64)
            % 12
        ).astype(np.int64)

        return x_st, sp_x, t_x, st_x, s_t_m, sp_m, t_m, st_m, months

    # ---- MAC estimate (thop) ---------------------------------------------

    def compute_macs(self) -> int:
        self._ensure_loaded()
        B, H, W, T = 1, 8, 8, 1
        dev = next(self._model.parameters()).device

        dummy_st = torch.randn(B, H, W, T, len(SPACE_TIME_BANDS), device=dev)
        dummy_sp = torch.randn(B, H, W, len(SPACE_BANDS), device=dev)
        dummy_t = torch.randn(B, T, len(TIME_BANDS), device=dev)
        dummy_static = torch.randn(B, len(STATIC_BANDS), device=dev)
        dummy_st_m = torch.zeros(B, H, W, T, len(SPACE_TIME_BANDS_GROUPS_IDX), device=dev)
        dummy_sp_m = torch.zeros(B, H, W, len(SPACE_BAND_GROUPS_IDX), device=dev)
        dummy_t_m = torch.zeros(B, T, len(TIME_BAND_GROUPS_IDX), device=dev)
        dummy_static_m = torch.zeros(B, len(STATIC_BAND_GROUPS_IDX), device=dev)
        dummy_months = torch.full((B, T), 6, dtype=torch.long, device=dev)

        model = self._model

        class _MacWrap(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.model = model

            def forward(self, st, sp, t, st_x, st_m, sp_m, t_m, st_m2, m):
                return self.model(
                    st,
                    sp,
                    t,
                    st_x,
                    st_m,
                    sp_m,
                    t_m,
                    st_m2,
                    m,
                    patch_size=H,  # use H as patch_size for 8x8 dummy (H=W=8, patch=8 -> 1x1 grid)
                )

        macs, _ = thop.profile(
            _MacWrap(),
            inputs=(
                dummy_st,
                dummy_sp,
                dummy_t,
                dummy_static,
                dummy_st_m,
                dummy_sp_m,
                dummy_t_m,
                dummy_static_m,
                dummy_months,
            ),
            verbose=False,
        )
        return int(macs)

    # ---- embedding extraction ---------------------------------------------

    @torch.no_grad()
    def encode(self, bench: Benchmark) -> np.ndarray:
        self._ensure_loaded()
        x_st, sp_x, t_x, st_x, s_t_m, sp_m, t_m, st_m, months = self._bench_to_galileo(bench)
        n = x_st.shape[0]
        out: list[np.ndarray] = []

        for start in range(0, n, self.batch_size):
            sl = slice(start, start + self.batch_size)
            # Run model
            result = self._model(
                torch.from_numpy(x_st[sl]).to(self.device),
                torch.from_numpy(sp_x[sl]).to(self.device),
                torch.from_numpy(t_x[sl]).to(self.device),
                torch.from_numpy(st_x[sl]).to(self.device),
                torch.from_numpy(s_t_m[sl]).to(self.device),
                torch.from_numpy(sp_m[sl]).to(self.device),
                torch.from_numpy(t_m[sl]).to(self.device),
                torch.from_numpy(st_m[sl]).to(self.device),
                torch.from_numpy(months[sl]).to(self.device),
                patch_size=self.patch_size,
                add_layernorm_on_exit=True,
            )
            s_t_x_out, sp_x_out, t_x_out, st_x_out, s_t_m_out, sp_m_out, t_m_out, st_m_out, _ = result
            pooled = GalileoNativeModel.average_tokens(
                s_t_x_out,
                sp_x_out,
                t_x_out,
                st_x_out,
                s_t_m_out,
                sp_m_out,
                t_m_out,
                st_m_out,
            )  # (B, D)
            out.append(pooled.detach().cpu().numpy().astype(np.float32))

        return np.concatenate(out, axis=0)

    @torch.no_grad()
    def encode_dense(self, tile) -> np.ndarray:
        """Return a per-pixel PASTIS feature map from native spatial tokens."""
        self._ensure_loaded()
        from evals.benchmarks.pastis_r import PASTIS_S1_BANDS, PASTIS_S2_BANDS, _monthly_patch

        # Galileo fuses S1 + S2 per timestep, so it composites the native tile to a common monthly
        # grid (its own temporal aggregation, matching how it treats the classification benchmarks);
        # empty months are masked. This reproduces the pre-native-refactor monthly tile bit-for-bit.
        s2_m, s2_mask = _monthly_patch(tile.s2, tile.s2_months)  # (12, 10, H, W), (12,)
        s1_m, s1_mask = _monthly_patch(tile.s1, tile.s1_months)
        height, width, timesteps = tile.height, tile.width, s2_m.shape[0]
        x = np.zeros((1, height, width, timesteps, len(SPACE_TIME_BANDS)), dtype=np.float32)
        sources = {
            **{band: s2_m[:, index] for index, band in enumerate(PASTIS_S2_BANDS)},
            **{band: s1_m[:, index] for index, band in enumerate(PASTIS_S1_BANDS)},
        }
        red, nir = sources["B4"], sources["B8"]
        sources["NDVI"] = np.divide(nir - red, nir + red, out=np.zeros_like(red), where=(nir + red) != 0)
        for band, target_index in _BENCH_TO_GALILEO_ST_IDX.items():
            if band in sources:
                x[0, :, :, :, target_index] = sources[band].transpose(1, 2, 0)
        x = (x - GALILEO_NORM_13_MEAN) / GALILEO_NORM_13_STD

        spatial = np.zeros((1, height, width, len(SPACE_BANDS)), dtype=np.float32)
        temporal = np.zeros((1, timesteps, len(TIME_BANDS)), dtype=np.float32)
        static = np.zeros((1, len(STATIC_BANDS)), dtype=np.float32)
        space_time_mask = np.ones(
            (1, height, width, timesteps, len(SPACE_TIME_BANDS_GROUPS_IDX)), dtype=np.float32
        )
        for timestep in range(timesteps):
            if s2_mask[timestep]:
                space_time_mask[:, :, :, timestep, _GALILEO_S2_GROUP_IDS] = 0.0
            if s1_mask[timestep]:
                space_time_mask[:, :, :, timestep, 0] = 0.0
        args = (
            x,
            spatial,
            temporal,
            static,
            space_time_mask,
            np.ones((1, height, width, len(SPACE_BAND_GROUPS_IDX)), dtype=np.float32),
            np.ones((1, timesteps, len(TIME_BAND_GROUPS_IDX)), dtype=np.float32),
            np.ones((1, len(STATIC_BAND_GROUPS_IDX)), dtype=np.float32),
            np.arange(timesteps, dtype=np.int64)[None],
        )
        result = self._model(
            *(torch.from_numpy(value).to(self.device) for value in args),
            patch_size=8,
            add_layernorm_on_exit=True,
        )
        tokens, token_mask = result[0], result[4]
        observed = (token_mask == 0).unsqueeze(-1)
        spatial_tokens = (tokens * observed).sum(dim=(3, 4)) / observed.sum(dim=(3, 4)).clamp_min(1)
        dense = F.interpolate(
            spatial_tokens.permute(0, 3, 1, 2),
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        )[0].permute(1, 2, 0).reshape(-1, self.embedding_dim)
        return dense[tile.valid.reshape(-1)].cpu().numpy().astype(np.float32)
