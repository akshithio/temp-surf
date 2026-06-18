"""OlmoEarth v1.1-Base frozen-encoder wrapper.

OlmoEarth (Ai2) is a ViT-Base (114M params) Earth observation foundation model
trained on Sentinel-2 L2A, Sentinel-1, and Landsat spatial chips. We use it
frozen: construct a :class:`MaskedOlmoEarthSample` from a :class:`Benchmark`,
run ``model.encoder``, pool the spatial-temporal feature map, and get
``(N, 768)`` embeddings.

Authoritative input contract (from olmoearth_pretrain docs):
  * ``sentinel2_l2a`` : (B, H, W, T, C=12) float32, band order = OLMOEARTH_S2_BANDS
  * ``sentinel2_l2a_mask`` : (B, H, W, T, S=1) float32, ONLINE_ENCODER value
  * ``timestamps``   : (B, T, 3) long, [day, month-0-indexed, year]
  * model.encoder(sample, fast_pass=True, patch_size=4)["tokens_and_masks"].sentinel2_l2a
    -> (B, H', W', T, S, D=768)
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

if TYPE_CHECKING:
    from dataio.get_input import Benchmark

# --------------------------------------------------------------------------- #
# OlmoEarth S2 band order (12 bands, from config.json tokenization config).
# The model was pre-trained on S2 L2A with these bands in this exact order.
# --------------------------------------------------------------------------- #
OLMOEARTH_S2_BANDS = [
    "B02",
    "B03",
    "B04",
    "B08",
    "B05",
    "B06",
    "B07",
    "B8A",
    "B11",
    "B12",
    "B01",
    "B09",
]
OLMOEARTH_NUM_S2_BANDS = len(OLMOEARTH_S2_BANDS)  # 12
OLMOEARTH_EMBEDDING_DIM = 768

# --------------------------------------------------------------------------- #
# Mapping from common Benchmark S2 band names to OlmoEarth's index.
# Benchmark bands (CropHarvest / EuroCropsML) use the nasaharvest ordering:
#   ["B2","B3","B4","B5","B6","B7","B8","B8A","B11","B12","NDVI"]
# OlmoEarth expects the raw L2A spectral bands (minus NDVI, plus B01/B09).
# Bands present in both are mapped; missing bands are zero-filled after normalization.
# --------------------------------------------------------------------------- #
_BENCH_TO_OLMOEARTH_IDX: dict[str, int] = {
    "B2": 0,
    "B3": 1,
    "B4": 2,
    "B8": 3,
    "B5": 4,
    "B6": 5,
    "B7": 6,
    "B8A": 7,
    "B11": 8,
    "B12": 9,
    # B01 -> 10, B09 -> 11: not present in CropHarvest / EuroCropsML Benchmark arrays
}

# --------------------------------------------------------------------------- #
# Weight paths
# --------------------------------------------------------------------------- #
_REPO = Path(__file__).resolve().parents[2]
_INPUT = Path(os.environ.get("ROBUSTNESS_INPUT", _REPO / "data" / "input"))
OLMOEARTH_HF_REPO = "allenai/OlmoEarth-v1_1-Base"
DEFAULT_OLMOEARTH_WEIGHTS = _INPUT / "models" / "olmoearth-v1_1-base"


def _default_load_model(weights_path: str | Path | None = None) -> Any:
    """Load the full OlmoEarth v1.1-Base model from Hugging Face.

    Returns the full model (encoder + decoder).  Callers access ``model.encoder``.
    """
    from huggingface_hub import snapshot_download
    from olmoearth_pretrain.model_loader import load_model_from_path

    model_dir = Path(weights_path or DEFAULT_OLMOEARTH_WEIGHTS).expanduser()
    if model_dir.is_file():
        model_dir = model_dir.parent
    if not (model_dir / "config.json").exists() or not (model_dir / "weights.pth").exists():
        snapshot_download(
            repo_id=OLMOEARTH_HF_REPO,
            local_dir=model_dir,
            allow_patterns=["config.json", "weights.pth"],
        )
    model = load_model_from_path(model_dir)
    model.eval()
    return model


@dataclass
class OlmoEarthEncoder:
    """Frozen OlmoEarth v1.1-Base encoder: Benchmark -> (N, 768) embeddings.

    ``encode`` expects an already-degraded Benchmark (apply ``degrade`` upstream);
    this class is condition-agnostic, exactly like every other encoder.
    """

    name: str = "olmoearth"
    embedding_dim: int = OLMOEARTH_EMBEDDING_DIM
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size: int = 8
    patch_size: int = 1
    tile_size: int = 1
    weights_path: str | Path | None = None
    load_model: Any = field(default=_default_load_model)
    _model: Any = field(default=None, repr=False)

    # ---- model loading -----------------------------------------------------

    def _ensure_loaded(self) -> None:
        if self._model is None:
            self._model = self.load_model(self.weights_path)
            self._model.to(self.device)
            self._model.eval()

    # ---- input assembly ----------------------------------------------------

    def _bench_to_olmoearth(
        self,
        bench: Benchmark,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Convert a Benchmark to OlmoEarth's expected numpy inputs.

        Returns
        -------
        images : (N, S, S, T, 12) float32
            Point-level series broadcast to the configured spatial tile size.
        masks   : (N, S, S, T, B) float32
            Availability mask filled with ONLINE_ENCODER value where data exists.
        timestamps : (N, T, 3) int64
            [day, month, year] for every observation.
        """
        from olmoearth_pretrain.datatypes import MaskValue

        n, t = bench.s2.shape[0], bench.s2.shape[1]
        bench_s2_bands: list[str] = bench.s2_bands
        bench_idx = {b: i for i, b in enumerate(bench_s2_bands)}

        # Build (N, T, 12) — map available bands, zero-fill missing ones
        x = np.zeros((n, t, OLMOEARTH_NUM_S2_BANDS), dtype=np.float32)
        mapped = np.zeros(OLMOEARTH_NUM_S2_BANDS, dtype=bool)
        for oe_band, oe_idx in _BENCH_TO_OLMOEARTH_IDX.items():
            if oe_band in bench_idx:
                x[:, :, oe_idx] = bench.s2[:, :, bench_idx[oe_band]]
                mapped[oe_idx] = True

        # Apply OlmoEarth's normalizer (COMPUTED strategy from pretrain stats)
        from olmoearth_pretrain.data.constants import Modality
        from olmoearth_pretrain.data.normalize import Normalizer, Strategy

        normalizer = Normalizer(Strategy.COMPUTED)
        x = normalizer.normalize(Modality.SENTINEL2_L2A, x).astype(np.float32)
        x[:, :, ~mapped] = 0.0
        x *= np.asarray(bench.s2_mask, dtype=np.float32)[:, :, None]

        # Tile the pixel/parcel series to a constant SxS spatial chip (a spatial ViT cannot
        # ingest a 1x1 chip); the encoder then patches/pools over a valid spatial grid.
        s = max(int(self.tile_size), int(self.patch_size))
        images = np.broadcast_to(x[:, None, None, :, :], (n, s, s, t, OLMOEARTH_NUM_S2_BANDS)).copy()

        num_band_sets = 1
        if self._model is not None:
            tokenization = getattr(self._model.encoder, "tokenization_config", None)
            if tokenization is not None:
                num_band_sets = tokenization.get_num_bandsets("sentinel2_l2a")
        observed = np.asarray(bench.s2_mask, dtype=bool)[:, None, None, :, None]
        avail = np.where(
            observed,
            float(MaskValue.ONLINE_ENCODER.value),
            float(MaskValue.MISSING.value),
        )
        avail = np.broadcast_to(avail, (n, s, s, t, num_band_sets)).copy().astype(np.float32)

        # Timestamps: one date per observation; use benchmark years where available.
        if bench.years is not None:
            years = bench.years
        else:
            years = np.full(n, 2021, dtype=np.int64)
        doy = np.clip(np.asarray(bench.doy, dtype=np.int64), 1, 365)
        reference = np.datetime64("2001-01-01") + (doy - 1).astype("timedelta64[D]")
        months = reference.astype("datetime64[M]").astype(np.int64) % 12
        month_start = reference.astype("datetime64[M]").astype("datetime64[D]")
        days = (reference - month_start).astype(np.int64) + 1
        timestamps = np.empty((n, t, 3), dtype=np.int64)
        timestamps[:, :, 0] = days
        timestamps[:, :, 1] = months
        timestamps[:, :, 2] = np.asarray(years, dtype=np.int64)[:, None]

        return images, avail, timestamps

    # ---- MAC estimate (thop) ---------------------------------------------

    def compute_macs(self) -> int:
        self._ensure_loaded()
        from olmoearth_pretrain.data.constants import Modality
        from olmoearth_pretrain.data.normalize import Normalizer, Strategy
        from olmoearth_pretrain.datatypes import MaskedOlmoEarthSample, MaskValue

        B, H, W, T = 1, 1, 1, 12
        dev = next(self._model.parameters()).device
        normalizer = Normalizer(Strategy.COMPUTED)
        dummy_bands = normalizer.normalize(
            Modality.SENTINEL2_L2A,
            np.random.randn(B, H, W, T, OLMOEARTH_NUM_S2_BANDS).astype(np.float32),
        )
        sample = MaskedOlmoEarthSample(
            sentinel2_l2a=torch.from_numpy(dummy_bands).to(dev),
            sentinel2_l2a_mask=torch.full(
                (B, H, W, T, 1),
                MaskValue.ONLINE_ENCODER.value,
                dtype=torch.float32,
                device=dev,
            ),
            timestamps=torch.tensor([15, 0, 2021], device=dev).reshape(B, 1, 3).expand(B, T, 3),
        )

        encoder = self._model.encoder

        class _MacWrap(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder = encoder

            def forward(self, s):
                return self.encoder(s, fast_pass=True, patch_size=1)["tokens_and_masks"].sentinel2_l2a

        macs, _ = thop.profile(_MacWrap(), inputs=(sample,), verbose=False)
        return int(macs)

    # ---- embedding extraction ---------------------------------------------

    @torch.no_grad()
    def encode(self, bench: Benchmark) -> np.ndarray:
        self._ensure_loaded()
        images, masks, ts = self._bench_to_olmoearth(bench)
        n = images.shape[0]
        out: list[np.ndarray] = []

        eff_patch = max(1, min(self.patch_size, int(images.shape[1]), int(images.shape[2])))
        for start in range(0, n, self.batch_size):
            sl = slice(start, start + self.batch_size)
            from olmoearth_pretrain.datatypes import MaskedOlmoEarthSample

            sample = MaskedOlmoEarthSample(
                sentinel2_l2a=torch.from_numpy(images[sl]).to(self.device),
                sentinel2_l2a_mask=torch.from_numpy(masks[sl]).to(self.device),
                timestamps=torch.from_numpy(ts[sl]).to(self.device),
            )
            from olmoearth_pretrain.datatypes import MaskValue
            from olmoearth_pretrain.nn.pooling import PoolingType, pool_unmasked_tokens

            fast_pass = not (sample.sentinel2_l2a_mask == MaskValue.MISSING.value).any().item()
            tokens = self._model.encoder(sample, fast_pass=fast_pass, patch_size=eff_patch)["tokens_and_masks"]
            pooled = pool_unmasked_tokens(tokens, PoolingType.MEAN)
            out.append(pooled.detach().cpu().numpy().astype(np.float32))

        return np.concatenate(out, axis=0)

    @torch.no_grad()
    def encode_dense(self, tile) -> np.ndarray:
        """Return per-pixel features from OlmoEarth's native spatial token grid."""
        self._ensure_loaded()
        from olmoearth_pretrain.data.constants import Modality
        from olmoearth_pretrain.data.normalize import Normalizer, Strategy
        from olmoearth_pretrain.datatypes import MaskedOlmoEarthSample, MaskValue

        from dataio.get_input import PASTIS_S2_BANDS

        height, width, timesteps = tile.height, tile.width, tile.s2.shape[0]
        values = np.zeros((1, height, width, timesteps, OLMOEARTH_NUM_S2_BANDS), dtype=np.float32)
        columns = {band: index for index, band in enumerate(PASTIS_S2_BANDS)}
        mapped = np.zeros(OLMOEARTH_NUM_S2_BANDS, dtype=bool)
        for band, target_index in _BENCH_TO_OLMOEARTH_IDX.items():
            if band in columns:
                values[0, :, :, :, target_index] = tile.s2[:, columns[band]].transpose(1, 2, 0)
                mapped[target_index] = True
        values = Normalizer(Strategy.COMPUTED).normalize(Modality.SENTINEL2_L2A, values).astype(np.float32)
        values[:, :, :, :, ~mapped] = 0.0
        values *= tile.s2_mask[None, None, None, :, None]

        tokenization = self._model.encoder.tokenization_config
        band_sets = tokenization.get_num_bandsets("sentinel2_l2a")
        mask = np.where(
            tile.s2_mask[None, None, None, :, None] > 0,
            float(MaskValue.ONLINE_ENCODER.value),
            float(MaskValue.MISSING.value),
        )
        mask = np.broadcast_to(mask, (1, height, width, timesteps, band_sets)).copy().astype(np.float32)
        timestamps = np.zeros((1, timesteps, 3), dtype=np.int64)
        timestamps[0, :, 0] = 15
        timestamps[0, :, 1] = np.arange(timesteps)
        timestamps[0, :, 2] = 2019
        sample = MaskedOlmoEarthSample(
            sentinel2_l2a=torch.from_numpy(values).to(self.device),
            sentinel2_l2a_mask=torch.from_numpy(mask).to(self.device),
            timestamps=torch.from_numpy(timestamps).to(self.device),
        )
        output = self._model.encoder(sample, fast_pass=False, patch_size=8)["tokens_and_masks"]
        tokens = output.sentinel2_l2a
        observed = (output.sentinel2_l2a_mask != MaskValue.MISSING.value).unsqueeze(-1)
        spatial_tokens = (tokens * observed).sum(dim=(3, 4)) / observed.sum(dim=(3, 4)).clamp_min(1)
        dense = F.interpolate(
            spatial_tokens.permute(0, 3, 1, 2),
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        )[0].permute(1, 2, 0).reshape(-1, self.embedding_dim)
        return dense[tile.valid.reshape(-1)].cpu().numpy().astype(np.float32)
