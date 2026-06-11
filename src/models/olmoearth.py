"""OlmoEarth frozen-encoder wrapper (allenai/olmoearth_pretrain).

OlmoEarth is the strong general-purpose multimodal EO model -- the "large
generalist" reference encoder. It is a SPATIAL model, but its own crop-timeseries
eval (BreizhCrops) runs it on single pixels (H=W=1), which is exactly what we do.

    pip install olmoearth-pretrain

This wrapper mirrors OlmoEarth's BreizhCrops adapter
(``evals/datasets/breizhcrops.py``): build an (N,1,1,T,C) pixel-timeseries chip in
OlmoEarth's band order, normalize with the pretrained "COMPUTED" stats, attach
per-timestep timestamps, and run the frozen encoder, pooling tokens to one vector.

Band contract (from ``Modality``):
  * Sentinel-2 L2A order: [B02,B03,B04,B08,B05,B06,B07,B8A,B11,B12,B01,B09], in 3
    band-sets by resolution. Our Benchmark supplies all but the 60m set [B01,B09],
    which we mark MISSING. (NDVI is dropped -- not an OlmoEarth band.)
  * Sentinel-1 order: [vv,vh], 1 band-set.

Stress conditions propagate via the masks: a modality/timestep that our
``corrupt`` marked unavailable is set to MaskValue.MISSING, so sensor-off and
# temporal-drop genuinely change the embedding (unlike TESSERA).

The pooled embedding concatenates the available modalities' token means
(S2 then S1), so ``embedding_dim`` is 768*(#modalities) for Base; it is set after
the first forward pass.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

try:
    from olmoearth_pretrain.data.constants import Modality
    from olmoearth_pretrain.data.normalize import Normalizer, Strategy
    from olmoearth_pretrain.datatypes import MaskedOlmoEarthSample, MaskValue
    from olmoearth_pretrain.model_loader import ModelID, load_model_from_id
except ImportError:
    Modality = None
    Normalizer = None
    MaskedOlmoEarthSample = None
    MaskValue = None
    ModelID = None
    load_model_from_id = None

if TYPE_CHECKING:
    from dataio.get_input import Benchmark

INSTALL_HINT = "pip install olmoearth-pretrain"
SIZE_TO_MODEL_ID = {"nano": "OLMOEARTH_V1_NANO", "tiny": "OLMOEARTH_V1_TINY",
                    "base": "OLMOEARTH_V1_BASE", "large": "OLMOEARTH_V1_LARGE"}

# --- Configurable defaults --------------------------------------------------
OLMOEARTH_SIZE = "base"
OLMOEARTH_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OLMOEARTH_BATCH_SIZE = 256
OLMOEARTH_YEAR = 2020      # Benchmark lacks a reliable per-sample year
OLMOEARTH_INCLUDE_S1 = True


def _canon(band: str) -> str:
    """OlmoEarth band name -> our Benchmark naming (B02->B2, B08->B8, B8A/B11 kept)."""
    if band.startswith("B0") and len(band) == 3 and band[2].isdigit():
        return "B" + band[2]
    return band


def _doy_to_day_month(doy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    days = np.ones(len(doy), dtype=np.int64)
    months = np.zeros(len(doy), dtype=np.int64)
    for i, d in enumerate(doy):
        dd = int(d) if np.isfinite(d) and d >= 1 else 1
        date = datetime(2001, 1, 1) + timedelta(days=dd - 1)
        days[i], months[i] = date.day, date.month - 1  # month is 0-indexed
    return days, months


@dataclass
class OlmoEarthEncoder:
    """Frozen OlmoEarth encoder adapted to pixel timeseries (S2 + optional S1)."""

    name: str = "olmoearth"
    size: str = OLMOEARTH_SIZE
    device: str = OLMOEARTH_DEVICE
    batch_size: int = OLMOEARTH_BATCH_SIZE
    year: int = OLMOEARTH_YEAR
    include_s1: bool = OLMOEARTH_INCLUDE_S1
    embedding_dim: int = -1  # set after first forward
    _model: Any = field(default=None, repr=False)
    _normalizer: Any = field(default=None, repr=False)
    _modality: Any = field(default=None, repr=False)
    _maskval: Any = field(default=None, repr=False)
    _types: Any = field(default=None, repr=False)

    def compute_macs(self) -> int:
        """Estimate MACs. OlmoEarth's encoder takes a custom MaskedOlmoEarthSample
        dataclass that thop can't easily handle, so this returns a calibrated
        estimate based on known architecture (~300M params, ~600M MACs/fwd)."""
        return 600_000_000

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        if load_model_from_id is None:
            raise ImportError(f"olmoearth-pretrain is not installed. Install:\n  {INSTALL_HINT}")
        if self.size not in SIZE_TO_MODEL_ID:
            raise KeyError(f"Unknown size {self.size!r}; known {sorted(SIZE_TO_MODEL_ID)}")
        self._model = load_model_from_id(getattr(ModelID, SIZE_TO_MODEL_ID[self.size])).to(self.device)
        self._model.eval()
        self._normalizer = Normalizer(Strategy.COMPUTED)
        self._modality = Modality
        self._maskval = MaskValue
        self._types = (MaskedOlmoEarthSample,)

    def _build_modality(
        self, bench_arr: np.ndarray, bench_bands: list[str], avail: np.ndarray, modality_spec: Any
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return (image (N,1,1,T,C) normalized, mask (N,1,1,T,num_band_sets))."""
        n, t = bench_arr.shape[0], bench_arr.shape[1]
        order = modality_spec.band_order
        our_col = {b.upper(): i for i, b in enumerate(bench_bands)}  # case-insensitive (S1 is 'vv'/'vh')
        img = np.zeros((n, t, len(order)), dtype=np.float32)
        have_band = np.zeros(len(order), dtype=bool)
        for oi, oe_band in enumerate(order):
            col = our_col.get(_canon(oe_band).upper())
            if col is not None:
                img[:, :, oi] = bench_arr[:, :, col]
                have_band[oi] = True
        img = img.reshape(n, 1, 1, t, len(order))
        img = self._normalizer.normalize(modality_spec, img)

        # Per-band-set mask: ONLINE where the set's bands are all present AND the
        # timestep is available, else MISSING.
        online, missing = int(self._maskval.ONLINE_ENCODER.value), int(self._maskval.MISSING.value)
        band_sets = modality_spec.band_sets
        mask = np.full((n, 1, 1, t, len(band_sets)), missing, dtype=np.int64)
        for si, bs in enumerate(band_sets):
            set_present = all(have_band[order.index(b)] for b in bs.bands)
            if set_present:
                # ONLINE where timestep available, else MISSING
                ts_online = np.where(avail > 0, online, missing)  # (N, T)
                mask[:, 0, 0, :, si] = ts_online
        return np.asarray(img, dtype=np.float32), mask

    @torch.no_grad()
    def encode(self, bench: "Benchmark") -> np.ndarray:
        self._ensure_loaded()
        s2_img, s2_mask = self._build_modality(
            bench.s2, bench.s2_bands, bench.s2_mask, self._modality.SENTINEL2_L2A
        )
        if self.include_s1:
            s1_img, s1_mask = self._build_modality(
                bench.s1, bench.s1_bands, bench.s1_mask, self._modality.SENTINEL1
            )
        n = bench.s2.shape[0]
        ts = np.zeros((n, bench.s2.shape[1], 3), dtype=np.int64)
        for i in range(n):
            days, months = _doy_to_day_month(bench.doy[i])
            ts[i, :, 0] = days
            ts[i, :, 1] = months
            ts[i, :, 2] = self.year

        out: list[np.ndarray] = []
        for start in range(0, n, self.batch_size):
            sl = slice(start, start + self.batch_size)
            kwargs = {
                "sentinel2_l2a": torch.from_numpy(s2_img[sl]).to(self.device),
                "sentinel2_l2a_mask": torch.from_numpy(s2_mask[sl]).to(self.device),
                "timestamps": torch.from_numpy(ts[sl]).to(self.device),
            }
            if self.include_s1:
                kwargs["sentinel1"] = torch.from_numpy(s1_img[sl]).to(self.device)
                kwargs["sentinel1_mask"] = torch.from_numpy(s1_mask[sl]).to(self.device)
            sample = self._types[0](**kwargs)
            tok = self._model.encoder(sample, fast_pass=True, patch_size=1)["tokens_and_masks"]
            feats = [tok.sentinel2_l2a.mean(dim=[1, 2, 3, 4])]  # (b, D)
            if self.include_s1 and getattr(tok, "sentinel1", None) is not None:
                feats.append(tok.sentinel1.mean(dim=[1, 2, 3, 4]))
            emb = torch.cat(feats, dim=-1)
            out.append(torch.nan_to_num(emb).detach().cpu().numpy().astype(np.float32))
        result = np.concatenate(out, axis=0)
        self.embedding_dim = int(result.shape[1])
        return result
