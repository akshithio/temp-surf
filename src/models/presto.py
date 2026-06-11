"""Presto frozen-encoder wrapper.

Presto (Tseng et al., "Lightweight, Pre-trained Transformers for Remote Sensing
Timeseries") is the lightweight pixel-timeseries baseline -- the model WorldCereal
actually deploys. We use it frozen: corrupt a :class:`Benchmark`, run it through
``PrestoEncoder.encode``, get ``(N, 128)`` embeddings, and feed those to the
robustness methods + probe like any other encoder.

Install (not on PyPI)::

    pip install git+https://github.com/nasaharvest/presto.git

Authoritative input contract (from Presto's single_file_presto.py):
  * ``x``            : (B, T, 17) float, band order = PRESTO_BANDS below
  * ``dynamic_world``: (B, T) long, value 9 == "missing" (we always pass missing)
  * ``latlons``      : (B, 2) float, [lat, lon]
  * ``mask``         : (B, T, 17) float, 1 == band missing at that timestep
  * ``month``        : (B,) long start month (0-11), Presto cycles it over T
  * encoder(..., eval_task=True) -> (B, 128) = norm(mean over time tokens)

Our Benchmark provides all 17 bands except ``slope`` (SRTM idx 15), which is
always marked missing in the mask. Two pieces live only in the full Presto
package and so are exposed as overridable hooks here (with safe defaults +
warnings), because they cannot be verified offline:
  * ``normalizer``  : per-band (mean, std) Presto expects on inputs.
  * ``load_model``  : how pretrained weights are obtained.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Any, Callable

import numpy as np
import thop
import torch

try:
    from presto import Presto
except ImportError:
    Presto = None  # type: ignore

if TYPE_CHECKING:
    from dataio.get_input import Benchmark

# Presto's 17-band order and groups (verbatim from single_file_presto.py).
PRESTO_BANDS = [
    "VV", "VH",                       # S1            [0, 1]
    "B2", "B3", "B4",                 # S2_RGB        [2, 3, 4]
    "B5", "B6", "B7",                 # S2_Red_Edge   [5, 6, 7]
    "B8",                             # S2_NIR_10m    [8]
    "B8A",                            # S2_NIR_20m    [9]
    "B11", "B12",                     # S2_SWIR       [10, 11]
    "temperature", "precipitation",   # ERA5          [12, 13]
    "elevation", "slope",             # SRTM          [14, 15]
    "NDVI",                           # NDVI          [16]
]
PRESTO_NUM_BANDS = len(PRESTO_BANDS)  # 17
DYNAMIC_WORLD_MISSING = 9
PRESTO_EMBEDDING_DIM = 128

# --- Configurable defaults --------------------------------------------------
PRESTO_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PRESTO_BATCH_SIZE = 2048
PRESTO_WEIGHTS_PATH = None
PRESTO_NORMALIZER = None  # None = use built-in PRESTO_ADD / PRESTO_DIVIDE

# Which Benchmark modality supplies each Presto band (None = never available).
_BAND_MODALITY: dict[str, str | None] = {
    "VV": "s1", "VH": "s1",
    "B2": "s2", "B3": "s2", "B4": "s2", "B5": "s2", "B6": "s2", "B7": "s2",
    "B8": "s2", "B8A": "s2", "B11": "s2", "B12": "s2", "NDVI": "s2",
    "temperature": "climate", "precipitation": "climate", "elevation": "climate",
    "slope": None,
}

# Presto's fixed per-band normalization in NORMED_BANDS order: norm = (x + ADD) / DIVIDE.
# Verbatim from presto/dataops/pipelines/s1_s2_era5_srtm.py (SHIFT/DIV values):
#   S1 (VV,VH): +25 / 25 ; S2 spectral: 0 / 1e4 ; temperature: -272.15 / 35 ;
#   precipitation: 0 / 0.03 ; elevation: 0 / 2000 ; slope: 0 / 50 ; NDVI: passthrough.
PRESTO_ADD = np.array(
    [25.0, 25.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -272.15, 0.0, 0.0, 0.0, 0.0], dtype=np.float32
)
PRESTO_DIVIDE = np.array(
    [25.0, 25.0, 1e4, 1e4, 1e4, 1e4, 1e4, 1e4, 1e4, 1e4, 1e4, 1e4, 35.0, 0.03, 2000.0, 50.0, 1.0],
    dtype=np.float32,
)


def _doy_to_month(doy: np.ndarray) -> np.ndarray:
    """Map day-of-year (1-366) to a 0-11 month index."""
    out = np.zeros(len(doy), dtype=np.int64)
    for i, d in enumerate(doy):
        day = int(d) if np.isfinite(d) and d >= 1 else 1
        out[i] = (datetime(2001, 1, 1) + timedelta(days=day - 1)).month - 1
    return out


def _default_load_model(weights_path: str | None) -> Any:
    """Load a pretrained Presto and return its encoder. Override via PrestoEncoder.load_model."""
    if Presto is None:
        raise ImportError(
            "Presto is not installed. Install it with:\n"
            "  pip install git+https://github.com/nasaharvest/presto.git"
        )
    model = Presto.load_pretrained() if weights_path is None else Presto.load_pretrained(weights_path)
    model.eval()
    return model.encoder


@dataclass
class PrestoEncoder:
    """Frozen Presto encoder: Benchmark -> (N, 128) embeddings.

    ``encode`` expects an already-corrupted Benchmark (apply ``corrupt`` upstream);
    this class is condition-agnostic, exactly like every other encoder.
    """

    name: str = "presto"
    embedding_dim: int = PRESTO_EMBEDDING_DIM
    device: str = PRESTO_DEVICE
    batch_size: int = PRESTO_BATCH_SIZE
    weights_path: str | None = PRESTO_WEIGHTS_PATH
    normalizer: tuple[np.ndarray, np.ndarray] | None = PRESTO_NORMALIZER
    load_model: Callable[[str | None], Any] = field(default=_default_load_model)
    _encoder: Any = field(default=None, repr=False)

    # ---- model loading -----------------------------------------------------
    def _ensure_loaded(self) -> None:
        if self._encoder is None:
            self._encoder = self.load_model(self.weights_path)
            self._encoder.to(self.device)
            self._encoder.eval()

    # ---- input assembly (verifiable without Presto) ------------------------
    def to_presto_inputs(
        self, bench: "Benchmark"
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Assemble (x, mask, dynamic_world, latlons, months) as numpy arrays.

        x:    (N, T, 17) normalized; mask: (N, T, 17) 1==missing;
        dynamic_world: (N, T) all-missing; latlons: (N, 2); months: (N,).
        """
        n, t = bench.s2.shape[0], bench.s2.shape[1]
        x = np.zeros((n, t, PRESTO_NUM_BANDS), dtype=np.float32)
        mask = np.ones((n, t, PRESTO_NUM_BANDS), dtype=np.float32)  # default missing

        # Per-modality (name -> column) lookup and per-timestep availability.
        cols = {
            "s2": {b: i for i, b in enumerate(bench.s2_bands)},
            "s1": {b: i for i, b in enumerate(bench.s1_bands)},
            "climate": {b: i for i, b in enumerate(bench.climate_bands)},
        }
        arrays = {"s2": bench.s2, "s1": bench.s1, "climate": bench.climate}
        avail = {"s2": bench.s2_mask, "s1": bench.s1_mask, "climate": bench.climate_mask}

        for b_idx, band in enumerate(PRESTO_BANDS):
            modality = _BAND_MODALITY[band]
            if modality is None or band not in cols[modality]:
                continue  # band never available (e.g. slope) -> stays masked
            x[:, :, b_idx] = arrays[modality][:, :, cols[modality][band]]
            mask[:, :, b_idx] = 1.0 - avail[modality]  # 1 where timestep unavailable

        add, divide = self.normalizer if self.normalizer is not None else (PRESTO_ADD, PRESTO_DIVIDE)
        x = (x + add.astype(np.float32)) / divide.astype(np.float32)
        # NDVI (idx 16) is recomputed from normalized B8 (idx 8) and B4 (idx 4), exactly as
        # Presto's normalize() does -- not taken from any precomputed NDVI column.
        b8, b4 = x[:, :, 8], x[:, :, 4]
        denom = b8 + b4
        x[:, :, 16] = np.where(denom > 0, (b8 - b4) / np.where(denom > 0, denom, 1.0), 0.0)
        x = np.where(mask > 0, 0.0, x).astype(np.float32)  # zero out masked entries

        dynamic_world = np.full((n, t), DYNAMIC_WORLD_MISSING, dtype=np.int64)
        latlons = np.nan_to_num(bench.latlon.astype(np.float32), nan=0.0)
        months = _doy_to_month(bench.doy[:, 0])
        return x, mask, dynamic_world, latlons, months

    # ---- MAC estimate (thop) ---------------------------------------------
    def compute_macs(self) -> int:
        self._ensure_loaded()
        B, T = 1, 12
        dummy = {
            "x": torch.randn(B, T, PRESTO_NUM_BANDS),
            "dynamic_world": torch.full((B, T), DYNAMIC_WORLD_MISSING, dtype=torch.long),
            "latlons": torch.randn(B, 2),
            "mask": torch.randn(B, T, PRESTO_NUM_BANDS),
            "month": torch.zeros(B, dtype=torch.long),
            "eval_task": True,
        }
        macs, _ = thop.profile(self._encoder, inputs=(dummy["x"],), kwargs=dummy, verbose=False)
        return int(macs)

    # ---- embedding extraction ---------------------------------------------
    @torch.no_grad()
    def encode(self, bench: "Benchmark") -> np.ndarray:
        self._ensure_loaded()
        x, mask, dw, latlons, months = self.to_presto_inputs(bench)
        n = x.shape[0]
        out: list[np.ndarray] = []
        for start in range(0, n, self.batch_size):
            sl = slice(start, start + self.batch_size)
            emb = self._encoder(
                x=torch.from_numpy(x[sl]).to(self.device),
                dynamic_world=torch.from_numpy(dw[sl]).to(self.device),
                latlons=torch.from_numpy(latlons[sl]).to(self.device),
                mask=torch.from_numpy(mask[sl]).to(self.device),
                month=torch.from_numpy(months[sl]).to(self.device),
                eval_task=True,
            )
            out.append(emb.detach().cpu().numpy().astype(np.float32))
        return np.concatenate(out, axis=0)
