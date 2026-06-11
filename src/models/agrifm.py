"""AgriFM frozen-encoder wrapper (runs the published S2 encoder, no mmcv needed).

AgriFM (Li et al. 2025, arXiv:2505.21357) is a multi-source temporal foundation
model: per-modality 3D patch embeds (HLSL30 / Sentinel-2 / MODIS) feeding ONE shared
Video-Swin-Transformer backbone. The public code is at https://github.com/flyakon/AgriFM
and the CC0 weights (`AgriFM.pth`) are distributed via GLASS/OneDrive.

We use the Sentinel-2 branch. The official repo builds the model through OpenMMLab
registries (mmseg + compiled mmcv), which has no wheel for Python 3.13. But the model
math is pure torch + timm + einops — mmcv only enters transitively via mmseg's registry
decorators. So we import the repo's real `SwinTransformer3D` with the registries
*stubbed out*, pair it with the S2 patch embed (a single 3D conv), and load the matching
weights straight from `AgriFM.pth`. No mmcv, no fragile environment.

Architecture is read directly off the released checkpoint
(`encoder.S2_patch_emd.{weights,bias}` + shared `encoder.backbone.*`):
patch embed = Conv3d(10→128, kernel/stride (4,4,4)); backbone Swin embed_dim=128,
depths [2,2,18,2], heads [4,8,16,32], window (8,7,7) → 1024-d.

The benchmark stores pixel/field time series, not 256² chips, so each S2 series is
normalized in AgriFM's band order, resampled to 32 frames, expanded to a constant
spatial tile, run through the encoder, and globally pooled to `(N, 1024)`.
"""

from __future__ import annotations

import importlib
import importlib.util
import os
import sys
import types
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import thop
except ImportError:
    thop = None

if TYPE_CHECKING:
    from dataio.get_input import Benchmark

AGRIFM_S2_BANDS = ["B2", "B3", "B4", "B8", "B5", "B6", "B7", "B8A", "B11", "B12"]
AGRIFM_NUM_FRAMES = 32
AGRIFM_EMBEDDING_DIM = 1024
AGRIFM_PATCH_SIZE = (4, 4, 4)

AGRIFM_MEAN = np.array(
    [4179.192015478227, 4065.9106675194444, 3957.274910960156, 5207.452475253116,
     4327.12234687, 4873.16102239, 5049.1637925, 5111.07806856, 3056.86349163, 2490.9675032],
    dtype=np.float32,
)
AGRIFM_STD = np.array(
    [4041.5212325268735, 3691.003119315892, 3629.331318356375, 2973.5178530908756,
     3569.73343885, 3085.9151435, 2937.56005119, 2806.04462314, 1808.30013156, 1694.20220774],
    dtype=np.float32,
)

_REPO = Path(__file__).resolve().parents[2]
DEFAULT_REPO_PATH = _REPO.parent / "AgriFM"
DEFAULT_WEIGHTS_PATH = DEFAULT_REPO_PATH / "AgriFM.pth"

# Swin backbone kwargs, matched to the released checkpoint's tensor shapes.
_BACKBONE_KWARGS: dict[str, Any] = dict(
    pretrained=None,
    pretrained2d=False,
    patch_size=AGRIFM_PATCH_SIZE,
    embed_dim=128,
    depths=[2, 2, 18, 2],
    num_heads=[4, 8, 16, 32],
    window_size=(8, 7, 7),
    out_indices=(0, 1, 2, 3),
    mlp_ratio=4.0,
    qkv_bias=True,
    drop_path_rate=0.2,
    patch_norm=False,
    frozen_stages=-1,
    downsample_steps=((2, 2, 2), (2, 2, 2), (2, 2, 2), (2, 2, 2)),
    feature_fusion="cat",
    mean_frame_down=True,
)


def _frame_indices(length: int, target: int = AGRIFM_NUM_FRAMES) -> np.ndarray:
    if length <= 0:
        raise ValueError("AgriFM input needs at least one timestep")
    return np.rint(np.linspace(0, length - 1, target)).astype(np.int64)


class _Registry:
    """Minimal stand-in for mmseg's registries: just ``register_module()`` + ``build()``.

    Lets us import the repo's model file (whose classes are decorated with
    ``@MODELS.register_module()`` / ``@BACKBONES.register_module()``) without importing
    real mmseg, which would pull in compiled mmcv.
    """

    def __init__(self) -> None:
        self._d: dict[str, type] = {}

    def register_module(self, name=None, module=None, force=False):
        if module is not None:
            self._d[name or module.__name__] = module
            return module

        def deco(cls):
            self._d[name or cls.__name__] = cls
            return cls

        return deco

    def build(self, cfg, default_args=None):
        cfg = dict(cfg)
        obj = cfg.pop("type")
        cls = obj if isinstance(obj, type) else self._d[obj]
        for k, v in (default_args or {}).items():
            cfg.setdefault(k, v)
        return cls(**cfg)

    def get(self, name):
        return self._d.get(name)


def _load_swin_cls(repo: Path):
    """Import the repo's ``SwinTransformer3D`` with mmseg's registries stubbed out."""
    reg = _Registry()
    stubs = {
        "mmseg": {},
        "mmseg.models": {},
        "mmseg.models.builder": {"BACKBONES": reg, "MODELS": reg},
        "mmseg.registry": {"MODELS": reg, "TRANSFORMS": reg},
        "mmseg.registry.registry": {"MODELS": reg, "TRANSFORMS": reg},
    }
    saved = {name: sys.modules.get(name) for name in stubs}
    for name, attrs in stubs.items():
        mod = types.ModuleType(name)
        for k, v in attrs.items():
            setattr(mod, k, v)
        sys.modules[name] = mod
    try:
        src = repo / "AgriFM" / "models" / "video_swin_transformer.py"
        spec = importlib.util.spec_from_file_location("_agrifm_vswin", src)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.SwinTransformer3D
    finally:
        for name, prev in saved.items():
            if prev is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = prev


class _S2PatchEmbed(nn.Module):
    """AgriFM's S2 patch embed: one 3D conv stored as raw ``weights`` / ``bias``."""

    def __init__(self, in_chans: int = 10, embed_dim: int = 128, patch_size=AGRIFM_PATCH_SIZE):
        super().__init__()
        self.weights = nn.Parameter(torch.zeros(embed_dim, in_chans, *patch_size))
        self.bias = nn.Parameter(torch.zeros(embed_dim))
        self.patch_size = patch_size

    def forward(self, x):  # x: (B, T, C, H, W)
        x = x.permute(0, 2, 1, 3, 4)  # -> (B, C, T, H, W) for Conv3d
        return F.conv3d(x, self.weights, self.bias, stride=self.patch_size)


class _AgriFMS2Encoder(nn.Module):
    """S2 patch embed + shared Swin backbone → {'encoder_features', 'features_list'}."""

    def __init__(self, swin_cls):
        super().__init__()
        self.patch_emd = _S2PatchEmbed()
        self.backbone = swin_cls(**_BACKBONE_KWARGS)

    def forward(self, inputs):
        return self.backbone(self.patch_emd(inputs))


def _default_load_model(repo_path: str | Path, weights_path: str | Path, device: str) -> Any:
    repo = Path(repo_path).expanduser().resolve()
    weights = Path(weights_path).expanduser().resolve()
    if not repo.exists():
        raise FileNotFoundError(
            f"AgriFM repository not found at {repo}. Clone https://github.com/flyakon/AgriFM "
            f"and set AGRIFM_REPO, or pass repo_path=."
        )
    if not weights.exists():
        raise FileNotFoundError(
            f"AgriFM weights not found at {weights}. Download AgriFM.pth (GLASS/OneDrive) "
            f"and set AGRIFM_WEIGHTS, or pass weights_path=."
        )
    missing = [n for n in ("timm", "einops", "mmengine") if importlib.util.find_spec(n) is None]
    if missing:
        raise ImportError(f"AgriFM needs: {', '.join(missing)} (pip install timm einops mmengine).")

    swin_cls = _load_swin_cls(repo)
    model = _AgriFMS2Encoder(swin_cls)

    ck = torch.load(weights, map_location="cpu")
    ck = ck.get("state_dict", ck) if isinstance(ck, dict) else ck
    patch_sd, back_sd = {}, {}
    for k, v in ck.items():
        if k.startswith("encoder.S2_patch_emd."):
            patch_sd[k[len("encoder.S2_patch_emd."):]] = v
        elif k.startswith("encoder.backbone."):
            back_sd[k[len("encoder.backbone."):]] = v
    model.patch_emd.load_state_dict(patch_sd, strict=True)
    miss, unexp = model.backbone.load_state_dict(back_sd, strict=False)
    # buffers (relative_position_index, attn_mask) are recomputed at init and may be absent
    # from the checkpoint; any other missing/unexpected key means a real architecture mismatch.
    real_missing = [m for m in miss if "relative_position_index" not in m and "attn_mask" not in m]
    if real_missing or unexp:
        raise RuntimeError(f"AgriFM backbone weight mismatch: missing={real_missing[:4]} unexpected={list(unexp)[:4]}")

    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


@dataclass
class AgriFMEncoder:
    """Frozen AgriFM S2 encoder adapted to benchmark time series."""

    name: str = "agrifm"
    embedding_dim: int = AGRIFM_EMBEDDING_DIM
    repo_path: str | Path | None = None
    weights_path: str | Path | None = None
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size: int = 8
    tile_size: int = 32
    num_frames: int = AGRIFM_NUM_FRAMES
    load_model: Callable[[str | Path, str | Path, str], Any] = field(default=_default_load_model)
    _model: Any = field(default=None, repr=False)

    def _repo_path(self) -> Path:
        return Path(self.repo_path or os.environ.get("AGRIFM_REPO") or DEFAULT_REPO_PATH)

    def _weights_path(self) -> Path:
        return Path(self.weights_path or os.environ.get("AGRIFM_WEIGHTS") or DEFAULT_WEIGHTS_PATH)

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        self._model = self.load_model(self._repo_path(), self._weights_path(), self.device)
        self._model.eval()

    def to_agrifm_series(self, bench: "Benchmark") -> np.ndarray:
        """Return normalized S2 series in AgriFM band order as (N, 32, 10)."""
        col = {band: idx for idx, band in enumerate(bench.s2_bands)}
        missing = [band for band in AGRIFM_S2_BANDS if band not in col]
        if missing:
            raise KeyError(f"Benchmark is missing AgriFM Sentinel-2 bands: {missing}")

        idx = [col[band] for band in AGRIFM_S2_BANDS]
        frame_idx = _frame_indices(bench.s2.shape[1], self.num_frames)
        x = bench.s2[:, frame_idx, :][:, :, idx].astype(np.float32)
        mask = bench.s2_mask[:, frame_idx].astype(np.float32)

        x = (x - AGRIFM_MEAN) / (AGRIFM_STD + 1e-9)
        x *= mask[:, :, None]
        return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    def compute_macs(self) -> int:
        self._ensure_loaded()
        if thop is None:
            return 0
        dummy = torch.zeros(1, self.num_frames, len(AGRIFM_S2_BANDS), self.tile_size, self.tile_size, device=self.device)
        try:
            macs, _ = thop.profile(self._model, inputs=(dummy,), verbose=False)
            return int(macs)
        except Exception:
            return 0

    @torch.no_grad()
    def encode(self, bench: "Benchmark") -> np.ndarray:
        self._ensure_loaded()
        series = self.to_agrifm_series(bench)
        out: list[np.ndarray] = []
        for start in range(0, series.shape[0], self.batch_size):
            sl = slice(start, start + self.batch_size)
            batch = torch.from_numpy(series[sl]).to(self.device)
            batch = batch[:, :, :, None, None].expand(-1, -1, -1, self.tile_size, self.tile_size).contiguous()
            features = self._model(batch)["encoder_features"]
            emb = features.mean(dim=tuple(range(2, features.ndim)))
            out.append(torch.nan_to_num(emb).detach().cpu().numpy().astype(np.float32))
        result = np.concatenate(out, axis=0)
        self.embedding_dim = int(result.shape[1])
        return result
