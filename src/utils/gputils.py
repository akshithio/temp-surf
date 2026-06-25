"""GPU work-splitting helpers used by ``src/main.py``."""

from __future__ import annotations

import collections.abc
import itertools
import os
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from timm.models.layers import DropPath
from torch import Tensor, vmap
from torch.jit import Final

REPO = Path(__file__).resolve().parents[2]


SHARD_ENV = "RB_SHARD"
NUM_SHARDS_ENV = "RB_NUM_SHARDS"
FORCED_NUM_SHARDS: int | None = None


def get_2d_sincos_pos_embed_with_resolution(
    embed_dim, grid_size, res, cls_token=False, device="cpu"
):
    """
    grid_size: int of the grid height and width
    res: array of size n, representing the resolution of a pixel (say, in meters),
    return:
    pos_embed: [n,grid_size*grid_size, embed_dim] or [n,1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    res = res.to(device)
    grid_h = torch.arange(grid_size, device=device)
    grid_w = torch.arange(grid_size, device=device)
    grid = torch.meshgrid(
        grid_w, grid_h, indexing="xy"
    )  # here h goes first,direction reversed for numpy
    grid = torch.stack(grid, dim=0)  # 2 x h x w

    # grid = grid.reshape([2, 1, grid_size, grid_size])
    grid = torch.einsum("chw,n->cnhw", grid, res)  # 2 x n x h x w
    _, n, h, w = grid.shape
    pos_embed = get_2d_sincos_pos_embed_from_grid_torch(embed_dim, grid)  #  # (nxH*W, D/2)
    pos_embed = pos_embed.reshape(n, h * w, embed_dim)
    if cls_token:
        pos_embed = torch.cat(
            [
                torch.zeros([n, 1, embed_dim], device=pos_embed.device),
                pos_embed,
            ],
            dim=1,
        )
    return pos_embed


def get_2d_sincos_pos_embed_from_grid_torch(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid_torch(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid_torch(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = torch.cat([emb_h, emb_w], dim=1)  # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid_torch(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = torch.arange(embed_dim // 2, device=pos.device) / embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = torch.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = torch.sin(out)  # (M, D/2)
    emb_cos = torch.cos(out)  # (M, D/2)

    emb = torch.cat([emb_sin, emb_cos], dim=1)  # (M, D)
    return emb


def get_month_encoding_table(embed_dim):
    """Sinusoid month encoding table, for 12 months indexed from 0-11"""
    assert embed_dim % 2 == 0
    angles = torch.arange(0, 13) / (12 / (2 * np.pi))

    sin_table = torch.sin(torch.stack([angles for _ in range(embed_dim // 2)], axis=-1))
    cos_table = torch.cos(torch.stack([angles for _ in range(embed_dim // 2)], axis=-1))
    month_table = torch.concatenate([sin_table[:-1], cos_table[:-1]], axis=-1)

    return month_table  # (M, D)


def to_2tuple(x: Any) -> tuple:
    if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
        return tuple(x)
    return tuple(itertools.repeat(x, 2))


class FlexiPatchEmbed(nn.Module):
    def __init__(
        self,
        patch_size: int | tuple[int, int],
        in_chans: int = 3,
        embed_dim: int = 128,
        norm_layer: nn.Module | None = None,
        bias: bool = True,
        patch_size_seq: Sequence[int] = (1, 2, 3, 4, 5, 6),
        interpolation: str = "bicubic",
        antialias: bool = True,
    ) -> None:
        super().__init__()

        self.patch_size = to_2tuple(patch_size)

        self.proj = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            bias=bias,
        )
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

        self.interpolation = interpolation
        self.antialias = antialias
        self.patch_size_seq = patch_size_seq
        self.pinvs = self._cache_pinvs()

    def _cache_pinvs(self) -> dict:
        pinvs = {}
        for ps in self.patch_size_seq:
            tuple_ps = to_2tuple(ps)
            pinvs[tuple_ps] = self._calculate_pinv(self.patch_size, tuple_ps)
        return pinvs

    def _resize(self, x: Tensor, shape: tuple[int, int]) -> Tensor:
        x_resized = F.interpolate(
            x[None, None, ...],
            shape,
            mode=self.interpolation,
            antialias=self.antialias,
        )
        return x_resized[0, 0, ...]

    def _calculate_pinv(self, old_shape: tuple[int, int], new_shape: tuple[int, int]) -> Tensor:
        mat = []
        for i in range(np.prod(old_shape)):
            basis_vec = torch.zeros(old_shape)
            basis_vec[np.unravel_index(i, old_shape)] = 1.0
            mat.append(self._resize(basis_vec, new_shape).reshape(-1))
        resize_matrix = torch.stack(mat)
        return torch.linalg.pinv(resize_matrix)

    def resize_patch_embed(self, patch_embed: Tensor, new_patch_size: tuple[int, int]):
        if self.patch_size == new_patch_size:
            return patch_embed
        if new_patch_size not in self.pinvs:
            self.pinvs[new_patch_size] = self._calculate_pinv(self.patch_size, new_patch_size)
        pinv = self.pinvs[new_patch_size]
        pinv = pinv.to(patch_embed.device)

        def resample_patch_embed(patch_embed: Tensor):
            h, w = new_patch_size
            resampled_kernel = pinv @ patch_embed.reshape(-1)
            return rearrange(resampled_kernel, "(h w) -> h w", h=h, w=w)

        v_resample_patch_embed = vmap(vmap(resample_patch_embed, 0, 0), 1, 1)
        return v_resample_patch_embed(patch_embed)

    def forward(
        self,
        x: Tensor,
        patch_size: int | tuple[int, int] | None = None,
    ) -> Tensor | tuple[Tensor, tuple[int, int]]:
        batch_size = x.shape[0]
        has_time_dimension = False
        num_timesteps = 0
        if len(x.shape) == 5:
            has_time_dimension = True
            num_timesteps = x.shape[3]
            x = rearrange(x, "b h w t c -> (b t) c h w")
        else:
            x = rearrange(x, "b h w c -> b c h w")

        if not patch_size:
            patch_size = self.patch_size
        patch_size = to_2tuple(patch_size)

        if patch_size == self.patch_size:
            weight = self.proj.weight
        else:
            weight = self.resize_patch_embed(self.proj.weight, patch_size)
        x = F.conv2d(x, weight, bias=self.proj.bias, stride=patch_size)

        if has_time_dimension:
            x = rearrange(x, "(b t) c h w -> b h w t c", b=batch_size, t=num_timesteps)
        else:
            x = rearrange(x, "b c h w -> b h w c")
        x = self.norm(x)
        return x


class Attention(nn.Module):
    fast_attn: Final[bool]

    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_norm=False,
        attn_drop=0.0,
        proj_drop=0.0,
        norm_layer=nn.LayerNorm,
        cross_attn: bool = False,
    ):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.fast_attn = hasattr(torch.nn.functional, "scaled_dot_product_attention")
        self.cross_attn = cross_attn

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)

        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, y=None, attn_mask=None):
        B, N, C = x.shape
        q = self.q(x)

        if y is None:
            assert not self.cross_attn
            k = self.k(x)
            v = self.v(x)
        else:
            assert self.cross_attn
            k = self.k(y)
            v = self.v(y)

        q = rearrange(q, "b n (h d) -> b h n d", h=self.num_heads)
        k = rearrange(k, "b n (h d) -> b h n d", h=self.num_heads)
        v = rearrange(v, "b n (h d) -> b h n d", h=self.num_heads)

        q, k = self.q_norm(q), self.k_norm(k)
        if self.fast_attn:
            if attn_mask is not None:
                attn_mask = attn_mask[:, None, None].repeat((1, self.num_heads, N, 1))
            x = F.scaled_dot_product_attention(
                q, k, v, attn_mask=attn_mask, dropout_p=self.attn_drop.p,
            )
        else:
            if attn_mask is not None:
                raise NotImplementedError
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Mlp(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        bias=True,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class LayerScale(nn.Module):
    def __init__(self, dim, init_values=1e-5, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x):
        return x.mul_(self.gamma) if self.inplace else x * self.gamma


class Block(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_norm=False,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        init_values=None,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        cross_attn: bool = False,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_norm=qk_norm,
            attn_drop=attn_drop, proj_drop=drop, norm_layer=norm_layer, cross_attn=cross_attn,
        )
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()

    def forward(self, x, y, attn_mask):
        x = x + self.drop_path(self.ls1(self.attn(self.norm1(x), y, attn_mask)))
        x = x + self.drop_path(self.ls2(self.mlp(self.norm2(x))))
        return x


class ModuleListWithInit(nn.ModuleList):
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

def gpu_count() -> int:
    if torch is None:
        return 0
    try:
        return int(torch.cuda.device_count())
    except Exception:
        return 0


def device() -> str:
    """The device main.py should hand to models: 'cuda' (the one visible GPU) or 'cpu'."""
    if torch is None:
        return "cpu"
    try:
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def shard_indices() -> tuple[int, int]:
    idx, n = int(os.environ.get(SHARD_ENV, 0)), int(os.environ.get(NUM_SHARDS_ENV, 1))
    if n < 1:
        raise ValueError(f"{NUM_SHARDS_ENV} must be >= 1, got {n}")
    if idx < 0 or idx >= n:
        raise ValueError(f"{SHARD_ENV} must satisfy 0 <= shard < {NUM_SHARDS_ENV}; got {idx}/{n}")
    return idx, n


def take_shard(items: list) -> list:
    """Round-robin subset of ``items`` for this process's shard (identity if unsharded)."""
    idx, n = shard_indices()
    if n <= 1:
        return list(items)
    return [x for i, x in enumerate(items) if i % n == idx]


def fan_out(num_shards: int | None = None) -> int:
    """Launch one sharded ``main.py`` per GPU in parallel; tee per-shard logs.

    Each child gets ``CUDA_VISIBLE_DEVICES=i`` (so it sees exactly one GPU as cuda:0) plus the shard
    env, and runs the disjoint, round-robin subset of (model, benchmark) pairs. Returns the max child
    exit code; falls back to a single process if no GPUs are visible.

    MULTI-MACHINE: the (model, benchmark) pairs are sharded GLOBALLY across all participating GPUs.
    Set two env vars per machine so each GPU gets a unique GLOBAL shard index out of the GLOBAL total
    (see docs/multi_machine.md):
      * ``RB_SHARD_BASE``  -- this machine's first global shard index (default 0). Set it to the sum
        of GPU counts on the machines ordered before this one.
      * ``RB_NUM_SHARDS``  -- the GLOBAL number of shards = total GPUs across all machines.
    Single-box runs need neither (base 0, total = local GPU count) and behave exactly as before.
    """
    local_gpus = num_shards or max(1, gpu_count())
    base = int(os.environ.get("RB_SHARD_BASE", "0"))
    total = int(os.environ.get(NUM_SHARDS_ENV, str(base + local_gpus)))  # GLOBAL shard count
    if local_gpus < 1:
        raise ValueError(f"local GPU shard count must be >= 1, got {local_gpus}")
    if base < 0 or total < 1 or base + local_gpus > total:
        raise ValueError(f"Invalid shard range: base={base}, local_gpus={local_gpus}, total={total}")
    scratch = REPO / "data"
    log_dir = scratch / "output" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    per_shard_cores = max(1, (os.cpu_count() or 2) // local_gpus)
    procs = []
    for i in range(local_gpus):
        shard = base + i
        env = {
            **os.environ,
            "CUDA_VISIBLE_DEVICES": str(i),
            SHARD_ENV: str(shard),
            NUM_SHARDS_ENV: str(total),
            "LOKY_MAX_CPU_COUNT": str(per_shard_cores),
            "RB_PARENT_LOG_CAPTURE": "1",
        }
        log = open(log_dir / f"shard_{shard}.log", "w")
        proc = subprocess.Popen(
            [sys.executable, "-u", "main.py"],
            cwd=str(REPO / "src"),
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        print(f"[gputils] shard {shard}/{total} -> GPU {i} | pid {proc.pid} | log {log.name}", flush=True)
        procs.append((proc, log))
    code = 0
    for proc, log in procs:
        code = max(code, proc.wait())
        log.close()
    print(f"[gputils] all {local_gpus} local shard(s) done (max exit code {code})", flush=True)
    return code


if __name__ == "__main__":
    raise SystemExit("Run src/main.py. Set LAUNCH_GPU_SHARDS=True in src/main.py for multi-GPU fan-out.")
