import collections.abc
import itertools
import json
from collections import OrderedDict
from collections import OrderedDict as OrderedDictType
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import Tensor, vmap
from torch.jit import Final

# constants
CONFIG_FILENAME = "config.json"
ENCODER_FILENAME = "encoder.pt"
BASE_GSD = 10

# band information
S1_BANDS = ["VV", "VH"]
S2_BANDS = [
    "B2",
    "B3",
    "B4",
    "B5",
    "B6",
    "B7",
    "B8",
    "B8A",
    "B11",
    "B12",
]
ERA5_BANDS = ["temperature_2m", "total_precipitation_sum"]
TC_BANDS = ["def", "soil", "aet"]
VIIRS_BANDS = ["avg_rad"]
SRTM_BANDS = ["elevation", "slope"]
DW_BANDS = [
    "DW_water",
    "DW_trees",
    "DW_grass",
    "DW_flooded_vegetation",
    "DW_crops",
    "DW_shrub_and_scrub",
    "DW_built",
    "DW_bare",
    "DW_snow_and_ice",
]
WC_BANDS = [
    "WC_temporarycrops",
    "WC_maize",
    "WC_wintercereals",
    "WC_springcereals",
    "WC_irrigation",
]
STATIC_DW_BANDS = [f"{x}_static" for x in DW_BANDS]
STATIC_WC_BANDS = [f"{x}_static" for x in WC_BANDS]

LANDSCAN_BANDS = ["b1"]
LOCATION_BANDS = ["x", "y", "z"]

SPACE_TIME_BANDS = S1_BANDS + S2_BANDS + ["NDVI"]
TIME_BANDS = ERA5_BANDS + TC_BANDS + VIIRS_BANDS
SPACE_BANDS = SRTM_BANDS + DW_BANDS + WC_BANDS
STATIC_BANDS = LANDSCAN_BANDS + LOCATION_BANDS + STATIC_DW_BANDS + STATIC_WC_BANDS


SPACE_TIME_BANDS_GROUPS_IDX: OrderedDictType[str, list[int]] = OrderedDict(
    {
        "S1": [SPACE_TIME_BANDS.index(b) for b in S1_BANDS],
        "S2_RGB": [SPACE_TIME_BANDS.index(b) for b in ["B2", "B3", "B4"]],
        "S2_Red_Edge": [SPACE_TIME_BANDS.index(b) for b in ["B5", "B6", "B7"]],
        "S2_NIR_10m": [SPACE_TIME_BANDS.index(b) for b in ["B8"]],
        "S2_NIR_20m": [SPACE_TIME_BANDS.index(b) for b in ["B8A"]],
        "S2_SWIR": [SPACE_TIME_BANDS.index(b) for b in ["B11", "B12"]],
        "NDVI": [SPACE_TIME_BANDS.index("NDVI")],
    }
)

TIME_BAND_GROUPS_IDX: OrderedDictType[str, list[int]] = OrderedDict(
    {
        "ERA5": [TIME_BANDS.index(b) for b in ERA5_BANDS],
        "TC": [TIME_BANDS.index(b) for b in TC_BANDS],
        "VIIRS": [TIME_BANDS.index(b) for b in VIIRS_BANDS],
    }
)

SPACE_BAND_GROUPS_IDX: OrderedDictType[str, list[int]] = OrderedDict(
    {
        "SRTM": [SPACE_BANDS.index(b) for b in SRTM_BANDS],
        "DW": [SPACE_BANDS.index(b) for b in DW_BANDS],
        "WC": [SPACE_BANDS.index(b) for b in WC_BANDS],
    }
)

STATIC_BAND_GROUPS_IDX: OrderedDictType[str, list[int]] = OrderedDict(
    {
        "LS": [STATIC_BANDS.index(b) for b in LANDSCAN_BANDS],
        "location": [STATIC_BANDS.index(b) for b in LOCATION_BANDS],
        "DW_static": [STATIC_BANDS.index(b) for b in STATIC_DW_BANDS],
        "WC_static": [STATIC_BANDS.index(b) for b in STATIC_WC_BANDS],
    }
)


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


def drop_path(x, drop_prob: float = 0.0, training: bool = False):
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


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


class GalileoBase(nn.Module):
    cross_attn: bool

    def __init__(
        self,
        embedding_size: int = 128,
        depth=2,
        mlp_ratio=2,
        num_heads=8,
        max_sequence_length=24,
        base_patch_size: int = 4,
        use_channel_embs: bool = True,
        drop_path: float = 0.0,
    ):
        super().__init__()

        self.space_time_groups = SPACE_TIME_BANDS_GROUPS_IDX
        self.space_groups = SPACE_BAND_GROUPS_IDX
        self.time_groups = TIME_BAND_GROUPS_IDX
        self.static_groups = STATIC_BAND_GROUPS_IDX
        self.embedding_size = embedding_size
        self.base_patch_size = base_patch_size

        self.blocks = ModuleListWithInit(
            [
                Block(
                    embedding_size, num_heads, mlp_ratio, qkv_bias=True,
                    norm_layer=nn.LayerNorm, cross_attn=self.cross_attn, drop_path=drop_path,
                )
                for _ in range(depth)
            ]
        )

        self.max_sequence_length = max_sequence_length
        self.pos_embed = nn.Parameter(
            get_1d_sincos_pos_embed_from_grid_torch(
                int(embedding_size * 0.25), torch.arange(max_sequence_length)
            ),
            requires_grad=False,
        )
        month_tab = get_month_encoding_table(int(embedding_size * 0.25))
        self.month_embed = nn.Embedding.from_pretrained(month_tab, freeze=True)
        if use_channel_embs:
            args = {"requires_grad": True}
        else:
            args = {"requires_grad": False}
        self.s_t_channel_embed = nn.Parameter(
            torch.zeros(len(SPACE_TIME_BANDS_GROUPS_IDX), int(embedding_size * 0.25)), **args
        )
        self.sp_channel_embed = nn.Parameter(
            torch.zeros(len(SPACE_BAND_GROUPS_IDX), int(embedding_size * 0.25)), **args
        )
        self.t_channel_embed = nn.Parameter(
            torch.zeros(len(TIME_BAND_GROUPS_IDX), int(embedding_size * 0.25)), **args
        )
        self.st_channel_embed = nn.Parameter(
            torch.zeros(len(STATIC_BAND_GROUPS_IDX), int(embedding_size * 0.25)), **args
        )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    @classmethod
    def collapse_and_combine_hwtc(
        cls, s_t_x, sp_x, t_x, st_x, s_t_m, sp_m, t_m, st_m,
    ):
        s_t_x = rearrange(s_t_x, "b h w t c_g d -> b (h w t c_g) d")
        sp_x = rearrange(sp_x, "b h w c_g d -> b (h w c_g) d")
        t_x = rearrange(t_x, "b t c_g d -> b (t c_g) d")
        s_t_m = rearrange(s_t_m, "b h w t c_g-> b (h w t c_g)")
        sp_m = rearrange(sp_m, "b h w c_g-> b (h w c_g)")
        t_m = rearrange(t_m, "b t c_g -> b (t c_g)")
        x = torch.cat([s_t_x, sp_x, t_x, st_x], dim=1)
        m = torch.cat([s_t_m, sp_m, t_m, st_m], dim=1)
        return x, m

    @classmethod
    def split_and_expand_hwtc(cls, x, h, w, t, s_t_c_g, sp_c_g, t_c_g, st_c_g):
        n_s_t_t = h * w * t * s_t_c_g
        n_t_t = t * t_c_g
        s_t_x = rearrange(x[:, :n_s_t_t], "b (h w t c) d -> b h w t c d", h=h, w=w, t=t, c=s_t_c_g)
        sp_x = rearrange(x[:, n_s_t_t : -(n_t_t + st_c_g)], "b (h w c) d -> b h w c d", h=h, w=w, c=sp_c_g)
        t_x = rearrange(x[:, -(n_t_t + st_c_g) : -st_c_g], "b (t c) d -> b t c d", t=t, c=t_c_g)
        st_x = x[:, -st_c_g:]
        return s_t_x, sp_x, t_x, st_x

    def apply_encodings(self, s_t_x, sp_x, t_x, st_x, months, patch_size, input_res):
        b, h, w, t, s_t_c_g, _ = s_t_x.shape
        sp_c_g, t_c_g = sp_x.shape[-2], t_x.shape[-2]
        st_c_g = st_x.shape[-2]

        s_t_channel = repeat(self.s_t_channel_embed, "c_g d -> b h w t c_g d", b=b, h=h, w=w, t=t)
        t_channel = repeat(self.t_channel_embed, "c_g d -> b t c_g d", b=b, t=t)
        st_channel = repeat(self.st_channel_embed, "c_g d -> b c_g d", b=b)
        sp_channel = repeat(self.sp_channel_embed, "c_g d -> b h w c_g d", b=b, h=h, w=w)

        pos_embed_s_t = repeat(self.pos_embed[:t], "t d -> b h w t c_g d", b=b, h=h, w=w, c_g=s_t_c_g)
        m_embed_s_t = repeat(self.month_embed(months), "b t d -> b h w t c_g d", h=h, w=w, c_g=s_t_c_g)
        pos_embed_t = repeat(self.pos_embed[:t], "t d -> b t c_g d", b=b, c_g=t_c_g)
        m_embed_t = repeat(self.month_embed(months), "b t d -> b t c_g d", c_g=t_c_g)
        t_zeros = torch.zeros(b, t, t_c_g, int(self.embedding_size * 0.25), device=t_x.device)
        sp_zeros = torch.zeros(b, h, w, sp_c_g, sp_channel.shape[-1] * 2, device=sp_channel.device)
        st_zeros = torch.zeros(b, st_c_g, st_channel.shape[-1] * 3, device=st_channel.device)

        if patch_size is None:
            patch_size = self.base_patch_size
        token_res = input_res * patch_size
        gsd_ratio = token_res / BASE_GSD

        assert h == w
        spatial_embed = get_2d_sincos_pos_embed_with_resolution(
            int(self.embedding_size * 0.25), h,
            torch.ones(b).to(s_t_x.device) * gsd_ratio, device=s_t_x.device,
        )
        spatial_embed = rearrange(spatial_embed, "b (h w) d -> b h w d", h=h, w=w)
        spatial_embed_s_t = repeat(spatial_embed, "b h w d -> b h w t c_g d", h=h, w=w, t=t, c_g=s_t_c_g)
        spatial_embed_s = repeat(spatial_embed, "b h w d -> b h w c_g d", h=h, w=w, c_g=sp_c_g)

        s_t_embed = torch.cat([s_t_channel, pos_embed_s_t, m_embed_s_t, spatial_embed_s_t], dim=-1)
        sp_embed = torch.cat([sp_channel, sp_zeros, spatial_embed_s], dim=-1)
        t_embed = torch.cat([t_channel, pos_embed_t, m_embed_t, t_zeros], dim=-1)
        st_embed = torch.cat([st_channel, st_zeros], dim=-1)
        return s_t_x + s_t_embed, sp_x + sp_embed, t_x + t_embed, st_x + st_embed


class Encoder(GalileoBase):
    cross_attn = False

    def __init__(
        self,
        max_patch_size: int = 8,
        embedding_size: int = 128,
        depth=2,
        mlp_ratio=2,
        num_heads=8,
        max_sequence_length=24,
        freeze_projections: bool = False,
        drop_path: float = 0.0,
    ):
        super().__init__(
            embedding_size, depth, mlp_ratio, num_heads, max_sequence_length,
            max_patch_size, use_channel_embs=True, drop_path=drop_path,
        )

        self.space_time_embed = nn.ModuleDict({
            group_name: FlexiPatchEmbed(in_chans=len(group), embed_dim=embedding_size, patch_size=max_patch_size)
            for group_name, group in self.space_time_groups.items()
        })
        self.space_embed = nn.ModuleDict({
            group_name: FlexiPatchEmbed(in_chans=len(group), embed_dim=embedding_size, patch_size=max_patch_size)
            for group_name, group in self.space_groups.items()
        })
        self.time_embed = nn.ModuleDict({
            group_name: nn.Linear(in_features=len(group), out_features=embedding_size)
            for group_name, group in self.time_groups.items()
        })
        self.static_embed = nn.ModuleDict({
            group_name: nn.Linear(in_features=len(group), out_features=embedding_size)
            for group_name, group in self.static_groups.items()
        })
        if freeze_projections:
            self.space_time_embed.requires_grad_(False)
            self.space_embed.requires_grad_(False)
            self.time_embed.requires_grad_(False)
            self.static_embed.requires_grad_(False)
        self.norm = nn.LayerNorm(embedding_size)
        self.apply(self._init_weights)

    def apply_linear_projection(self, s_t_x, sp_x, t_x, st_x, s_t_m, sp_m, t_m, st_m, patch_size):
        b, h, w, t, _ = s_t_x.shape
        new_h, new_w = h // patch_size, w // patch_size

        s_t_l, sp_l, t_l, st_l, s_t_m_l, sp_m_l, t_m_l, st_m_l = [], [], [], [], [], [], [], []
        for idx, (channel_group, channel_idxs) in enumerate(self.space_time_groups.items()):
            s_t_m_l.append(s_t_m[:, 0::patch_size, 0::patch_size, :, idx])
            if s_t_m_l[-1].min() == 0:
                s_t_l.append(self.space_time_embed[channel_group](s_t_x[:, :, :, :, channel_idxs], patch_size=patch_size))
            else:
                s_t_l.append(torch.zeros(b, new_h, new_w, t, self.embedding_size, dtype=s_t_x.dtype, device=s_t_x.device))
        for idx, (channel_group, channel_idxs) in enumerate(self.space_groups.items()):
            sp_m_l.append(sp_m[:, 0::patch_size, 0::patch_size, idx])
            if sp_m_l[-1].min() == 0:
                sp_l.append(self.space_embed[channel_group](sp_x[:, :, :, channel_idxs], patch_size=patch_size))
            else:
                sp_l.append(torch.zeros(b, new_h, new_w, self.embedding_size, dtype=sp_x.dtype, device=sp_x.device))
        for idx, (channel_group, channel_idxs) in enumerate(self.time_groups.items()):
            t_m_l.append(t_m[:, :, idx])
            if t_m_l[-1].min() == 0:
                t_l.append(self.time_embed[channel_group](t_x[:, :, channel_idxs]))
            else:
                t_l.append(torch.zeros(b, t, self.embedding_size, dtype=t_x.dtype, device=t_x.device))
        for idx, (channel_group, channel_idxs) in enumerate(self.static_groups.items()):
            st_m_l.append(st_m[:, idx])
            if st_m_l[-1].min() == 0:
                st_l.append(self.static_embed[channel_group](st_x[:, channel_idxs]))
            else:
                st_l.append(torch.zeros(b, self.embedding_size, dtype=st_x.dtype, device=st_x.device))

        return (
            torch.stack(s_t_l, dim=-2), torch.stack(sp_l, dim=-2),
            torch.stack(t_l, dim=-2), torch.stack(st_l, dim=-2),
            torch.stack(s_t_m_l, dim=-1), torch.stack(sp_m_l, dim=-1),
            torch.stack(t_m_l, dim=-1), torch.stack(st_m_l, dim=-1),
        )

    @staticmethod
    def remove_masked_tokens(x, mask):
        org_mask_dtype = mask.dtype
        mask = mask.bool()
        sorted_mask, indices = torch.sort((~mask).int(), dim=1, descending=True, stable=True)
        x = x.gather(1, indices[:, :, None].expand_as(x))
        x = x * sorted_mask.unsqueeze(-1)
        max_length = sorted_mask.sum(-1).max()
        x = x[:, :max_length]
        updated_mask = 1 - sorted_mask[:, :max_length]
        return x, indices, updated_mask.to(dtype=org_mask_dtype)

    @staticmethod
    def add_removed_tokens(x, indices, mask):
        masked_tokens = repeat(torch.zeros_like(x[0, 0, :]), "d -> b t d", b=x.shape[0], t=indices.shape[1])
        full_mask = torch.cat((mask, torch.ones((x.shape[0], indices.shape[1] - x.shape[1]), device=x.device, dtype=mask.dtype)), dim=-1)
        out = masked_tokens.clone()
        out[~full_mask.bool()] = x[~mask.bool()]
        out = out.scatter(1, indices[:, :, None].expand_as(out), out)
        full_mask = full_mask.scatter(1, indices.expand_as(full_mask), full_mask)
        return out, full_mask

    def apply_attn(self, s_t_x, sp_x, t_x, st_x, s_t_m, sp_m, t_m, st_m, months, patch_size, input_res, exit_after, token_exit_cfg):
        if token_exit_cfg:
            exit_s_t, exit_sp, exit_t, exit_st = self.create_token_exit_ids(s_t_x, sp_x, t_x, st_x, token_exit_cfg)
            exit_ids_seq, _ = self.collapse_and_combine_hwtc(exit_s_t, exit_sp, exit_t, exit_st, s_t_m, sp_m, t_m, st_m)
            exited_tokens, _ = self.collapse_and_combine_hwtc(s_t_x, sp_x, t_x, st_x, s_t_m, sp_m, t_m, st_m)
        else:
            exit_ids_seq = None
            exited_tokens = None

        _, h, w, t, s_t_c_g, _ = s_t_x.shape
        sp_c_g, t_c_g, st_c_g = sp_x.shape[3], t_x.shape[-2], st_x.shape[-2]
        s_t_x, sp_x, t_x, st_x = self.apply_encodings(s_t_x, sp_x, t_x, st_x, months, patch_size, input_res)
        x, m = self.collapse_and_combine_hwtc(s_t_x, sp_x, t_x, st_x, s_t_m, sp_m, t_m, st_m)

        new_m = m >= 1
        x, indices, new_m = self.remove_masked_tokens(x, new_m)

        if exit_ids_seq is not None:
            exit_ids_seq, _, _ = self.remove_masked_tokens(exit_ids_seq, m >= 1)
            exited_tokens, _, _ = self.remove_masked_tokens(exited_tokens, m >= 1)

        for i_blk, blk in enumerate(self.blocks):
            if (exit_after is not None) and ((i_blk + 1) > exit_after):
                break
            if (exit_ids_seq is not None) and (i_blk > 0):
                assert exited_tokens is not None
                exited_tokens = torch.where(condition=(exit_ids_seq == i_blk), input=x.detach(), other=exited_tokens.detach())
            x = blk(x=x, y=None, attn_mask=~new_m.bool())

        if exit_ids_seq is not None:
            assert exited_tokens is not None
            x = torch.where(condition=(exit_ids_seq == (i_blk + 1)), input=x.detach(), other=exited_tokens.detach())

        x, _ = self.add_removed_tokens(x, indices, new_m)
        return (*self.split_and_expand_hwtc(x, h, w, t, s_t_c_g, sp_c_g, t_c_g, st_c_g), s_t_m, sp_m, t_m, st_m)

    @classmethod
    def average_tokens(cls, s_t_x, sp_x, t_x, st_x, s_t_m, sp_m, t_m, st_m):
        x, m = cls.collapse_and_combine_hwtc(s_t_x, sp_x, t_x, st_x, s_t_m, sp_m, t_m, st_m)
        x, _, m = cls.remove_masked_tokens(x, m)
        x_for_mean = x * (1 - m.unsqueeze(-1))
        return x_for_mean.sum(dim=1) / torch.sum(1 - m, -1, keepdim=True)

    def forward(
        self,
        s_t_x, sp_x, t_x, st_x, s_t_m, sp_m, t_m, st_m, months,
        patch_size: int,
        input_resolution_m: int | None = BASE_GSD,
        exit_after: int | None = None,
        token_exit_cfg: dict | None = None,
        add_layernorm_on_exit: bool = True,
    ):
        s_t_x, sp_x, t_x, st_x, s_t_m, sp_m, t_m, st_m = self.apply_linear_projection(
            s_t_x, sp_x, t_x, st_x, s_t_m, sp_m, t_m, st_m, patch_size,
        )

        if (exit_after is None) or (exit_after > 0):
            s_t_x, sp_x, t_x, st_x, s_t_m, sp_m, t_m, st_m = self.apply_attn(
                s_t_x, sp_x, t_x, st_x, s_t_m, sp_m, t_m, st_m,
                months, patch_size, input_resolution_m, exit_after=exit_after, token_exit_cfg=token_exit_cfg,
            )

        if add_layernorm_on_exit:
            s_t_x = self.norm(s_t_x)
            sp_x = self.norm(sp_x)
            t_x = self.norm(t_x)
            st_x = self.norm(st_x)
        return s_t_x, sp_x, t_x, st_x, s_t_m, sp_m, t_m, st_m, months

    @classmethod
    def load_from_folder(cls, folder: Path, device: torch.device):
        if not (folder / CONFIG_FILENAME).exists():
            raise ValueError(f"Expected {CONFIG_FILENAME} in {folder}")
        if not (folder / ENCODER_FILENAME).exists():
            raise ValueError(f"Expected {ENCODER_FILENAME} in {folder}")

        with (folder / CONFIG_FILENAME).open("r") as f:
            config = json.load(f)
            model_config = config["model"]
            encoder_config = model_config["encoder"]
        encoder = cls(**encoder_config)

        state_dict = torch.load(folder / ENCODER_FILENAME, map_location=device)
        for key in list(state_dict.keys()):
            state_dict[key.replace(".backbone", "")] = state_dict.pop(key)
        encoder.load_state_dict(state_dict)
        return encoder
