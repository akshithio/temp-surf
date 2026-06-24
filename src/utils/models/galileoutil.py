import json
from collections import OrderedDict
from collections import OrderedDict as OrderedDictType
from pathlib import Path

import torch
import torch.nn as nn
from einops import rearrange, repeat

from utils.gputils import (
    Block,
    FlexiPatchEmbed,
    ModuleListWithInit,
    get_1d_sincos_pos_embed_from_grid_torch,
    get_2d_sincos_pos_embed_with_resolution,
    get_month_encoding_table,
)

# constants
CONFIG_FILENAME = "config.json"
MODEL_FILENAME = "model.pt"
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


class GalileoNativeModel(GalileoBase):
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
        if not (folder / MODEL_FILENAME).exists():
            raise ValueError(f"Expected {MODEL_FILENAME} in {folder}")

        with (folder / CONFIG_FILENAME).open("r") as f:
            config = json.load(f)
            model_section = config["model"]
            # Pinned Galileo releases store the inference model under ``encoder``. Retain the
            # nested ``model`` fallback for older exported folders.
            model_config = model_section.get("encoder", model_section.get("model"))
            if model_config is None:
                raise ValueError(f"Expected model.encoder or model.model in {folder / CONFIG_FILENAME}")
        model = cls(**model_config)

        state_dict = torch.load(folder / MODEL_FILENAME, map_location=device, weights_only=True)
        for key in list(state_dict.keys()):
            state_dict[key.replace(".backbone", "")] = state_dict.pop(key)
        model.load_state_dict(state_dict)
        return model
