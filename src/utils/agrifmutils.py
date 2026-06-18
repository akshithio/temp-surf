"""Vendored AgriFM video_swin_transformer.py — SwinTransformer3D for the S2 model.

Stubs out mmseg registries so no compiled mmcv is needed.
"""

import sys
import types
from functools import lru_cache, reduce
from operator import mul

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from einops import rearrange
from einops import reduce as einops_reduce
from timm.models.layers import DropPath, trunc_normal_


# Stub mmseg registries
class _Registry:
    def __init__(self):
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

_reg = _Registry()
for _name, _attrs in {
    "mmseg": {},
    "mmseg.models": {},
    "mmseg.models.builder": {"BACKBONES": _reg, "MODELS": _reg},
    "mmseg.registry": {"MODELS": _reg, "TRANSFORMS": _reg},
    "mmseg.registry.registry": {"MODELS": _reg, "TRANSFORMS": _reg},
    "mmengine": {},
    "mmengine.model": {"BaseModule": nn.Module, "BaseModel": nn.Module},
    "mmengine.runner": {"load_checkpoint": lambda *a, **kw: None},
}.items():
    _mod = types.ModuleType(_name)
    for _k, _v in _attrs.items():
        setattr(_mod, _k, _v)
    sys.modules[_name] = _mod

from mmengine.runner import load_checkpoint  # noqa: E402
from mmseg.models.builder import BACKBONES  # noqa: E402


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def window_partition(x, window_size):
    B, D, H, W, C = x.shape
    x = x.view(B, D // window_size[0], window_size[0], H // window_size[1], window_size[1], W // window_size[2],
               window_size[2], C)
    windows = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous().view(-1, reduce(mul, window_size), C)
    return windows


def window_reverse(windows, window_size, B, D, H, W):
    x = windows.view(B, D // window_size[0], H // window_size[1], W // window_size[2], window_size[0], window_size[1],
                     window_size[2], -1)
    x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous().view(B, D, H, W, -1)
    return x


def get_window_size(x_size, window_size, shift_size=None):
    use_window_size = list(window_size)
    if shift_size is not None:
        use_shift_size = list(shift_size)
    for i in range(len(x_size)):
        if x_size[i] <= window_size[i]:
            use_window_size[i] = x_size[i]
            if shift_size is not None:
                use_shift_size[i] = 0
    if shift_size is None:
        return tuple(use_window_size)
    return tuple(use_window_size), tuple(use_shift_size)


class WindowAttention3D(nn.Module):
    def __init__(self, dim, window_size, num_heads, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1) * (2 * window_size[2] - 1), num_heads))
        coords_d = torch.arange(self.window_size[0])
        coords_h = torch.arange(self.window_size[1])
        coords_w = torch.arange(self.window_size[2])
        coords = torch.stack(torch.meshgrid(coords_d, coords_h, coords_w))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 2] += self.window_size[2] - 1
        relative_coords[:, :, 0] *= (2 * self.window_size[1] - 1) * (2 * self.window_size[2] - 1)
        relative_coords[:, :, 1] *= (2 * self.window_size[2] - 1)
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index[:N, :N].reshape(-1)].reshape(N, N, -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)
        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SwinTransformerBlock3D(nn.Module):
    def __init__(self, dim, num_heads, window_size=(2, 7, 7), shift_size=(0, 0, 0),
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, use_checkpoint=False):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        self.use_checkpoint = use_checkpoint
        assert 0 <= self.shift_size[0] < self.window_size[0]
        assert 0 <= self.shift_size[1] < self.window_size[1]
        assert 0 <= self.shift_size[2] < self.window_size[2]
        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention3D(
            dim, window_size=self.window_size, num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward_part1(self, x, mask_matrix):
        B, D, H, W, C = x.shape
        window_size, shift_size = get_window_size((D, H, W), self.window_size, self.shift_size)
        x = self.norm1(x)
        pad_l = pad_t = pad_d0 = 0
        pad_d1 = (window_size[0] - D % window_size[0]) % window_size[0]
        pad_b = (window_size[1] - H % window_size[1]) % window_size[1]
        pad_r = (window_size[2] - W % window_size[2]) % window_size[2]
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b, pad_d0, pad_d1))
        _, Dp, Hp, Wp, _ = x.shape
        if any(i > 0 for i in shift_size):
            shifted_x = torch.roll(x, shifts=(-shift_size[0], -shift_size[1], -shift_size[2]), dims=(1, 2, 3))
            attn_mask = mask_matrix
        else:
            shifted_x = x
            attn_mask = None
        x_windows = window_partition(shifted_x, window_size)
        attn_windows = self.attn(x_windows, mask=attn_mask)
        attn_windows = attn_windows.view(-1, *(window_size + (C,)))
        shifted_x = window_reverse(attn_windows, window_size, B, Dp, Hp, Wp)
        if any(i > 0 for i in shift_size):
            x = torch.roll(shifted_x, shifts=(shift_size[0], shift_size[1], shift_size[2]), dims=(1, 2, 3))
        else:
            x = shifted_x
        if pad_d1 > 0 or pad_r > 0 or pad_b > 0:
            x = x[:, :D, :H, :W, :].contiguous()
        return x

    def forward_part2(self, x):
        return self.drop_path(self.mlp(self.norm2(x)))

    def forward(self, x, mask_matrix):
        shortcut = x
        if self.use_checkpoint:
            x = checkpoint.checkpoint(self.forward_part1, x, mask_matrix)
        else:
            x = self.forward_part1(x, mask_matrix)
        x = shortcut + self.drop_path(x)
        if self.use_checkpoint:
            x = x + checkpoint.checkpoint(self.forward_part2, x)
        else:
            x = x + self.forward_part2(x)
        return x


class PatchMerging(nn.Module):
    def __init__(self, dim, norm_layer=nn.LayerNorm, downsample_step=(2, 2, 2), mean_frame_down=False):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)
        self.downsample_step = downsample_step
        self.mean_frame_down = mean_frame_down

    def forward(self, x):
        B, D, H, W, C = x.shape
        pad_input = (H % 2 == 1) or (W % 2 == 1)
        if pad_input:
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))
        D_step, H_step, W_step = self.downsample_step
        if D_step >= 2 and self.mean_frame_down and D % D_step != 0:
            x = F.pad(x, (0, 0, 0, 0, 0, 0, 0, D_step - D % D_step))
        h = 0 if H_step == 1 else 1
        w = 0 if W_step == 1 else 1
        x0 = x[:, ::D_step, 0::H_step, 0::W_step, :]
        x1 = x[:, ::D_step, h::H_step, 0::W_step, :]
        x2 = x[:, ::D_step, 0::H_step, w::W_step, :]
        x3 = x[:, ::D_step, h::H_step, w::W_step, :]
        if D_step >= 2 and self.mean_frame_down:
            for i in range(1, D_step):
                x0 = x0 + x[:, i::D_step, 0::H_step, 0::W_step, :]
                x1 = x1 + x[:, i::D_step, h::H_step, 0::W_step, :]
                x2 = x2 + x[:, i::D_step, 0::H_step, w::W_step, :]
                x3 = x3 + x[:, i::D_step, h::H_step, w::W_step, :]
            x0 = x0 / D_step
            x1 = x1 / D_step
            x2 = x2 / D_step
            x3 = x3 / D_step
        x = torch.cat([x0, x1, x2, x3], -1)
        x = self.norm(x)
        x = self.reduction(x)
        return x


@lru_cache
def compute_mask(D, H, W, window_size, shift_size, device):
    img_mask = torch.zeros((1, D, H, W, 1), device=device)
    cnt = 0
    for d in slice(-window_size[0]), slice(-window_size[0], -shift_size[0]), slice(-shift_size[0], None):
        for h in slice(-window_size[1]), slice(-window_size[1], -shift_size[1]), slice(-shift_size[1], None):
            for w in slice(-window_size[2]), slice(-window_size[2], -shift_size[2]), slice(-shift_size[2], None):
                img_mask[:, d, h, w, :] = cnt
                cnt += 1
    mask_windows = window_partition(img_mask, window_size)
    mask_windows = mask_windows.squeeze(-1)
    attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
    attn_mask = attn_mask.masked_fill(attn_mask != 0, (-100.0)).masked_fill(attn_mask == 0, 0.0)
    return attn_mask


class BasicLayer(nn.Module):
    def __init__(self, dim, depth, num_heads, window_size=(1, 7, 7), mlp_ratio=4.,
                 qkv_bias=False, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 norm_layer=nn.LayerNorm, downsample=None, downsample_step=(2, 2, 2),
                 mean_frame_down=False, use_checkpoint=False):
        super().__init__()
        self.window_size = window_size
        self.shift_size = tuple(i // 2 for i in window_size)
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        self.mean_frame_down = mean_frame_down
        self.blocks = nn.ModuleList([
            SwinTransformerBlock3D(
                dim=dim, num_heads=num_heads, window_size=window_size,
                shift_size=(0, 0, 0) if (i % 2 == 0) else self.shift_size,
                mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop, attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer, use_checkpoint=use_checkpoint)
            for i in range(depth)])
        self.downsample = downsample
        if self.downsample is not None:
            self.downsample = downsample(dim=dim, norm_layer=norm_layer, downsample_step=downsample_step,
                                         mean_frame_down=mean_frame_down)

    def forward(self, x):
        B, C, D, H, W = x.shape
        window_size, shift_size = get_window_size((D, H, W), self.window_size, self.shift_size)
        x = rearrange(x, 'b c d h w -> b d h w c')
        Dp = int(np.ceil(D / window_size[0])) * window_size[0]
        Hp = int(np.ceil(H / window_size[1])) * window_size[1]
        Wp = int(np.ceil(W / window_size[2])) * window_size[2]
        attn_mask = compute_mask(Dp, Hp, Wp, window_size, shift_size, x.device)
        for blk in self.blocks:
            x = blk(x, attn_mask)
        x = x.view(B, D, H, W, -1)
        if self.downsample is not None:
            x = self.downsample(x)
        x = rearrange(x, 'b d h w c -> b c d h w')
        return x


@BACKBONES.register_module()
class SwinTransformer3D(nn.Module):
    def __init__(self, pretrained=None, pretrained2d=True, patch_size=(4, 4, 4),
                 embed_dim=96, depths=None, num_heads=None,
                 window_size=(2, 7, 7), mlp_ratio=4., out_indices=(0, 1, 2, 3),
                 qkv_bias=True, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0.2, norm_layer=nn.LayerNorm, patch_norm=False,
                 downsample_steps=((2, 2, 2), (2, 2, 2), (2, 2, 2), (2, 2, 2)),
                 frozen_stages=-1, use_checkpoint=False, feature_fusion='cat',
                 mean_frame_down=False, reduce_feature_scale=None):
        if num_heads is None:
            num_heads = [3, 6, 12, 24]
        if depths is None:
            depths = [2, 2, 6, 2]
        super().__init__()
        self.pretrained = pretrained
        self.pretrained2d = pretrained2d
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        self.frozen_stages = frozen_stages
        self.window_size = window_size
        self.patch_size = patch_size
        self.mean_frame_down = mean_frame_down
        self.reduce_feature_scale = reduce_feature_scale
        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(
                dim=int(embed_dim * 2 ** i_layer),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                downsample=PatchMerging if i_layer < self.num_layers - 1 else None,
                downsample_step=downsample_steps[i_layer],
                mean_frame_down=mean_frame_down,
                use_checkpoint=use_checkpoint,
            )
            self.layers.append(layer)
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.feature_fusion = feature_fusion
        self.norm = nn.LayerNorm(self.num_features)
        self.out_indices = out_indices
        self._freeze_stages()

    def _freeze_stages(self):
        if self.frozen_stages >= 1:
            self.pos_drop.eval()
            for i in range(0, self.frozen_stages):
                m = self.layers[i]
                m.eval()
                for param in m.parameters():
                    param.requires_grad = False

    def inflate_weights(self):
        checkpoint_w = torch.load(self.pretrained, map_location='cpu')
        state_dict = checkpoint_w['model']
        relative_position_index_keys = [k for k in state_dict.keys() if "relative_position_index" in k]
        for k in relative_position_index_keys:
            del state_dict[k]
        attn_mask_keys = [k for k in state_dict.keys() if "attn_mask" in k]
        for k in attn_mask_keys:
            del state_dict[k]
        relative_position_bias_table_keys = [k for k in state_dict.keys() if "relative_position_bias_table" in k]
        for k in relative_position_bias_table_keys:
            relative_position_bias_table_pretrained = state_dict[k]
            if k not in self.state_dict():
                continue
            relative_position_bias_table_current = self.state_dict()[k]
            L1, nH1 = relative_position_bias_table_pretrained.size()
            L2, nH2 = relative_position_bias_table_current.size()
            L2 = (2 * self.window_size[1] - 1) * (2 * self.window_size[2] - 1)
            wd = self.window_size[0]
            if nH1 != nH2:
                print(f"Error in loading {k}, passing")
            else:
                if L1 != L2:
                    S1 = int(L1 ** 0.5)
                    relative_position_bias_table_pretrained_resized = torch.nn.functional.interpolate(
                        relative_position_bias_table_pretrained.permute(1, 0).view(1, nH1, S1, S1),
                        size=(2 * self.window_size[1] - 1, 2 * self.window_size[2] - 1),
                        mode='bicubic')
                    relative_position_bias_table_pretrained = relative_position_bias_table_pretrained_resized.view(
                        nH2, L2).permute(1, 0)
            state_dict[k] = relative_position_bias_table_pretrained.repeat(2 * wd - 1, 1)
        for key in list(state_dict.keys()):
            if key.startswith('patch_embed'):
                state_dict.pop(key)
        msg = self.load_state_dict(state_dict, strict=False)
        print(msg)
        del checkpoint_w
        torch.cuda.empty_cache()

    def init_weights(self, pretrained=None):
        def _init_weights(m):
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
        if pretrained:
            self.pretrained = pretrained
        if isinstance(self.pretrained, str):
            self.apply(_init_weights)
            print(f'load model from: {self.pretrained}')
            if self.pretrained2d:
                self.inflate_weights()
            else:
                load_checkpoint(self, self.pretrained, strict=False)
        elif self.pretrained is None:
            self.apply(_init_weights)
        else:
            raise TypeError('pretrained must be a str or None')

    def forward(self, x):
        if 0 in self.out_indices:
            if self.feature_fusion == 'cat':
                feat = einops_reduce(x, 'b c (t s) h w -> b c t h w', 'mean', s=self.reduce_feature_scale) if self.reduce_feature_scale is not None else x
                B, C, D, H, W = feat.shape
                out_feats = [torch.reshape(feat, (B, C * D, H, W))]
            elif self.feature_fusion == 'mean':
                out_feats = [torch.mean(x, dim=2)]
            else:
                out_feats = [x]
        x = self.pos_drop(x)
        for i, layer in enumerate(self.layers):
            x = layer(x.contiguous())
            if i + 1 in self.out_indices:
                B, C, D, H, W = x.shape
                if self.feature_fusion == 'cat':
                    if self.reduce_feature_scale is not None and D % self.reduce_feature_scale != 0:
                        feat = torch.nn.functional.pad(x, (0, 0, 0, 0, 0, self.reduce_feature_scale - D % self.reduce_feature_scale))
                    else:
                        feat = x
                    feat = einops_reduce(feat, 'b c (t s) h w -> b c t h w', 'mean', s=self.reduce_feature_scale) if self.reduce_feature_scale is not None else x
                    B, C, D, H, W = feat.shape
                    out_feats.append(torch.reshape(feat, (B, C * D, H, W)))
                elif self.feature_fusion == 'mean':
                    out_feats.append(torch.mean(x, dim=2))
                else:
                    out_feats.append(x)
        x = rearrange(x, 'n c d h w -> n d h w c')
        x = self.norm(x)
        x = rearrange(x, 'n d h w c -> n c d h w')
        B, C, D, H, W = x.shape
        if self.feature_fusion == 'cat':
            feat = einops_reduce(x, 'b c (t s) h w -> b c t h w', 'mean', s=self.reduce_feature_scale) if self.reduce_feature_scale is not None else x
            B, C, D, H, W = feat.shape
            x = torch.reshape(feat, (B, C * D, H, W))
        elif self.feature_fusion == 'mean':
            x = torch.mean(x, dim=2)
        return {'model_features': x, 'features_list': out_feats}

    def train(self, mode=True):
        super().train(mode)
        self._freeze_stages()
