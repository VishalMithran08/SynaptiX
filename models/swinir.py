"""
models/swinir.py — SwinIR-style transformer for joint denoise + 2x super-resolution.

Why a transformer at all, given a well-tuned NAFNet already exists:

    A convolution aggregates from a LOCAL neighbourhood. Where speckle has
    destroyed local texture there is nothing left to reconstruct from, so an
    L1-trained CNN returns the conditional mean -- which is smooth. Measured on
    this dataset, the width-160 NAFNet reaches only ~30% of the ground truth's
    high-frequency energy, and the shortfall is concentrated in stochastic
    texture (rock, grain, weave).

    Self-attention can aggregate from SIMILAR PATCHES ELSEWHERE in the image.
    Texture repeats; if patch A is destroyed but a similar patch B survived, a
    transformer can borrow B's detail. This is non-local self-similarity -- the
    principle BM3D exploited by hand -- and it is the one mechanism our CNN
    structurally lacks.

    NOTE this is why window (spatial) attention is used and not Restormer-style
    channel attention: channel attention provides no non-local SPATIAL
    aggregation, and NAFNet already has channel attention. Paying transformer
    costs for a mechanism we already own would be pointless.

Architecture (Liang et al., "SwinIR", ICCVW 2021), adapted:

    input  [B, 1, H, W]     float32, may exceed [0,1] -- NEVER clipped
      -> shallow feature    Conv 3x3, 1 -> embed_dim
      -> deep feature       N x RSTB (each = depth x Swin blocks + conv + residual)
      -> Conv 3x3 + long skip from shallow feature
      -> upsample           Conv -> PixelShuffle(2), ICNR-initialised
      -> clamp(0, 1)        <- the ONLY clamp
    output [B, 1, 2H, 2W]

Unlike classification Swin there is no patch merging: resolution is constant
throughout, which is what restoration needs and what makes it memory-hungry.
`enable_gradient_checkpointing()` is provided for that reason.

Any input size is accepted: the forward pass pads up to a multiple of the
window size and crops the result back.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .nafnet import icnr_init


# ---------------------------------------------------------------------------
# Window helpers
# ---------------------------------------------------------------------------

def window_partition(x: torch.Tensor, ws: int) -> torch.Tensor:
    """[B, H, W, C] -> [B*nW, ws, ws, C]"""
    b, h, w, c = x.shape
    x = x.view(b, h // ws, ws, w // ws, ws, c)
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, ws, ws, c)


def window_reverse(windows: torch.Tensor, ws: int, h: int, w: int) -> torch.Tensor:
    """[B*nW, ws, ws, C] -> [B, H, W, C]"""
    b = windows.shape[0] // (h * w // ws // ws)
    x = windows.view(b, h // ws, w // ws, ws, ws, -1)
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(b, h, w, -1)


class WindowAttention(nn.Module):
    """Multi-head self-attention inside a window, with relative position bias."""

    def __init__(self, dim: int, window_size: int, num_heads: int):
        super().__init__()
        self.dim = dim
        self.ws = window_size
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5

        # A learnable bias per (relative offset, head). Offsets range over
        # [-(ws-1), ws-1] in each axis, hence (2*ws-1)^2 entries.
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) ** 2, num_heads)
        )
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

        coords = torch.stack(torch.meshgrid(
            torch.arange(window_size), torch.arange(window_size), indexing="ij"))
        coords_flat = torch.flatten(coords, 1)                    # [2, ws*ws]
        rel = coords_flat[:, :, None] - coords_flat[:, None, :]   # [2, N, N]
        rel = rel.permute(1, 2, 0).contiguous()
        rel[:, :, 0] += window_size - 1
        rel[:, :, 1] += window_size - 1
        rel[:, :, 0] *= 2 * window_size - 1
        self.register_buffer("relative_position_index", rel.sum(-1),
                             persistent=False)

        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None):
        """x: [B*nW, N, C] where N = ws*ws."""
        bn, n, c = x.shape
        qkv = (self.qkv(x)
               .reshape(bn, n, 3, self.num_heads, c // self.num_heads)
               .permute(2, 0, 3, 1, 4))
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q * self.scale) @ k.transpose(-2, -1)

        bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(n, n, -1).permute(2, 0, 1).contiguous()
        attn = attn + bias.unsqueeze(0)

        if mask is not None:
            nw = mask.shape[0]
            attn = attn.view(bn // nw, nw, self.num_heads, n, n) \
                       + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, n, n)

        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(bn, n, c)
        return self.proj(out)


class SwinBlock(nn.Module):
    """LayerNorm -> (shifted) window attention -> MLP, both residual."""

    def __init__(self, dim: int, num_heads: int, window_size: int,
                 shift_size: int, mlp_ratio: float = 2.0):
        super().__init__()
        self.ws = window_size
        self.shift = shift_size
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, window_size, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim)
        )

    def _attn_mask(self, h: int, w: int, device) -> torch.Tensor | None:
        """Mask so that pixels rolled across the image edge do not attend to
        each other after a cyclic shift."""
        if self.shift == 0:
            return None
        img = torch.zeros((1, h, w, 1), device=device)
        cnt = 0
        for hs in (slice(0, -self.ws), slice(-self.ws, -self.shift),
                   slice(-self.shift, None)):
            for wsl in (slice(0, -self.ws), slice(-self.ws, -self.shift),
                        slice(-self.shift, None)):
                img[:, hs, wsl, :] = cnt
                cnt += 1
        mw = window_partition(img, self.ws).view(-1, self.ws * self.ws)
        mask = mw.unsqueeze(1) - mw.unsqueeze(2)
        return mask.masked_fill(mask != 0, -100.0).masked_fill(mask == 0, 0.0)

    def forward(self, x: torch.Tensor, h: int, w: int) -> torch.Tensor:
        """x: [B, H*W, C]"""
        b, l, c = x.shape
        shortcut = x
        x = self.norm1(x).view(b, h, w, c)

        if self.shift > 0:
            x = torch.roll(x, (-self.shift, -self.shift), dims=(1, 2))

        windows = window_partition(x, self.ws).view(-1, self.ws * self.ws, c)
        attended = self.attn(windows, self._attn_mask(h, w, x.device))
        x = window_reverse(attended.view(-1, self.ws, self.ws, c), self.ws, h, w)

        if self.shift > 0:
            x = torch.roll(x, (self.shift, self.shift), dims=(1, 2))

        x = shortcut + x.view(b, h * w, c)
        return x + self.mlp(self.norm2(x))


class RSTB(nn.Module):
    """Residual Swin Transformer Block: several Swin blocks, a conv, a residual.

    The conv reintroduces the locality/translation-equivariance inductive bias
    that pure attention lacks -- important here, where the dataset is small.
    """

    def __init__(self, dim: int, depth: int, num_heads: int, window_size: int,
                 mlp_ratio: float = 2.0):
        super().__init__()
        self.blocks = nn.ModuleList([
            SwinBlock(dim, num_heads, window_size,
                      0 if (i % 2 == 0) else window_size // 2, mlp_ratio)
            for i in range(depth)
        ])
        self.conv = nn.Conv2d(dim, dim, 3, padding=1)

    def forward(self, x: torch.Tensor, h: int, w: int) -> torch.Tensor:
        shortcut = x
        for blk in self.blocks:
            x = blk(x, h, w)
        b, l, c = x.shape
        x = self.conv(x.transpose(1, 2).view(b, c, h, w))
        return x.flatten(2).transpose(1, 2) + shortcut


# ---------------------------------------------------------------------------
# SwinIR
# ---------------------------------------------------------------------------

SWINIR_CONFIG = dict(
    in_channels=1,
    embed_dim=120,
    depths=(6, 6, 6, 6),
    num_heads=(4, 4, 4, 4),
    window_size=8,
    mlp_ratio=2.0,
    upscale=2,
)


class SwinIR(nn.Module):
    def __init__(self, in_channels=1, embed_dim=120, depths=(6, 6, 6, 6),
                 num_heads=(4, 4, 4, 4), window_size=8, mlp_ratio=2.0,
                 upscale=2):
        super().__init__()
        assert len(depths) == len(num_heads)
        self.window_size = window_size
        self.upscale = upscale

        self.shallow = nn.Conv2d(in_channels, embed_dim, 3, padding=1)
        self.layers = nn.ModuleList([
            RSTB(embed_dim, depths[i], num_heads[i], window_size, mlp_ratio)
            for i in range(len(depths))
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.conv_after = nn.Conv2d(embed_dim, embed_dim, 3, padding=1)

        # SR tail: same ICNR-initialised PixelShuffle used by the NAFNet model,
        # which keeps the upsampler from starting with checkerboard structure.
        self.sr_conv = nn.Conv2d(embed_dim, in_channels * upscale ** 2, 3,
                                 padding=1)
        self.ps = nn.PixelShuffle(upscale)
        icnr_init(self.sr_conv.weight, upscale_factor=upscale)
        nn.init.zeros_(self.sr_conv.bias)

        self.grad_checkpoint = False

    def enable_gradient_checkpointing(self, enable: bool = True) -> None:
        """Recompute RSTB activations in backward. Training-only; inference and
        parameters are unaffected."""
        self.grad_checkpoint = bool(enable)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, 1, H, W], any size, values may exceed [0,1] -- not clipped."""
        _, _, h0, w0 = x.shape
        ws = self.window_size
        ph, pw = (ws - h0 % ws) % ws, (ws - w0 % ws) % ws
        if ph or pw:
            x = F.pad(x, (0, pw, 0, ph), mode="reflect")
        _, _, h, w = x.shape

        feat = self.shallow(x)
        shortcut = feat

        tokens = feat.flatten(2).transpose(1, 2)          # [B, H*W, C]
        for layer in self.layers:
            if self.grad_checkpoint and self.training and torch.is_grad_enabled():
                tokens = checkpoint(layer, tokens, h, w, use_reentrant=False)
            else:
                tokens = layer(tokens, h, w)
        tokens = self.norm(tokens)

        feat = tokens.transpose(1, 2).view(x.shape[0], -1, h, w)
        feat = self.conv_after(feat) + shortcut           # long skip

        out = self.ps(self.sr_conv(feat))
        if ph or pw:
            out = out[..., : h0 * self.upscale, : w0 * self.upscale]
        return out.clamp(0.0, 1.0)                        # ONLY clamp
