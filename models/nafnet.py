"""
models/nafnet.py — NAFNet-32 with PixelShuffle SR tail.

SINGLE canonical model file imported by BOTH inference.py and train.py.
Never redefine the architecture elsewhere.

Architecture (default config — matches nafnet_base.yaml):
  Input  [B, 1, H, W]   float32 — values may exceed 1.0, DO NOT CLIP
    → InputConv (1 → width=32, 3×3)
    → Encoder 4 stages  ch=[32,64,128,256], blocks=[2,2,4,8], stride-2 between
    → Bottleneck 256 ch, 12 blocks
    → Decoder 4 stages  blocks=[2,2,2,2], skip-cat then 1×1 reduce
    → SR tail: Conv2d(32→4, ICNR-init) + PixelShuffle(2)
    → clamp(0, 1)   ← ONLY clamp in the entire forward pass
  Output [B, 1, H*2, W*2]

Use PaddedInference (end of file) for arbitrary H×W input.
When calling NAFNet directly H and W must both be multiples of 16.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Primitives
# ---------------------------------------------------------------------------

class LayerNormChannel(nn.Module):
    """Per-pixel channel-wise LayerNorm (works at any batch size including 1)."""

    def __init__(self, num_channels: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias   = nn.Parameter(torch.zeros(num_channels))
        self.eps    = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W]
        mean = x.mean(dim=1, keepdim=True)
        var  = x.var (dim=1, keepdim=True, unbiased=False)
        x    = (x - mean) / (var + self.eps).sqrt()
        return x * self.weight[None, :, None, None] + self.bias[None, :, None, None]


class SimpleGate(nn.Module):
    """Split channels in half and multiply — replaces GELU/ReLU, no dead neurons."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


# ---------------------------------------------------------------------------
# NAFBlock
# ---------------------------------------------------------------------------

class NAFBlock(nn.Module):
    """
    One NAFNet residual block.

    Spatial branch:
      LayerNorm → DWConv3×3 (expand 2×) → SimpleGate → ChanAttn → scale → +residual
    Channel branch (FFN):
      LayerNorm → 1×1 conv (expand 2×) → SimpleGate → 1×1 conv → scale → +residual
    """

    def __init__(self, channels: int, dw_expand: int = 2, ffn_expand: int = 2):
        super().__init__()
        dw_ch  = channels * dw_expand
        ffn_ch = channels * ffn_expand

        self.norm1  = LayerNormChannel(channels)
        self.dw     = nn.Conv2d(channels, dw_ch, kernel_size=3, padding=1,
                                groups=channels, bias=True)
        self.sg1    = SimpleGate()                       # dw_ch → channels
        self.ca_avg = nn.AdaptiveAvgPool2d(1)
        self.ca_fc  = nn.Conv2d(channels, channels, 1)  # pointwise, sigmoid gate

        self.norm2  = LayerNormChannel(channels)
        self.ffn1   = nn.Conv2d(channels, ffn_ch, 1)
        self.sg2    = SimpleGate()                       # ffn_ch → channels
        self.ffn2   = nn.Conv2d(channels, channels, 1)

        self.beta  = nn.Parameter(torch.ones(1, channels, 1, 1) * 0.1)
        self.gamma = nn.Parameter(torch.ones(1, channels, 1, 1) * 0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Spatial branch
        y = self.norm1(x)
        y = self.dw(y)
        y = self.sg1(y)
        y = y * torch.sigmoid(self.ca_fc(self.ca_avg(y)))
        x = x + y * self.beta

        # FFN branch
        y = self.norm2(x)
        y = self.ffn1(y)
        y = self.sg2(y)
        y = self.ffn2(y)
        x = x + y * self.gamma

        return x


def _make_stage(channels: int, num_blocks: int) -> nn.Sequential:
    return nn.Sequential(*[NAFBlock(channels) for _ in range(num_blocks)])


# ---------------------------------------------------------------------------
# ICNR initialization — prevents checkerboard in PixelShuffle
# ---------------------------------------------------------------------------

def icnr_init(weight: torch.Tensor, upscale_factor: int = 2) -> torch.Tensor:
    """
    Initialize a Conv2d weight feeding PixelShuffle(r) so all r² sub-pixel
    predictions start equal.  weight shape: [out_ch, in_ch, kH, kW] where
    out_ch = C * r².
    """
    r    = upscale_factor
    base = weight.shape[0] // (r * r)
    sub  = torch.empty(base, *weight.shape[1:])
    nn.init.kaiming_normal_(sub, a=0.0, mode='fan_in', nonlinearity='relu')
    tiled = sub.repeat_interleave(r * r, dim=0)
    with torch.no_grad():
        weight.copy_(tiled)
    return weight


# ---------------------------------------------------------------------------
# Encoder / Decoder helpers
# ---------------------------------------------------------------------------

class _Downsample(nn.Module):
    """Stride-2 conv — halves spatial, doubles channels."""
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.c = nn.Conv2d(in_ch, out_ch, kernel_size=2, stride=2)
    def forward(self, x): return self.c(x)


class _Upsample(nn.Module):
    """PixelShuffle 2× — doubles spatial, halves channels."""
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.c  = nn.Conv2d(in_ch, out_ch * 4, kernel_size=1)
        self.ps = nn.PixelShuffle(2)
    def forward(self, x): return self.ps(self.c(x))


# ---------------------------------------------------------------------------
# NAFNet
# ---------------------------------------------------------------------------

# Default config used by both train.py and inference.py — import this dict.
NAFNET_CONFIG = dict(
    in_channels   = 1,
    width         = 32,
    enc_blocks    = [2, 2, 4, 8],
    middle_blocks = 12,
    dec_blocks    = [2, 2, 2, 2],
)


def config_for_state(state: dict) -> dict:
    """
    NAFNET_CONFIG with `width` recovered from a checkpoint's own tensors.

    Width is a free parameter of the architecture, so a checkpoint trained at a
    different width loads into the default config only as a cryptic shape error.
    `self.inp` is Conv2d(in_channels, width, 3), so its weight is
    [width, in_channels, 3, 3] -- the width is simply its first dimension. This
    lets evaluation and inference stay width-agnostic and keeps every existing
    width-32 checkpoint loadable after wider models are introduced.
    """
    cfg = dict(NAFNET_CONFIG)

    if isinstance(state, dict):
        for key in ("inp.weight", "model.inp.weight"):
            if key in state:
                cfg["width"] = int(state[key].shape[0])
                break

    return cfg


class NAFNet(nn.Module):
    """
    NAFNet-32 + PixelShuffle SR tail.

    Parameters
    ----------
    in_channels   : input image channels (1 for grayscale)
    width         : base channel width (32)
    enc_blocks    : NAFBlocks per encoder stage [2, 2, 4, 8]
    middle_blocks : NAFBlocks in bottleneck (12)
    dec_blocks    : NAFBlocks per decoder stage [2, 2, 2, 2]
    """

    def __init__(
        self,
        in_channels:   int  = 1,
        width:         int  = 32,
        enc_blocks:    list = None,
        middle_blocks: int  = 12,
        dec_blocks:    list = None,
    ):
        super().__init__()
        enc_blocks = enc_blocks or [2, 2, 4, 8]
        dec_blocks = dec_blocks or [2, 2, 2, 2]

        assert len(enc_blocks) == 4 and len(dec_blocks) == 4

        chs = [width * (2 ** i) for i in range(4)]   # [32, 64, 128, 256]

        # ---- Input projection ----
        self.inp = nn.Conv2d(in_channels, width, kernel_size=3, padding=1)

        # ---- Encoder ----
        self.enc   = nn.ModuleList([_make_stage(chs[i], enc_blocks[i]) for i in range(4)])
        self.downs = nn.ModuleList([_Downsample(chs[i], chs[i+1]) for i in range(3)]
                                   + [_Downsample(chs[3], chs[3])])  # 256→256 before bottleneck

        # ---- Bottleneck ----
        self.mid = _make_stage(chs[3], middle_blocks)

        # ---- Decoder ----
        # After upsample, concat with skip (doubles channels), then 1×1 reduce.
        # ups[0]: chs[3]   → chs[3]/2 = 128 ... wait, let's keep chs[3] out for symmetry
        # Standard NAFNet decoder: upsample halves channels, skip is same size → cat→reduce.
        # ups[i] goes from chs[3-i] (or bottleneck dim) to chs[3-i], skip also chs[3-i]
        # After cat: 2*chs[3-i] → reduce to chs[3-i]
        dec_up_in  = [chs[3],   chs[3],   chs[2],   chs[1]]
        dec_up_out = [chs[3],   chs[2],   chs[1],   chs[0]]
        # After upsample + skip cat, reduce 2×out_ch → out_ch
        self.ups    = nn.ModuleList([_Upsample(dec_up_in[i], dec_up_out[i]) for i in range(4)])
        self.reduce = nn.ModuleList([
            nn.Conv2d(dec_up_out[i] * 2, dec_up_out[i], kernel_size=1) for i in range(4)
        ])
        self.dec = nn.ModuleList([_make_stage(dec_up_out[i], dec_blocks[i]) for i in range(4)])

        # ---- SR tail ----
        # Conv2d(32 → 4, ICNR) + PixelShuffle(2) → [B, 1, H*2, W*2]
        self.sr_conv = nn.Conv2d(width, 4, kernel_size=3, padding=1)
        self.ps      = nn.PixelShuffle(2)

        self._init_weights()

    def _init_weights(self):
        icnr_init(self.sr_conv.weight, upscale_factor=2)
        nn.init.zeros_(self.sr_conv.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, 1, H, W]  — H, W must be multiples of 16.
           Values may exceed 1.0. DO NOT CLIP INPUT.
        Returns [B, 1, H*2, W*2] clamped to [0, 1].
        """
        x = self.inp(x)                          # [B, 32, H, W]

        # Encoder — save feature maps before downsampling as skip connections
        skips, feat = [], x
        for i in range(4):
            feat = self.enc[i](feat)
            skips.append(feat)                   # before downsample
            feat = self.downs[i](feat)

        # Bottleneck
        feat = self.mid(feat)                    # [B, 256, H/16, W/16]

        # Decoder — upsample, cat skip, reduce, apply blocks
        for i in range(4):
            feat  = self.ups[i](feat)            # spatial doubles, channels halve
            skip  = skips[3 - i]                 # matching encoder level
            feat  = torch.cat([feat, skip], dim=1)
            feat  = self.reduce[i](feat)
            feat  = self.dec[i](feat)

        # SR tail — no clamp until after PixelShuffle
        feat = self.sr_conv(feat)               # [B, 4, H, W]
        feat = self.ps(feat)                    # [B, 1, H*2, W*2]
        return feat.clamp(0.0, 1.0)             # ← ONLY clamp here


# ---------------------------------------------------------------------------
# PaddedInference — handles any arbitrary H × W
# ---------------------------------------------------------------------------

class PaddedInference(nn.Module):
    """
    Wraps a NAFNet so it accepts input of ANY height and width.

    The raw encoder has 4 stride-2 stages (2^4 = 16), so H and W must each
    be a multiple of 16.  This wrapper:
      1. Pads x symmetrically with reflect-padding to the next multiple of 16.
      2. Runs the NAFNet forward pass.
      3. Crops the output back to exactly (H*2, W*2).

    Usage
    -----
      model  = NAFNet(**NAFNET_CONFIG)
      runner = PaddedInference(model)
      out    = runner(x)   # x can be any [B, 1, H, W]
    """

    STRIDE = 16   # 2^4 encoder downsampling steps

    def __init__(self, model: NAFNet):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        pad_h = (self.STRIDE - H % self.STRIDE) % self.STRIDE
        pad_w = (self.STRIDE - W % self.STRIDE) % self.STRIDE

        if pad_h > 0 or pad_w > 0:
            # reflect padding requires pad < dim; fall back to constant for
            # pathologically tiny inputs (never occurs with real 128×128 data).
            mode = 'reflect' if (pad_h < H and pad_w < W) else 'constant'
            x = F.pad(x, [0, pad_w, 0, pad_h], mode=mode)

        out = self.model(x)                      # [B, 1, (H+pad_h)*2, (W+pad_w)*2]
        return out[:, :, : H * 2, : W * 2]      # crop to exact 2× input size
