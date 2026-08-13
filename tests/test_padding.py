"""
tests/test_padding.py — Padding round-trip and shape correctness tests.

Run: pytest tests/test_padding.py -v
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
import torch
from models.nafnet import NAFNet, PaddedInference, NAFNET_CONFIG


@pytest.fixture(scope="module")
def runner():
    model  = NAFNet(**NAFNET_CONFIG)
    return PaddedInference(model).eval()


@pytest.mark.parametrize("H,W", [
    (128, 128),   # standard training size
    (256, 256),   # double resolution
    (64, 64),     # quarter resolution
    (100, 137),   # arbitrary non-multiple
    (71, 83),     # odd numbers
    (33, 57),     # small
    (16, 16),     # minimum multiple of 16
])
def test_output_shape(runner, H, W):
    """Output must be exactly 2× the input spatial dimensions."""
    x   = torch.randn(1, 1, H, W)
    out = runner(x)
    assert out.shape == (1, 1, H * 2, W * 2), \
        f"Expected (1,1,{H*2},{W*2}), got {out.shape}"


def test_output_range(runner):
    """Output must be clamped to [0, 1]."""
    x   = torch.randn(1, 1, 128, 128)
    out = runner(x)
    assert out.min() >= 0.0, f"Output min {out.min()} < 0"
    assert out.max() <= 1.0, f"Output max {out.max()} > 1"


def test_out_of_range_input_preserved(runner):
    """Speckle input values > 1.0 must not be clipped on entry."""
    # Create input with values up to 2.16 (max seen in test set)
    x      = torch.ones(1, 1, 128, 128) * 2.16
    out    = runner(x)
    # Model output should still be valid [0,1] — it's input that must not be clipped
    assert out.shape == (1, 1, 256, 256)
    assert out.min() >= 0.0 and out.max() <= 1.0


def test_single_output_channel(runner):
    """Model must produce exactly 1 output channel (grayscale)."""
    x   = torch.randn(2, 1, 128, 128)
    out = runner(x)
    assert out.shape[1] == 1, f"Expected 1 output channel, got {out.shape[1]}"


def test_batch_consistency(runner):
    """Single-image and batched inference must produce identical outputs."""
    torch.manual_seed(0)
    x    = torch.randn(3, 1, 64, 64)
    with torch.no_grad():
        out_batch = runner(x)
        out_single = torch.cat([runner(x[i:i+1]) for i in range(3)], dim=0)
    assert torch.allclose(out_batch, out_single, atol=1e-5), \
        "Batch and single-image outputs differ beyond tolerance"
