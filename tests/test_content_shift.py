"""The content-shift probe must degrade correctly and be honest about scale."""
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools.content_shift import bicubic_baseline, degrade, load_gray


def _gt(h=64, w=64):
    g = torch.rand(1, h, w)
    return g


def test_degrade_halves_resolution():
    lr = degrade(_gt(64, 64), 0.2, 0.02, torch.Generator().manual_seed(0))
    assert lr.shape == (1, 32, 32)


def test_degrade_is_not_clipped():
    """Speckle pushes values outside [0,1]; clipping there would destroy signal
    and would not match how the supplied NoisyLR pairs behave."""
    gen = torch.Generator().manual_seed(0)
    lr = degrade(torch.ones(1, 64, 64), 0.5, 0.1, gen)
    assert lr.max() > 1.0, "degradation must not clamp the high side"


def test_degrade_is_deterministic_given_a_seed():
    gt = _gt()
    a = degrade(gt, 0.2, 0.02, torch.Generator().manual_seed(7))
    b = degrade(gt, 0.2, 0.02, torch.Generator().manual_seed(7))
    assert torch.equal(a, b)


def test_zero_noise_is_a_pure_downsample():
    gt = _gt()
    lr = degrade(gt, 0.0, 0.0, torch.Generator().manual_seed(0))
    ref = torch.nn.functional.interpolate(
        gt[None], scale_factor=0.5, mode="bicubic",
        align_corners=False, antialias=True)[0]
    assert torch.allclose(lr, ref, atol=1e-6)


def test_bicubic_baseline_restores_the_original_size():
    lr = degrade(_gt(64, 96), 0.2, 0.02, torch.Generator().manual_seed(0))
    up = bicubic_baseline(lr, (64, 96))
    assert up.shape == (1, 64, 96)
    assert 0.0 <= float(up.min()) and float(up.max()) <= 1.0


def test_load_gray_rejects_tiny_images(tmp_path):
    import numpy as np
    p = tmp_path / "tiny.npy"
    np.save(p, np.zeros((8, 8), dtype=np.float32))
    assert load_gray(p) is None


def test_load_gray_makes_dimensions_even(tmp_path):
    import numpy as np
    p = tmp_path / "odd.npy"
    np.save(p, np.random.rand(65, 67).astype(np.float32))
    g = load_gray(p)
    assert g.shape == (1, 64, 66)
