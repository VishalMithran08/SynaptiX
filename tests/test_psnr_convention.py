"""psnr_per_image must follow the reporting convention, and must not inherit
the batch-size dependence that pooling MSE introduces."""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.metrics import psnr, psnr_per_image


def test_identical_on_a_single_image():
    """With one image there is nothing to pool, so the two must agree."""
    torch.manual_seed(0)
    a = torch.rand(1, 1, 32, 32)
    b = (a + 0.05 * torch.randn_like(a)).clamp(0, 1)
    assert abs(psnr(a, b) - psnr_per_image(a, b)) < 1e-4


def test_pooling_understates_when_errors_differ():
    """Jensen's inequality: per-image averaging is >= pooled, never below."""
    a = torch.zeros(2, 1, 16, 16)
    b = a.clone()
    b[0] += 0.01          # one easy image
    b[1] += 0.40          # one hard image
    assert psnr_per_image(a, b) > psnr(a, b)


def test_equal_errors_make_them_agree():
    """Equality holds exactly when every image carries identical error."""
    a = torch.zeros(4, 1, 16, 16)
    b = a + 0.1
    assert abs(psnr_per_image(a, b) - psnr(a, b)) < 1e-4


def test_per_image_is_batch_size_invariant():
    """The reported metric must not change with how images are grouped."""
    torch.manual_seed(1)
    a = torch.rand(8, 1, 16, 16)
    b = (a + 0.1 * torch.randn_like(a)).clamp(0, 1)
    whole = psnr_per_image(a, b)
    halves = 0.5 * (psnr_per_image(a[:4], b[:4]) + psnr_per_image(a[4:], b[4:]))
    assert abs(whole - halves) < 1e-4


def test_pooled_psnr_is_not_batch_size_invariant():
    """Documents precisely why psnr() must not be the reported figure."""
    a = torch.zeros(2, 1, 16, 16)
    b = a.clone(); b[0] += 0.01; b[1] += 0.40
    assert abs(psnr(a, b) - 0.5 * (psnr(a[:1], b[:1]) + psnr(a[1:], b[1:]))) > 1.0


def test_accepts_an_unbatched_image():
    a = torch.rand(1, 16, 16)
    b = (a + 0.05 * torch.randn_like(a)).clamp(0, 1)
    assert psnr_per_image(a, b) > 0
