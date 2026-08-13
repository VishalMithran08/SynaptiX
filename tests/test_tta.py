"""TTA correctness: the transform pairs must be exact inverses, the 8 views must
be distinct, and averaging must be a no-op for a genuinely equivariant model."""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.tta import _D4, _forward, _inverse, tta_predict  # noqa: E402


def test_d4_has_eight_distinct_views():
    assert len(_D4) == 8
    assert len(set(_D4)) == 8


def test_inverse_undoes_forward():
    """A wrong inverse would not raise — it would quietly blur the average."""
    x = torch.randn(2, 1, 16, 16)
    for k, flip in _D4:
        back = _inverse(_forward(x, k, flip), k, flip)
        assert torch.equal(back, x), f"inverse failed for k={k} flip={flip}"


def test_views_are_actually_different():
    """If transforms collapsed to identity, TTA would be a silent no-op."""
    x = torch.arange(16 * 16, dtype=torch.float32).reshape(1, 1, 16, 16)
    views = {_forward(x, k, flip).numpy().tobytes() for k, flip in _D4}
    assert len(views) == 8


def test_tta_preserves_shape_and_scale():
    """A 2x upscaler: transforms must commute with the upscaling."""

    def fake_model(t):
        return torch.nn.functional.interpolate(t, scale_factor=2, mode="nearest")

    x = torch.rand(2, 1, 16, 16)
    out = tta_predict(fake_model, x)
    assert out.shape == (2, 1, 32, 32)


def test_tta_is_identity_for_equivariant_model():
    """Nearest-neighbour upsampling is exactly dihedral-equivariant, so the
    8-view average must reproduce the single-pass result."""

    def fake_model(t):
        return torch.nn.functional.interpolate(t, scale_factor=2, mode="nearest")

    x = torch.rand(2, 1, 16, 16)
    single = fake_model(x).clamp(0, 1)
    assert torch.allclose(tta_predict(fake_model, x), single, atol=1e-6)


def test_tta_averages_a_non_equivariant_model():
    """A model that only responds to one orientation must be softened, not
    passed through — this is what TTA actually buys on a real network."""

    def biased_model(t):
        up = torch.nn.functional.interpolate(t, scale_factor=2, mode="nearest")
        up[..., :, :4] = 1.0  # orientation-specific artefact on the left edge
        return up

    x = torch.zeros(1, 1, 16, 16)
    single = biased_model(x.clone()).clamp(0, 1)
    averaged = tta_predict(biased_model, x)
    assert not torch.allclose(averaged, single)
    assert averaged.max() <= 1.0 and averaged.min() >= 0.0


@pytest.mark.parametrize("shape", [(1, 1, 8, 8), (3, 1, 32, 32)])
def test_tta_handles_batch_shapes(shape):
    def fake_model(t):
        return torch.nn.functional.interpolate(t, scale_factor=2, mode="bilinear")

    out = tta_predict(fake_model, torch.rand(*shape))
    assert out.shape == (shape[0], 1, shape[2] * 2, shape[3] * 2)
