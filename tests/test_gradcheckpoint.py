"""Gradient checkpointing must change memory, never results.

Checkpointing recomputes activations in backward. If it were wired in wrongly
the model would still run and still train -- just produce different numbers --
so these tests pin the invariants rather than trusting it.
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.nafnet import NAFNET_CONFIG, NAFNet  # noqa: E402


def _tiny():
    cfg = dict(NAFNET_CONFIG)
    cfg["width"] = 8
    return NAFNet(**cfg)


def test_off_by_default():
    assert _tiny().grad_checkpoint is False


def test_forward_identical_in_eval():
    """Inference must be bit-identical: checkpointing is a training-only trick."""
    torch.manual_seed(0)
    m = _tiny().eval()
    x = torch.rand(2, 1, 32, 32)

    with torch.no_grad():
        a = m(x)
        m.enable_gradient_checkpointing(True)
        b = m(x)

    assert torch.equal(a, b), "checkpointing altered inference output"


def test_gradients_match():
    """Same weights, same batch -> same gradients, with and without."""
    torch.manual_seed(0)
    m = _tiny().train()
    x = torch.rand(2, 1, 32, 32)
    gt = torch.rand(2, 1, 64, 64)

    def grads():
        m.zero_grad(set_to_none=True)
        (m(x) - gt).abs().mean().backward()
        return {n: p.grad.detach().clone()
                for n, p in m.named_parameters() if p.grad is not None}

    m.enable_gradient_checkpointing(False)
    g_off = grads()
    m.enable_gradient_checkpointing(True)
    g_on = grads()

    assert set(g_off) == set(g_on)
    assert g_off, "no gradients were produced"
    for n in g_off:
        assert torch.allclose(g_off[n], g_on[n], atol=1e-5, rtol=1e-4), \
            f"gradient mismatch for {n}"


def test_no_grad_mode_is_a_noop():
    """Under no_grad there is nothing to recompute; must not raise."""
    m = _tiny().train()
    m.enable_gradient_checkpointing(True)
    with torch.no_grad():
        out = m(torch.rand(1, 1, 32, 32))
    assert out.shape == (1, 1, 64, 64)


def test_toggle_does_not_change_parameters():
    """Checkpoints stay compatible: no parameters added, removed or renamed."""
    m = _tiny()
    before = set(m.state_dict())
    m.enable_gradient_checkpointing(True)
    assert set(m.state_dict()) == before
