"""
tests/test_checkpoint.py — Checkpoint loading diagnostic tests.

A bad checkpoint (shape mismatch, wrong architecture) must produce a clear,
actionable error message — not a raw Python traceback.

Run: pytest tests/test_checkpoint.py -v
"""

import sys, tempfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
import torch
import numpy as np


def _save_random_checkpoint(path: Path):
    from models.nafnet import NAFNet, NAFNET_CONFIG
    model = NAFNet(**NAFNET_CONFIG)
    torch.save(model.state_dict(), str(path))


def _save_wrong_checkpoint(path: Path):
    """Save a state_dict with a mismatched key (simulates architecture change)."""
    from models.nafnet import NAFNet, NAFNET_CONFIG
    model = NAFNet(**NAFNET_CONFIG)
    sd = model.state_dict()
    # Rename a key to simulate architecture mismatch
    wrong_sd = {"wrong_key." + k: v for k, v in sd.items()}
    torch.save(wrong_sd, str(path))


def test_valid_checkpoint_loads():
    """A correctly-saved state_dict must load without error."""
    from evaluate import load_model
    with tempfile.TemporaryDirectory() as tmp:
        ckpt = Path(tmp) / "model.pth"
        _save_random_checkpoint(ckpt)
        runner = load_model(ckpt, device=torch.device("cpu"))
        assert runner is not None


def test_bad_checkpoint_exits_cleanly(capsys):
    """
    A state_dict with wrong keys must call sys.exit(1) and log a human-readable
    error — not raise an unhandled exception with a raw traceback.
    """
    from evaluate import load_model
    with tempfile.TemporaryDirectory() as tmp:
        ckpt = Path(tmp) / "bad.pth"
        _save_wrong_checkpoint(ckpt)

        with pytest.raises(SystemExit) as exc_info:
            load_model(ckpt, device=torch.device("cpu"))

        assert exc_info.value.code == 1, \
            f"Expected exit code 1, got {exc_info.value.code}"


def test_missing_checkpoint_exits_cleanly():
    """A missing model.pth must call sys.exit(1) with a helpful message."""
    from evaluate import load_model

    with pytest.raises(SystemExit) as exc_info:
        load_model(Path("/nonexistent/model.pth"), device=torch.device("cpu"))

    assert exc_info.value.code == 1


def test_wrapped_checkpoint_loads():
    """Checkpoints saved as {'model': state_dict, 'optimizer': ...} must load."""
    from models.nafnet import NAFNet, NAFNET_CONFIG
    from evaluate import load_model

    with tempfile.TemporaryDirectory() as tmp:
        ckpt = Path(tmp) / "wrapped.pth"
        model = NAFNet(**NAFNET_CONFIG)
        torch.save({"model": model.state_dict(), "iteration": 1000}, str(ckpt))
        runner = load_model(ckpt, device=torch.device("cpu"))
        assert runner is not None
