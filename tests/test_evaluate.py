"""
tests/test_evaluate.py — correctness of the submission entry point.

These guard failure modes that produce NO error, only silently wrong output:

  * clipping or /255-scaling a .npy input, which destroys the out-of-range
    speckle values the brief calls "a feature not a bug"
  * writing 8-bit PNGs for float32 .npy inputs, losing precision
  * mishandling a directory containing several image sizes
  * mishandling sizes not divisible by the encoder's /16 stride

Run: python -m pytest tests/test_evaluate.py -v
"""

import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluate import load_image, save_image  # noqa: E402

REPO = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# Loader — out-of-range values must survive
# ---------------------------------------------------------------------------

def test_npy_loader_preserves_out_of_range():
    """Speckle pushes values above 1.0. Clipping them destroys real signal."""
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "t.npy"
        np.save(str(p), np.array([[0.5, 1.5, 2.16]], dtype=np.float32))
        t = load_image(p)

    assert t is not None
    assert t.max().item() > 1.0, "values > 1.0 were clipped"
    assert abs(t.max().item() - 2.16) < 1e-4


def test_npy_loader_preserves_negative():
    """The real data reaches -0.278; negatives must survive too."""
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "t.npy"
        np.save(str(p), np.array([[-0.28, 0.5]], dtype=np.float32))
        t = load_image(p)

    assert t.min().item() < 0.0
    assert abs(t.min().item() + 0.28) < 1e-4


def test_npy_loader_shape():
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "t.npy"
        np.save(str(p), np.zeros((128, 128), dtype=np.float32))
        t = load_image(p)
    assert t.shape == (1, 128, 128)


def test_unreadable_file_returns_none():
    """A corrupt file must be skipped, not crash the whole run."""
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "bad.npy"
        p.write_bytes(b"not a numpy file")
        assert load_image(p) is None


# ---------------------------------------------------------------------------
# Writer — .npy must stay float32
# ---------------------------------------------------------------------------

def test_npy_output_is_float32_npy():
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "out"
        out.mkdir()
        src = Path(tmp) / "img.npy"
        np.save(str(src), np.zeros((4, 4), dtype=np.float32))

        save_image(torch.rand(1, 8, 8), src, out)

        written = out / "img.npy"
        assert written.exists(), "expected .npy output for .npy input"
        arr = np.load(written)
        assert arr.dtype == np.float32
        assert arr.shape == (8, 8)


def test_image_input_writes_png():
    pytest.importorskip("PIL")
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "out"
        out.mkdir()
        src = Path(tmp) / "img.png"
        src.write_bytes(b"")                       # only the suffix matters here
        save_image(torch.rand(1, 8, 8), src, out)
        assert (out / "img.png").exists()


# ---------------------------------------------------------------------------
# End-to-end, exactly as a reviewer would invoke it
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not (REPO / "weights" / "nafnet64_final.pth").exists(),
                    reason="weights not present")
def test_end_to_end_mixed_sizes_cpu():
    """Both scale regimes plus a non-/16 size, in one directory, one run."""
    with tempfile.TemporaryDirectory() as tmp:
        src, out = Path(tmp) / "in", Path(tmp) / "out"
        src.mkdir()
        for name, shape in [("a.npy", (128, 128)),
                            ("b.npy", (256, 256)),
                            ("c.npy", (100, 137))]:
            np.save(str(src / name), np.random.rand(*shape).astype(np.float32))

        r = subprocess.run(
            [sys.executable, str(REPO / "evaluate.py"),
             "--input_dir", str(src), "--output_dir", str(out),
             "--device", "cpu", "--no_tta"],
            capture_output=True, text=True, timeout=900,
        )
        assert r.returncode == 0, f"evaluate.py failed:\n{r.stdout}\n{r.stderr}"

        for name, shape in [("a.npy", (256, 256)),
                            ("b.npy", (512, 512)),
                            ("c.npy", (200, 274))]:
            arr = np.load(out / name)
            assert arr.shape == shape, f"{name}: {arr.shape} != {shape}"
            assert arr.dtype == np.float32
            assert np.isfinite(arr).all()
            assert arr.min() >= 0.0 and arr.max() <= 1.0
