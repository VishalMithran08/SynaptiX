"""
The competition entry point is validated against the published checklist:
positional args, .npy in and out, matching filenames, 2x resolution, grayscale,
finite, inside [0,1]. These tests pin each of those so a refactor cannot
silently break the contract the submission is graded on.
"""
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from run import check_output, find_weights, load_npy  # noqa: E402


# --------------------------------------------------------------------------
# Input handling
# --------------------------------------------------------------------------

def test_loads_plain_2d(tmp_path):
    p = tmp_path / "a.npy"
    np.save(p, np.random.rand(16, 16).astype(np.float32))
    assert load_npy(p).shape == (16, 16)


def test_accepts_hw1(tmp_path):
    """(H, W, 1) is an allowed grayscale layout per the checklist."""
    p = tmp_path / "a.npy"
    np.save(p, np.random.rand(16, 16, 1).astype(np.float32))
    assert load_npy(p).shape == (16, 16)


def test_accepts_1hw(tmp_path):
    p = tmp_path / "a.npy"
    np.save(p, np.random.rand(1, 16, 16).astype(np.float32))
    assert load_npy(p).shape == (16, 16)


def test_rejects_rgb(tmp_path):
    p = tmp_path / "a.npy"
    np.save(p, np.random.rand(16, 16, 3).astype(np.float32))
    assert load_npy(p) is None


def test_unreadable_file_returns_none(tmp_path):
    p = tmp_path / "broken.npy"
    p.write_bytes(b"not an npy file")
    assert load_npy(p) is None


def test_input_is_not_clipped(tmp_path):
    """Speckle pushes inputs outside [0,1]; clipping there would destroy signal."""
    p = tmp_path / "a.npy"
    arr = np.array([[-0.3, 1.9], [0.5, 0.5]], dtype=np.float32)
    np.save(p, arr)
    got = load_npy(p)
    assert got.min() < 0.0 and got.max() > 1.0


# --------------------------------------------------------------------------
# Output validation — one test per checklist rule
# --------------------------------------------------------------------------

def _write(tmp_path, arr):
    p = tmp_path / "out.npy"
    np.save(p, arr)
    return p


def test_valid_output_passes(tmp_path):
    p = _write(tmp_path, np.full((32, 32), 0.5, dtype=np.float32))
    assert check_output(p, (16, 16)) is None


def test_rejects_nan(tmp_path):
    a = np.full((32, 32), 0.5, dtype=np.float32); a[0, 0] = np.nan
    assert "non-finite" in check_output(_write(tmp_path, a), (16, 16))


def test_rejects_inf(tmp_path):
    a = np.full((32, 32), 0.5, dtype=np.float32); a[5, 5] = np.inf
    assert "non-finite" in check_output(_write(tmp_path, a), (16, 16))


def test_rejects_values_above_one(tmp_path):
    a = np.full((32, 32), 0.5, dtype=np.float32); a[1, 1] = 1.5
    assert "outside [0, 1]" in check_output(_write(tmp_path, a), (16, 16))


def test_rejects_negative_values(tmp_path):
    a = np.full((32, 32), 0.5, dtype=np.float32); a[2, 2] = -0.2
    assert "outside [0, 1]" in check_output(_write(tmp_path, a), (16, 16))


def test_rejects_wrong_resolution(tmp_path):
    """Output must be exactly 2x the input in both dimensions."""
    p = _write(tmp_path, np.full((24, 32), 0.5, dtype=np.float32))
    assert "expected (32, 32)" in check_output(p, (16, 16))


def test_rejects_non_float32(tmp_path):
    p = _write(tmp_path, np.full((32, 32), 0.5, dtype=np.float64))
    assert "float32" in check_output(p, (16, 16))


def test_rejects_non_2d(tmp_path):
    p = _write(tmp_path, np.full((32, 32, 3), 0.5, dtype=np.float32))
    assert "2-D" in check_output(p, (16, 16))


# --------------------------------------------------------------------------
# Packaging
# --------------------------------------------------------------------------

def test_weights_are_discoverable():
    """Ships with the repo: no download step may be required."""
    assert find_weights().name == "nafnet160_final.pth"


def test_help_advertises_positional_args():
    """`python run.py <input-dir> <output-dir>` is the graded invocation."""
    out = subprocess.run([sys.executable, str(_ROOT / "run.py"), "--help"],
                         capture_output=True, text=True, cwd=str(_ROOT))
    assert out.returncode == 0
    assert "input_dir" in out.stdout and "output_dir" in out.stdout


def test_missing_input_dir_exits_nonzero(tmp_path):
    out = subprocess.run(
        [sys.executable, str(_ROOT / "run.py"),
         str(tmp_path / "nope"), str(tmp_path / "out")],
        capture_output=True, text=True, cwd=str(_ROOT))
    assert out.returncode != 0


def test_creates_output_directory(tmp_path):
    """The checklist requires the output directory be created if absent."""
    in_dir = tmp_path / "in"
    in_dir.mkdir()
    np.save(in_dir / "x.npy", np.random.rand(16, 16).astype(np.float32))
    out_dir = tmp_path / "made" / "by" / "run"
    assert not out_dir.exists()

    r = subprocess.run(
        [sys.executable, str(_ROOT / "run.py"), str(in_dir), str(out_dir),
         "--tta_views", "1", "--device", "cpu"],
        capture_output=True, text=True, cwd=str(_ROOT))
    assert out_dir.is_dir(), r.stdout + r.stderr
    assert (out_dir / "x.npy").exists(), r.stdout + r.stderr
    assert r.returncode == 0, r.stdout + r.stderr

    got = np.load(out_dir / "x.npy")
    assert got.shape == (32, 32) and got.dtype == np.float32
    assert np.isfinite(got).all() and got.min() >= 0.0 and got.max() <= 1.0
