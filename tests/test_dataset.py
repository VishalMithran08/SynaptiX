"""
tests/test_dataset.py — Dataset loading and split correctness tests.

Run: pytest tests/test_dataset.py -v
"""

import sys, os, tempfile, random
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pytest
import torch


# ---------------------------------------------------------------------------
# Helpers to create a tiny synthetic dataset on disk
# ---------------------------------------------------------------------------

def _make_fake_dataset(tmp_dir: Path, n: int = 20):
    """Create n fake (NoisyLR 128×128, GT 256×256) .npy pairs."""
    noisy_dir = tmp_dir / "NoisyLR"
    gt_dir    = tmp_dir / "GT"
    noisy_dir.mkdir(); gt_dir.mkdir()

    for i in range(n):
        name = f"{i:06d}.npy"
        # NoisyLR: values up to 2.0 (speckle)
        np.save(str(noisy_dir / name),
                (np.random.rand(128, 128) * 2.0).astype(np.float32))
        # GT: clean, always in [0, 1]
        np.save(str(gt_dir / name),
                np.random.rand(256, 256).astype(np.float32))
    return noisy_dir, gt_dir


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_build_splits_sizes():
    """train + val covers all pairs; val_hard is a subset of train."""
    from data.dataset import build_splits
    with tempfile.TemporaryDirectory() as tmp:
        noisy_dir, gt_dir = _make_fake_dataset(Path(tmp))
        train, val, hard  = build_splits(noisy_dir, gt_dir, val_frac=0.10, hard_frac=0.05)

        total = len(train) + len(val) + len(hard)

        assert total == 20, f"Expected 20 total pairs, got {total}"
        assert len(hard) >= 1, "val_hard must have at least 1 sample"

        train_set = {str(p[0]) for p in train}
        val_set   = {str(p[0]) for p in val}
        hard_set  = {str(p[0]) for p in hard}

        assert train_set.isdisjoint(val_set), \
            "train and val overlap"

        assert train_set.isdisjoint(hard_set), \
            "train and val_hard overlap — validation leakage"

        assert val_set.isdisjoint(hard_set), \
            "val and val_hard overlap"


def test_no_val_train_overlap():
    """train and val must not share any file."""
    from data.dataset import build_splits
    with tempfile.TemporaryDirectory() as tmp:
        noisy_dir, gt_dir = _make_fake_dataset(Path(tmp), n=40)
        train, val, _     = build_splits(noisy_dir, gt_dir)
    train_names = {p[0].name for p in train}
    val_names   = {p[0].name for p in val}
    overlap = train_names & val_names
    assert not overlap, f"train/val overlap: {overlap}"


def test_out_of_range_values_preserved():
    """NoisyLR values > 1.0 must survive loading without clipping."""
    from data.dataset import SemiconDataset
    with tempfile.TemporaryDirectory() as tmp:
        noisy_dir, gt_dir = _make_fake_dataset(Path(tmp), n=5)
        pairs = [(noisy_dir / f"{i:06d}.npy", gt_dir / f"{i:06d}.npy") for i in range(5)]
        ds = SemiconDataset(pairs, augment=False, extra_degrade=False)
        noisy, gt = ds[0]
        assert noisy.max().item() > 1.0, \
            "NoisyLR values > 1.0 were clipped — this destroys speckle signal"


def test_gt_in_range():
    """GT must always be in [0, 1] as loaded from disk."""
    from data.dataset import SemiconDataset
    with tempfile.TemporaryDirectory() as tmp:
        noisy_dir, gt_dir = _make_fake_dataset(Path(tmp), n=5)
        pairs = [(noisy_dir / f"{i:06d}.npy", gt_dir / f"{i:06d}.npy") for i in range(5)]
        ds = SemiconDataset(pairs, augment=False, extra_degrade=False)
        for i in range(len(ds)):
            _, gt = ds[i]
            assert gt.min() >= 0.0 and gt.max() <= 1.0


def test_augmentation_preserves_shape():
    """Augmentation must not change tensor shapes."""
    from data.dataset import SemiconDataset
    with tempfile.TemporaryDirectory() as tmp:
        noisy_dir, gt_dir = _make_fake_dataset(Path(tmp), n=5)
        pairs = [(noisy_dir / f"{i:06d}.npy", gt_dir / f"{i:06d}.npy") for i in range(5)]
        ds = SemiconDataset(pairs, augment=True, extra_degrade=True)
        for i in range(len(ds)):
            noisy, gt = ds[i]
            assert noisy.shape == torch.Size([1, 128, 128])
            assert gt.shape    == torch.Size([1, 256, 256])


def test_val_hard_has_highest_noise():
    """val_hard must contain the samples with the highest max pixel values."""
    from data.dataset import build_splits
    with tempfile.TemporaryDirectory() as tmp:
        noisy_dir = Path(tmp) / "NoisyLR"
        gt_dir    = Path(tmp) / "GT"
        noisy_dir.mkdir(); gt_dir.mkdir()

        # Create files with known max values: file i has max = i/10
        for i in range(20):
            arr = np.zeros((128, 128), dtype=np.float32)
            arr[0, 0] = i / 10.0
            np.save(str(noisy_dir / f"{i:06d}.npy"), arr)
            np.save(str(gt_dir    / f"{i:06d}.npy"),
                    np.random.rand(256, 256).astype(np.float32))

        train, _, hard = build_splits(noisy_dir, gt_dir, val_frac=0.10, hard_frac=0.10)

        # Assert inside the `with` block while temp files still exist
        hard_maxes = [np.load(str(nf)).max() for nf, _ in hard]
        assert all(m >= 0.9 for m in hard_maxes), \
            f"val_hard contains low-noise samples: {hard_maxes}"
