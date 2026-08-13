"""
Dataset and augmentation pipeline for semiconductor restoration.

Critical correctness guarantees:
- No clipping of NoisyLR input.
- Train/validation/hard-validation sets are non-overlapping.
- Curriculum difficulty is stored in shared memory so persistent workers see
  updates from the main process.
- Photometric transforms are applied consistently to noisy and GT.
- Validation has no stochastic augmentation.
"""

from __future__ import annotations

import math
import multiprocessing as mp
import random
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# ---------------------------------------------------------------------------
# Worker seeding
# ---------------------------------------------------------------------------

def _worker_init_fn(worker_id: int):
    seed = torch.initial_seed() % (2 ** 32)
    random.seed(seed)
    np.random.seed(seed)


# ---------------------------------------------------------------------------
# Shared curriculum state
# ---------------------------------------------------------------------------

def _make_shared_float(value: float):
    """
    One-element shared-memory tensor.

    Unlike a normal Python attribute, this is visible to DataLoader workers.
    """
    t = torch.tensor(
        [float(value)],
        dtype=torch.float32,
    )
    return t.share_memory_()


# ---------------------------------------------------------------------------
# Geometric augmentation
# ---------------------------------------------------------------------------

def _geometric_augment(noisy, gt):
    if random.random() < 0.5:
        noisy = noisy.flip(-1)
        gt = gt.flip(-1)

    if random.random() < 0.5:
        noisy = noisy.flip(-2)
        gt = gt.flip(-2)

    k = random.randint(0, 3)

    if k:
        noisy = torch.rot90(
            noisy,
            k,
            dims=(-2, -1),
        )
        gt = torch.rot90(
            gt,
            k,
            dims=(-2, -1),
        )

    return noisy, gt


# ---------------------------------------------------------------------------
# Paired photometric augmentation
# ---------------------------------------------------------------------------

def _signed_gamma(x, gamma):
    """
    Gamma-like monotonic transform that is valid for negative noisy pixels.

    sign(x) * |x|^gamma

    This preserves the paired relationship between NoisyLR and GT instead of
    changing GT alone.
    """
    return (
        torch.sign(x)
        * torch.abs(x).clamp_min(0.0).pow(gamma)
    )


def _gamma_augment(noisy, gt, difficulty):
    dev = 0.25 * difficulty
    gamma = random.uniform(
        max(0.75, 1.0 - dev),
        min(1.25, 1.0 + dev),
    )

    noisy = _signed_gamma(
        noisy,
        gamma,
    )

    gt = gt.clamp(0.0, 1.0).pow(gamma)

    return noisy, gt


def _intensity_scale(noisy, gt, difficulty):
    dev = 0.15 * difficulty
    scale = random.uniform(
        1.0 - dev,
        1.0 + dev,
    )

    noisy = noisy * scale
    gt = (gt * scale).clamp(0.0, 1.0)

    return noisy, gt


# ---------------------------------------------------------------------------
# CutBlur
# ---------------------------------------------------------------------------

def _cutblur(noisy, difficulty):
    _, h, w = noisy.shape

    max_frac = 0.25 + 0.25 * difficulty

    ph = int(
        h * random.uniform(
            0.10,
            max_frac,
        )
    )

    pw = int(
        w * random.uniform(
            0.10,
            max_frac,
        )
    )

    if ph < 4 or pw < 4:
        return noisy

    y0 = random.randint(0, h - ph)
    x0 = random.randint(0, w - pw)

    ks = random.choice([3, 5, 7])
    pad = ks // 2

    kernel = (
        torch.ones(
            1,
            1,
            ks,
            ks,
            dtype=noisy.dtype,
            device=noisy.device,
        )
        / float(ks * ks)
    )

    blurred = F.conv2d(
        noisy[None],
        kernel,
        padding=pad,
    )[0]

    result = noisy.clone()

    result[
        :,
        y0:y0 + ph,
        x0:x0 + pw,
    ] = blurred[
        :,
        y0:y0 + ph,
        x0:x0 + pw,
    ]

    return result


# ---------------------------------------------------------------------------
# Frequency augmentation
# ---------------------------------------------------------------------------

def _freq_augment(noisy, difficulty):
    if (
        difficulty < 0.3
        or random.random() > 0.5
    ):
        return noisy

    x = noisy.float()

    f = torch.fft.rfft2(
        x,
        norm="ortho",
    )

    h, w = f.shape[-2:]

    # Suppress a contiguous band while keeping the mask symmetric in the
    # vertical frequency dimension.
    axis = random.choice([0, 1])

    if axis == 0:
        width = random.randint(
            1,
            max(
                1,
                int(h * 0.10 * difficulty),
            ),
        )

        center = random.randint(
            0,
            h - 1,
        )

        idx = torch.arange(
            h,
            device=f.device,
        )

        dist = torch.minimum(
            (idx - center).abs(),
            h - (idx - center).abs(),
        )

        mask1 = (
            dist >= width
        ).float().view(
            1, 1, h, 1
        )

    else:
        width = random.randint(
            1,
            max(
                1,
                int(w * 0.15 * difficulty),
            ),
        )

        center = random.randint(
            0,
            w - 1,
        )

        idx = torch.arange(
            w,
            device=f.device,
        )

        dist = (
            idx - center
        ).abs()

        mask1 = (
            dist >= width
        ).float().view(
            1, 1, 1, w
        )

    f = f * mask1

    return torch.fft.irfft2(
        f,
        s=x.shape[-2:],
        norm="ortho",
    ).to(noisy.dtype)


# ---------------------------------------------------------------------------
# Gaussian blur
# ---------------------------------------------------------------------------

BLUR_KERNELS = [3, 5, 7, 9]


def _make_gaussian_kernel(ks, sigma):
    ax = (
        torch.arange(
            ks,
            dtype=torch.float32,
        )
        - ks // 2
    )

    g = torch.exp(
        -(ax ** 2)
        / (2.0 * sigma ** 2)
    )

    g = g / g.sum()

    return (
        g.outer(g)
        .unsqueeze(0)
        .unsqueeze(0)
    )


# ---------------------------------------------------------------------------
# Synthetic extra degradation
# ---------------------------------------------------------------------------

def _extra_degrade(noisy, difficulty):
    ops = []

    sigma_max_speckle = (
        0.10 + 0.20 * difficulty
    )

    sigma_max_gaussian = (
        0.02 + 0.06 * difficulty
    )

    # Blur disabled: the organizer's live Q&A explicitly confirmed degradation
    # is limited to speckle + Gaussian noise + downsampling — no blur. Training
    # against a self-inflicted blur op the real test data never contains
    # wastes capacity and risks over-sharpening artifacts on real inputs.
    # See FINDINGS.md. Kept as a no-op (blur_prob=0.0) rather than deleted so
    # the ops/BLUR_KERNELS machinery stays available if this is ever revisited.
    blur_prob = 0.0

    if random.random() < blur_prob:
        ks = random.choice(
            BLUR_KERNELS
        )
        sigma = random.uniform(
            0.5,
            1.5 + difficulty,
        )
        ops.append(
            ("blur", ks, sigma)
        )

    if random.random() < 0.60:
        sigma = random.uniform(
            0.02,
            sigma_max_speckle,
        )
        ops.append(
            ("speckle", sigma)
        )

    if random.random() < 0.50:
        sigma = random.uniform(
            0.005,
            sigma_max_gaussian,
        )
        ops.append(
            ("gaussian", sigma)
        )

    random.shuffle(ops)

    for op in ops:

        if op[0] == "blur":
            _, ks, sigma = op

            kernel = _make_gaussian_kernel(
                ks,
                sigma,
            )

            noisy = F.conv2d(
                noisy[None],
                kernel,
                padding=ks // 2,
            )[0]

        elif op[0] == "speckle":
            sigma = op[1]

            alpha = 1.0 / max(
                sigma ** 2,
                1e-6,
            )

            gamma_dist = torch.distributions.Gamma(
                concentration=torch.tensor(
                    alpha,
                    dtype=noisy.dtype,
                ),
                rate=torch.tensor(
                    alpha,
                    dtype=noisy.dtype,
                ),
            )

            mult = gamma_dist.sample(
                noisy.shape
            )

            noisy = noisy * mult

        elif op[0] == "gaussian":
            noisy = (
                noisy
                + torch.randn_like(noisy)
                * op[1]
            )

    return noisy


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class SemiconDataset(Dataset):

    def __init__(
        self,
        pairs,
        augment=True,
        extra_degrade=True,
        difficulty=1.0,
        photometric=True,
        cutblur_p=0.3,
        freq_aug_p=0.2,
    ):
        self.pairs = list(pairs)
        self.augment = augment
        self.extra_degrade = extra_degrade
        self.photometric = photometric
        self.cutblur_p = cutblur_p
        self.freq_aug_p = freq_aug_p

        self._difficulty = _make_shared_float(
            difficulty
        )

    def __len__(self):
        return len(self.pairs)

    def set_difficulty(self, value):
        value = float(
            max(0.0, min(1.0, value))
        )
        self._difficulty[0] = value

    @property
    def difficulty(self):
        return float(
            self._difficulty[0].item()
        )

    @staticmethod
    def _load_npy(path):
        arr = np.load(
            str(path),
            mmap_mode="r",
        )

        if arr.dtype != np.float32:
            arr = arr.astype(
                np.float32,
                copy=False,
            )

        tensor = torch.from_numpy(
            np.asarray(arr).copy()
        )

        if tensor.ndim == 2:
            tensor = tensor.unsqueeze(0)

        return tensor.float()

    def __getitem__(self, idx):

        noisy_path, gt_path = self.pairs[idx]

        noisy = self._load_npy(
            noisy_path
        )

        gt = self._load_npy(
            gt_path
        )

        difficulty = self.difficulty

        # Never clip noisy input.
        if (
            self.extra_degrade
            and random.random() < 0.5
        ):
            noisy = _extra_degrade(
                noisy,
                difficulty,
            )

        if (
            self.augment
            and random.random()
            < self.cutblur_p * difficulty
        ):
            noisy = _cutblur(
                noisy,
                difficulty,
            )

        if (
            self.augment
            and random.random()
            < self.freq_aug_p * difficulty
        ):
            noisy = _freq_augment(
                noisy,
                difficulty,
            )

        if (
            self.augment
            and self.photometric
        ):
            if random.random() < 0.3:
                noisy, gt = _gamma_augment(
                    noisy,
                    gt,
                    difficulty,
                )

            if random.random() < 0.3:
                noisy, gt = _intensity_scale(
                    noisy,
                    gt,
                    difficulty,
                )

        if self.augment:
            noisy, gt = _geometric_augment(
                noisy,
                gt,
            )

        # ---------------------------------------------------------
        # Final shape safety check
        # ---------------------------------------------------------
        # Dataset samples must be [C,H,W].
        # Remove an accidental batch dimension [1,C,H,W].
        if noisy.ndim == 4 and noisy.shape[0] == 1:
            noisy = noisy.squeeze(0)

        if gt.ndim == 4 and gt.shape[0] == 1:
            gt = gt.squeeze(0)

        if noisy.ndim != 3:
            raise ValueError(
                f"Invalid NoisyLR shape after augmentation: "
                f"{tuple(noisy.shape)}"
            )

        if gt.ndim != 3:
            raise ValueError(
                f"Invalid GT shape after augmentation: "
                f"{tuple(gt.shape)}"
            )

        return noisy, gt


# ---------------------------------------------------------------------------
# Split builder
# ---------------------------------------------------------------------------

def build_splits(
    noisy_dir,
    gt_dir,
    val_frac=0.10,
    hard_frac=0.05,
    seed=42,
):
    noisy_dir = Path(noisy_dir)
    gt_dir = Path(gt_dir)

    noisy_files = sorted(
        noisy_dir.glob("*.npy")
    )

    if not noisy_files:
        raise FileNotFoundError(
            f"No .npy files in {noisy_dir}"
        )

    pairs = []

    for nf in noisy_files:
        gf = gt_dir / nf.name

        if gf.exists():
            pairs.append(
                (nf, gf)
            )

    if not pairs:
        raise FileNotFoundError(
            "No matching NoisyLR/GT pairs."
        )

    # ---------------------------------------------------------
    # Determine hard-validation samples FIRST.
    # ---------------------------------------------------------
    # This is important: hard samples are selected from ALL
    # available pairs, so random validation cannot accidentally
    # remove the genuinely hardest samples first.
    # ---------------------------------------------------------

    n_hard = max(
        1,
        int(len(pairs) * hard_frac),
    )

    difficulty_scores = []

    for nf, gf in pairs:

        noisy = torch.from_numpy(
            np.asarray(
                np.load(
                    str(nf),
                    mmap_mode="r",
                ),
                dtype=np.float32,
            ).copy()
        ).float()

        gt = torch.from_numpy(
            np.asarray(
                np.load(
                    str(gf),
                    mmap_mode="r",
                ),
                dtype=np.float32,
            ).copy()
        ).float()

        if noisy.ndim == 2:
            noisy = noisy.unsqueeze(0)

        if gt.ndim == 2:
            gt = gt.unsqueeze(0)

        if noisy.ndim != 3 or gt.ndim != 3:
            raise ValueError(
                f"Expected [C,H,W]: "
                f"NoisyLR={tuple(noisy.shape)}, "
                f"GT={tuple(gt.shape)}, "
                f"file={nf.name}"
            )

        # Hardness is based on the largest NoisyLR excursion.
        # NoisyLR is deliberately NOT clipped.
        score = torch.amax(
            torch.abs(noisy)
        ).item()

        if not math.isfinite(score):
            raise ValueError(
                f"Non-finite hard-validation score: {nf.name}"
            )

        difficulty_scores.append(
            (score, nf, gf)
        )

    # Highest score = hardest sample.
    difficulty_scores.sort(
        key=lambda x: x[0],
        reverse=True,
    )

    hard = [
        (nf, gf)
        for _, nf, gf in difficulty_scores[:n_hard]
    ]

    # ---------------------------------------------------------
    # Remove hard-validation samples from the remaining pool.
    # ---------------------------------------------------------

    hard_set = {
        nf
        for nf, _ in hard
    }

    remaining = [
        pair
        for pair in pairs
        if pair[0] not in hard_set
    ]

    # ---------------------------------------------------------
    # Randomly split the remaining samples into train/val.
    # ---------------------------------------------------------

    rng = random.Random(seed)
    rng.shuffle(remaining)

    n_val = max(
        1,
        int(len(pairs) * val_frac),
    )

    # Make sure we don't request more validation samples than
    # are actually available after removing hard samples.
    n_val = min(
        n_val,
        len(remaining) - 1,
    )

    val = remaining[:n_val]
    train = remaining[n_val:]

    return train, val, hard

def make_loaders(
    noisy_dir,
    gt_dir,
    batch_size=16,
    num_workers=8,
    val_frac=0.10,
    hard_frac=0.05,
    extra_degrade=True,
    seed=42,
    difficulty=1.0,
    cutblur_p=0.3,
    freq_aug_p=0.2,
):
    """
    cutblur_p / freq_aug_p are the *base* probabilities of the corresponding
    synthetic degradation; the effective probability is base_p * difficulty.
    Setting freq_aug_p=0.0 disables frequency-band suppression entirely, which
    is required for the controlled Phase-2 diagnostic.

    Validation loaders are unaffected: they are always built with
    augment=False, extra_degrade=False, difficulty=0.0.
    """
    train_pairs, val_pairs, hard_pairs = build_splits(
        noisy_dir,
        gt_dir,
        val_frac,
        hard_frac,
        seed,
    )

    train_ds = SemiconDataset(
        train_pairs,
        augment=True,
        extra_degrade=extra_degrade,
        difficulty=difficulty,
        cutblur_p=cutblur_p,
        freq_aug_p=freq_aug_p,
    )

    val_ds = SemiconDataset(
        val_pairs,
        augment=False,
        extra_degrade=False,
        difficulty=0.0,
    )

    hard_ds = SemiconDataset(
        hard_pairs,
        augment=False,
        extra_degrade=False,
        difficulty=0.0,
    )

    common = dict(
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(
            num_workers > 0
        ),
        prefetch_factor=(
            2 if num_workers > 0 else None
        ),
        worker_init_fn=_worker_init_fn,
    )

    generator = torch.Generator()
    generator.manual_seed(seed)

    return {
        "train": DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            generator=generator,
            drop_last=True,
            **common,
        ),
        "val": DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            **common,
        ),
        "val_hard": DataLoader(
            hard_ds,
            batch_size=batch_size,
            shuffle=False,
            **common,
        ),
        "_train_ds": train_ds,
    }
