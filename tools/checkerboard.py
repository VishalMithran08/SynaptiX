#!/usr/bin/env python3
"""
Quantify PixelShuffle checkerboard artifacts.

A PixelShuffle(2) decoder produces 4 sub-pixel predictions per output location.
When they disagree, the result is a period-2 alternating pattern -- a
checkerboard. That has an exact signature: over each 2x2 block,
|a - b - c + d| is large when the block alternates and ~0 when it is smooth.

Measured on FLAT regions only (lowest-variance GT patches), because real texture
also produces high 2x2 response and would mask the artifact. In a flat region
the ground truth's own score is the noise floor; a model scoring well above GT
is manufacturing structure that is not there.

Usage:
    python tools/checkerboard.py --checkpoints a.pth b.pth [--tta]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F

from data.dataset import SemiconDataset, build_splits
from models.nafnet import NAFNet, config_for_state
from utils.tta import tta_predict


def checker_response(x: torch.Tensor) -> torch.Tensor:
    """Per-image mean |a-b-c+d| over non-overlapping 2x2 blocks."""
    x = x.float().clamp(0.0, 1.0)
    a = x[..., 0::2, 0::2]
    b = x[..., 0::2, 1::2]
    c = x[..., 1::2, 0::2]
    d = x[..., 1::2, 1::2]
    return (a - b - c + d).abs().mean(dim=(1, 2, 3))


def flatness(x: torch.Tensor) -> torch.Tensor:
    """Per-image variance; low = flat region, where the artifact is visible."""
    return x.float().var(dim=(1, 2, 3))


def load(path: Path, device):
    st = torch.load(str(path), map_location="cpu", weights_only=False)
    if isinstance(st, dict) and "model" in st:
        st = st["model"]
    m = NAFNet(**config_for_state(st))
    m.load_state_dict(st, strict=True)
    return m.to(device).eval()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", default="D:/semicon/train_new")
    p.add_argument("--checkpoints", nargs="+", required=True)
    p.add_argument("--labels", nargs="+", default=None)
    p.add_argument("--flattest", type=int, default=40,
                   help="How many of the flattest validation images to score.")
    p.add_argument("--tta", action="store_true")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    root = Path(args.data_root)
    _, val_pairs, _ = build_splits(
        root / "train" / "NoisyLR", root / "train" / "GT",
        val_frac=0.10, hard_frac=0.05, seed=42,
    )

    # Rank validation GTs by flatness and keep the flattest.
    ds_all = SemiconDataset(val_pairs, augment=False, extra_degrade=False,
                            difficulty=0.0, photometric=False)
    scored = []
    for i in range(len(ds_all)):
        _, gt = ds_all[i]
        scored.append((float(flatness(gt.unsqueeze(0))[0]), i))
    scored.sort()
    idxs = [i for _, i in scored[:args.flattest]]

    ds = SemiconDataset([val_pairs[i] for i in idxs], augment=False,
                        extra_degrade=False, difficulty=0.0, photometric=False)
    loader = torch.utils.data.DataLoader(ds, batch_size=8, shuffle=False)

    labels = args.labels or [Path(c).parent.name for c in args.checkpoints]

    # Ground truth is the reference: it is what "no artifact" actually measures.
    gt_tot, n = 0.0, 0
    for _, gt in loader:
        gt_tot += float(checker_response(gt.to(device)).sum())
        n += gt.shape[0]
    gt_score = gt_tot / n

    print(f"device={device}  flattest {len(idxs)} of {len(val_pairs)} val images"
          f"  TTA={'on' if args.tta else 'off'}\n")
    print(f"{'model':<26} {'checker':>10} {'vs GT':>10}")
    print("-" * 50)
    print(f"{'GROUND TRUTH (reference)':<26} {gt_score:>10.6f} {'1.00x':>10}")

    for label, ck in zip(labels, args.checkpoints):
        m = load(Path(ck), device)
        tot, n = 0.0, 0
        with torch.no_grad():
            for noisy, _ in loader:
                noisy = noisy.to(device)
                out = tta_predict(m, noisy) if args.tta else m(noisy)
                tot += float(checker_response(out).sum())
                n += noisy.shape[0]
        s = tot / n
        print(f"{label:<26} {s:>10.6f} {s / gt_score:>9.2f}x")
        del m
        if device.type == "cuda":
            torch.cuda.empty_cache()

    print("\nLower is smoother. A model well ABOVE ground truth in flat regions")
    print("is generating periodic structure that is not in the target.")


if __name__ == "__main__":
    main()
