#!/usr/bin/env python3
"""
Out-of-distribution robustness probe.

The competition test set contains images "from different sources than the
training data", but every metric in this project so far is measured on the
in-distribution validation split. That split cannot detect a robustness
difference, so a recipe tuned on it may be tuned against generalisation.

This probe scores checkpoints on the SAME validation images put through
progressively heavier synthetic degradation. It does not reproduce the real
distribution shift -- nothing local can -- but degradation strength beyond what
was trained on is a proxy the training pipeline already supports, and it is the
axis the difficulty curriculum actually controls.

Determinism: torch/numpy/random are reseeded identically before every
(checkpoint, difficulty) pass, so each model sees byte-identical degraded inputs
and the comparison is exact.

Usage:
    python ood_probe.py --checkpoints a.pth b.pth --data_root D:/semicon/train_new
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from data.dataset import SemiconDataset, build_splits
from models.nafnet import NAFNet, config_for_state
from utils.metrics import compute_metrics
from utils.tta import tta_predict


def reseed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load(path: Path, device):
    payload = torch.load(str(path), map_location="cpu", weights_only=False)
    state = payload["model"] if isinstance(payload, dict) and "model" in payload else payload
    if isinstance(payload, dict) and payload.get("ema", {}).get("shadow"):
        merged = {k: v.clone() for k, v in state.items()}
        for k, v in payload["ema"]["shadow"].items():
            merged[k] = v.to(merged[k].dtype)
        state = merged
    model = NAFNet(**config_for_state(state))
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()


@torch.no_grad()
def score(model, pairs, difficulty, seed, device, tta):
    reseed(seed)
    ds = SemiconDataset(
        pairs,
        augment=False,
        extra_degrade=difficulty > 0.0,
        difficulty=difficulty,
        photometric=False,
    )
    loader = DataLoader(ds, batch_size=16, shuffle=False, num_workers=0)

    reseed(seed)
    tot, n = {}, 0
    for noisy, gt in loader:
        noisy, gt = noisy.to(device), gt.to(device).float()
        pred = tta_predict(model, noisy) if tta else model(noisy).float()
        m = compute_metrics(pred, gt, None)
        for k, v in m.items():
            if isinstance(v, (int, float)):
                tot[k] = tot.get(k, 0.0) + float(v)
        n += 1
    return {k: v / n for k, v in tot.items()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", default="D:/semicon/train_new")
    p.add_argument("--checkpoints", nargs="+", required=True)
    p.add_argument("--difficulties", nargs="+", type=float,
                   default=[0.0, 0.5, 1.0])
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--tta", action="store_true")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    root = Path(args.data_root)
    _, val_pairs, _ = build_splits(
        root / "train" / "NoisyLR", root / "train" / "GT",
        val_frac=0.10, hard_frac=0.05, seed=42,
    )
    print(f"device={device}  val pairs={len(val_pairs)}  "
          f"TTA={'on' if args.tta else 'off'}  probe seed={args.seed}\n")

    header = "difficulty  " + "".join(f"{Path(c).parent.name[:22]:>24}"
                                      for c in args.checkpoints)
    print(header)
    print(" (0=as-shipped)" + "".join(f"{Path(c).stem[:22]:>24}"
                                      for c in args.checkpoints))
    print("-" * len(header))

    models = [(c, load(Path(c), device)) for c in args.checkpoints]

    for d in args.difficulties:
        cells = []
        for _, m in models:
            r = score(m, val_pairs, d, args.seed, device, args.tta)
            cells.append(f"{r['psnr']:.4f}dB/{r['ssim']:.4f}")
        print(f"{d:>10.2f}  " + "".join(f"{c:>24}" for c in cells))

    print("\nHigher difficulty = degradation harsher than the model trained on.")
    print("A checkpoint that holds up better as difficulty rises is the more")
    print("robust one, whatever its in-distribution score says.")


if __name__ == "__main__":
    main()
