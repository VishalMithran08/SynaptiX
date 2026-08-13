#!/usr/bin/env python3
"""
Measure the generalisation gap of a checkpoint.

Scores the model on TRAIN pairs and VAL pairs under identical validation
conditions -- no augmentation, difficulty 0, same sample count. The only
difference between the two sets is whether the model was trained on them.

    gap ~ 0     not overfitting; capacity or the objective is the limit
    gap large   overfitting; more capacity or more training will hurt

This matters when scaling capacity: a 183M model on 2720 images is far past the
size where a gap was last measured, and validation improving does not by itself
rule overfitting out -- a model can improve on validation while its train/val
gap widens.

    python tools/gen_gap.py --checkpoints a.pth b.pth [--n 128] [--tta]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from data.dataset import SemiconDataset, build_splits
from models.nafnet import NAFNet, config_for_state
from utils.metrics import compute_metrics
from utils.tta import tta_predict


def load(path: Path, device):
    payload = torch.load(str(path), map_location="cpu", weights_only=False)
    if isinstance(payload, dict) and "model" in payload:
        state = {k: v.clone() for k, v in payload["model"].items()}
        for k, v in (payload.get("ema") or {}).get("shadow", {}).items():
            state[k] = v.to(state[k].dtype)
    else:
        state = payload
    m = NAFNet(**config_for_state(state))
    m.load_state_dict(state, strict=True)
    return m.to(device).eval()


@torch.no_grad()
def score(model, pairs, device, tta):
    ds = SemiconDataset(pairs, augment=False, extra_degrade=False,
                        difficulty=0.0, photometric=False)
    loader = DataLoader(ds, batch_size=8, shuffle=False, num_workers=0)
    tot, n = {}, 0
    for noisy, gt in loader:
        noisy, gt = noisy.to(device), gt.to(device).float()
        pred = tta_predict(model, noisy) if tta else model(noisy).float()
        for k, v in compute_metrics(pred, gt, None).items():
            if isinstance(v, (int, float)):
                tot[k] = tot.get(k, 0.0) + float(v)
        n += 1
    return {k: v / n for k, v in tot.items()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", default="D:/semicon/train_new")
    p.add_argument("--checkpoints", nargs="+", required=True)
    p.add_argument("--labels", nargs="+", default=None)
    p.add_argument("--n", type=int, default=128,
                   help="Images per side. Both sides use the same count.")
    p.add_argument("--tta", action="store_true")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    root = Path(args.data_root)
    train_pairs, val_pairs, _ = build_splits(
        root / "train" / "NoisyLR", root / "train" / "GT",
        val_frac=0.10, hard_frac=0.05, seed=42,
    )
    n = min(args.n, len(val_pairs))
    labels = args.labels or [Path(c).stem for c in args.checkpoints]

    print(f"device={device}  {n} images per side  TTA={'on' if args.tta else 'off'}\n")
    print(f"{'model':<28} {'train L1':>10} {'val L1':>10} {'gap':>8} "
          f"{'train PSNR':>11} {'val PSNR':>10}")
    print("-" * 82)

    for label, ck in zip(labels, args.checkpoints):
        m = load(Path(ck), device)
        tr = score(m, train_pairs[:n], device, args.tta)
        va = score(m, val_pairs[:n], device, args.tta)
        gap = (va["l1"] - tr["l1"]) / va["l1"] * 100
        print(f"{label[:28]:<28} {tr['l1']:>10.6f} {va['l1']:>10.6f} "
              f"{gap:>+7.1f}% {tr['psnr']:>11.4f} {va['psnr']:>10.4f}")
        del m
        if device.type == "cuda":
            torch.cuda.empty_cache()

    print("\nGap is (val - train) / val. Near zero means the model does no better")
    print("on data it trained on than on data it has never seen.")
    print("Reference: the 29.56M width-64 model measured -1.0%, i.e. none.")


if __name__ == "__main__":
    main()
