#!/usr/bin/env python3
"""
Ensembling: two free ways to improve a trained model without more training.

  --mode weights      Average the PARAMETERS of several checkpoints into one
                      model. Free at inference (still a single forward pass),
                      but only valid for checkpoints lying in the same loss
                      basin -- here every fine-tune branched from the same
                      width-64 15k checkpoint and ran only a few thousand
                      steps, so they qualify. Widths must match.

  --mode predictions  Average the OUTPUTS of several models. Works across
                      differing widths and training objectives, and is the more
                      promising option when models sit at different points of
                      the perception-distortion curve: their errors are partly
                      uncorrelated, so averaging cancels some of both. Costs N
                      forward passes at inference.

Both are evaluated on the same deterministic 320-image split used everywhere
else in this project, so results are directly comparable.

Examples:
    python tools/ensemble.py --mode predictions --tta \
        --checkpoints checkpoints_w64/model_best.pth \
                      checkpoints_w64_perc01/model_best.pth

    python tools/ensemble.py --mode weights --tta --save avg.pth \
        --checkpoints checkpoints_w64_perc01/phase2_joint_iter00{1,2,3}000.pth
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from data.dataset import make_loaders
from models.nafnet import NAFNet, config_for_state
from utils.metrics import LPIPSMetric, compute_metrics
from utils.tta import tta_predict
from train import monitoring_score

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_report import edge_energy, hf_energy  # noqa: E402


def extract_ema(path: Path) -> dict:
    """Return EMA weights when the checkpoint carries them, else the raw ones."""
    payload = torch.load(str(path), map_location="cpu", weights_only=False)

    if not (isinstance(payload, dict) and "model" in payload):
        return payload                      # bare EMA export

    state = {k: v.clone() for k, v in payload["model"].items()}
    shadow = (payload.get("ema") or {}).get("shadow")
    if shadow:
        for k, v in shadow.items():
            state[k] = v.to(state[k].dtype)
    return state


def build(state: dict, device):
    m = NAFNet(**config_for_state(state))
    m.load_state_dict(state, strict=True)
    return m.to(device).eval()


@torch.no_grad()
def evaluate(models, loader, device, tta: bool, lpips_fn):
    totals, batches = {}, 0

    for noisy, gt in loader:
        noisy, gt = noisy.to(device), gt.to(device).float()

        preds = []
        for m in models:
            preds.append(tta_predict(m, noisy) if tta else m(noisy).float())
        pred = torch.stack(preds).mean(dim=0).clamp(0.0, 1.0)

        metrics = compute_metrics(pred, gt, lpips_fn)
        metrics["edge_energy"] = edge_energy(pred)
        metrics["hf_energy"] = hf_energy(pred)
        for k, v in metrics.items():
            totals[k] = totals.get(k, 0.0) + float(v)
        batches += 1

    means = {k: v / batches for k, v in totals.items()}
    means["monitoring_score"] = monitoring_score(means)
    return means


def report(tag: str, m: dict) -> None:
    lp = f"LPIPS={m['lpips']:.6f} " if "lpips" in m else ""
    print(f"  {tag:<12} L1={m['l1']:.6f} PSNR={m['psnr']:.4f} "
          f"SSIM={m['ssim']:.6f} {lp}edge={m['edge_energy']:.6f} "
          f"hf={m['hf_energy']:.6f} score={m['monitoring_score']:.6f}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", default="D:/semicon/train_new")
    p.add_argument("--checkpoints", nargs="+", required=True)
    p.add_argument("--mode", choices=["weights", "predictions"], required=True)
    p.add_argument("--tta", action="store_true")
    p.add_argument("--lpips", action="store_true")
    p.add_argument("--save", default=None,
                   help="weights mode: where to write the averaged checkpoint.")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    root = Path(args.data_root)
    loaders = make_loaders(
        noisy_dir=root / "train" / "NoisyLR", gt_dir=root / "train" / "GT",
        batch_size=args.batch_size, num_workers=0, val_frac=0.10,
        hard_frac=0.05, extra_degrade=False, seed=args.seed, difficulty=0.0,
    )
    lpips_fn = LPIPSMetric() if args.lpips else None

    states = [extract_ema(Path(c)) for c in args.checkpoints]
    widths = {config_for_state(s)["width"] for s in states}

    print(f"device={device}  mode={args.mode}  TTA={'on' if args.tta else 'off'}")
    for c, s in zip(args.checkpoints, states):
        print(f"  - {c}  (width {config_for_state(s)['width']})")
    print()

    if args.mode == "weights":
        if len(widths) != 1:
            raise SystemExit(
                f"Weight averaging needs identical architectures; got widths {widths}. "
                "Use --mode predictions instead."
            )
        avg = {}
        for k in states[0]:
            if states[0][k].is_floating_point():
                avg[k] = torch.stack([s[k].float() for s in states]).mean(0) \
                              .to(states[0][k].dtype)
            else:
                # Integer buffers (counters etc.) cannot be averaged meaningfully.
                avg[k] = states[0][k].clone()
        models = [build(avg, device)]
        if args.save:
            torch.save(avg, args.save)
            print(f"wrote averaged checkpoint: {args.save}\n")
    else:
        models = [build(s, device) for s in states]

    print("ENSEMBLE")
    report("val", evaluate(models, loaders["val"], device, args.tta, lpips_fn))
    report("hard", evaluate(models, loaders["val_hard"], device, args.tta, lpips_fn))

    if len(states) > 1:
        print("\nINDIVIDUAL (same conditions, for reference)")
        for c, s in zip(args.checkpoints, states):
            m = build(s, device)
            r = evaluate([m], loaders["val"], device, args.tta, lpips_fn)
            report(Path(c).parent.name[:12], r)
            del m
            if device.type == "cuda":
                torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
