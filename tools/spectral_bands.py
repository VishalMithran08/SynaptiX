#!/usr/bin/env python3
"""
Per-band spectral analysis: WHERE is the model losing detail, and does the
FFT loss actually push there?

Two questions the aggregate "hf energy" number cannot answer:

  1. Is the high-frequency deficit uniform across the band, or concentrated?
     A uniform shortfall suggests a global smoothing bias; a shortfall that
     grows toward Nyquist points at aliased sub-pixel structure specifically.

  2. Where does the FFT loss actually apply pressure? It is an unweighted L1 on
     |FFT|, and natural spectra fall off roughly as 1/f, so low frequencies may
     dominate the loss by orders of magnitude -- in which case a term added to
     fix a high-frequency deficit is doing almost nothing about it.

    python tools/spectral_bands.py --checkpoints model.pth [--tta] [--bands 8]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from data.dataset import make_loaders  # noqa: E402
from models.nafnet import NAFNet, config_for_state  # noqa: E402
from utils.tta import tta_predict  # noqa: E402


def radial_bands(h: int, w: int, n: int, device) -> list[torch.Tensor]:
    """Masks selecting n equal-width radial frequency bands, 0 -> Nyquist."""
    fy = torch.fft.fftshift(torch.fft.fftfreq(h, device=device))
    fx = torch.fft.fftshift(torch.fft.fftfreq(w, device=device))
    r = torch.sqrt((fy[:, None] * 2) ** 2 + (fx[None, :] * 2) ** 2)
    edges = torch.linspace(0, 1.0, n + 1, device=device)
    return [((r >= edges[i]) & (r < edges[i + 1])).float() for i in range(n)]


def band_power(x: torch.Tensor, masks) -> torch.Tensor:
    spec = torch.fft.fftshift(torch.fft.fft2(x.float(), norm="ortho"),
                              dim=(-2, -1))
    power = spec.real ** 2 + spec.imag ** 2
    return torch.stack([(power * m).sum(dim=(-2, -1)).mean() for m in masks])


def band_l1_magnitude(pred, target, masks) -> torch.Tensor:
    """Contribution of each band to the CURRENT unweighted FFT loss."""
    p = torch.abs(torch.fft.fftshift(torch.fft.fft2(pred.float(), norm="ortho"),
                                     dim=(-2, -1)))
    t = torch.abs(torch.fft.fftshift(torch.fft.fft2(target.float(), norm="ortho"),
                                     dim=(-2, -1)))
    diff = (p - t).abs()
    return torch.stack([(diff * m).sum(dim=(-2, -1)).mean() for m in masks])


def load(path: Path, device):
    st = torch.load(str(path), map_location="cpu", weights_only=False)
    if isinstance(st, dict) and "model" in st:
        merged = {k: v.clone() for k, v in st["model"].items()}
        for k, v in (st.get("ema") or {}).get("shadow", {}).items():
            merged[k] = v.to(merged[k].dtype)
        st = merged
    m = NAFNet(**config_for_state(st))
    m.load_state_dict(st, strict=True)
    return m.to(device).eval()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", default="D:/semicon/train_new")
    p.add_argument("--checkpoints", nargs="+", required=True)
    p.add_argument("--bands", type=int, default=8)
    p.add_argument("--tta", action="store_true")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    root = Path(args.data_root)
    loaders = make_loaders(
        noisy_dir=root / "train" / "NoisyLR", gt_dir=root / "train" / "GT",
        batch_size=16, num_workers=0, val_frac=0.10, hard_frac=0.05,
        extra_degrade=False, seed=42, difficulty=0.0,
    )

    masks = None
    for ck in args.checkpoints:
        model = load(Path(ck), device)
        gt_tot = pr_tot = fl_tot = None
        n = 0
        with torch.no_grad():
            for noisy, gt in loaders["val"]:
                noisy, gt = noisy.to(device), gt.to(device).float()
                pred = tta_predict(model, noisy) if args.tta else model(noisy).float()
                if masks is None:
                    masks = radial_bands(*gt.shape[-2:], args.bands, device)
                g, q = band_power(gt, masks), band_power(pred, masks)
                f = band_l1_magnitude(pred, gt, masks)
                gt_tot = g if gt_tot is None else gt_tot + g
                pr_tot = q if pr_tot is None else pr_tot + q
                fl_tot = f if fl_tot is None else fl_tot + f
                n += 1

        gt_tot, pr_tot, fl_tot = gt_tot / n, pr_tot / n, fl_tot / n
        share = fl_tot / fl_tot.sum() * 100

        print(f"\n{'=' * 74}\n{Path(ck).parent.name}/{Path(ck).name}"
              f"   TTA={'on' if args.tta else 'off'}\n{'=' * 74}")
        print(f"{'band (x Nyquist)':>18} {'GT power':>12} {'pred power':>12} "
              f"{'recovered':>10} {'FFT-loss share':>15}")
        print("-" * 74)
        for i in range(args.bands):
            lo, hi = i / args.bands, (i + 1) / args.bands
            pct = 100.0 * float(pr_tot[i] / gt_tot[i]) if gt_tot[i] > 0 else 0.0
            print(f"{lo:>8.2f}-{hi:<8.2f} {float(gt_tot[i]):>12.3e} "
                  f"{float(pr_tot[i]):>12.3e} {pct:>9.1f}% {float(share[i]):>14.1f}%")

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    print("\n'recovered' = prediction power / GT power in that band.")
    print("'FFT-loss share' = how much of the CURRENT unweighted FFT loss each")
    print("band contributes. If it is concentrated at low frequency, the term")
    print("is not pushing where the deficit is.")


if __name__ == "__main__":
    main()
