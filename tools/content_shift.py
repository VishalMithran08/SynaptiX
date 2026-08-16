#!/usr/bin/env python3
"""
Content-shift generalisation test: does the model work on images it has never
seen the *kind* of?

`tools/ood_probe.py` answers a different question -- it varies degradation
strength on the training content. This varies the content and holds the
degradation fixed at the training recipe, which is what the competition
actually does: "test data will come from different sources."

Method
------
Take clean grayscale images from a source unrelated to the training set,
degrade them with the exact pipeline the training data was built with
(2x bicubic downsample -> multiplicative speckle -> additive Gaussian),
restore them, and score against the clean original.

Because we synthesise the degradation ourselves, the ground truth is exact and
the numbers are directly comparable to the in-distribution validation score.

Usage
-----
    # any directory of images (png/jpg/tif/bmp) or .npy arrays
    python tools/content_shift.py --source_dir /path/to/other/images

    # built-in fallback: scikit-image's sample photographs, no download needed
    python tools/content_shift.py --builtin

External datasets are explicitly permitted by the brief. Nothing here is used
for training -- this is evaluation only.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from evaluate import DEFAULT_WEIGHTS, load_model  # noqa: E402
from utils.metrics import compute_metrics  # noqa: E402
from utils.tta import tta_predict  # noqa: E402

_EXT = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".npy"}


# ---------------------------------------------------------------------------
# Degradation -- mirrors how the supplied training pairs were generated
# ---------------------------------------------------------------------------

def degrade(gt: torch.Tensor, speckle: float, sigma: float,
            gen: torch.Generator) -> torch.Tensor:
    """
    gt: [1, H, W] in [0,1], H and W even.  Returns [1, H/2, W/2].

    Multiplicative speckle then additive Gaussian, matching the statistics
    measured on the supplied NoisyLR/GT pairs. The result is deliberately
    NOT clipped: speckle pushes values outside [0,1] and that is signal.
    """
    lr = F.interpolate(gt[None], scale_factor=0.5, mode="bicubic",
                       align_corners=False, antialias=True)[0]
    noise = torch.randn(lr.shape, generator=gen, dtype=lr.dtype)
    lr = lr * (1.0 + speckle * noise)
    lr = lr + sigma * torch.randn(lr.shape, generator=gen, dtype=lr.dtype)
    return lr


def bicubic_baseline(lr: torch.Tensor, size) -> torch.Tensor:
    """What you get with no model at all -- the floor any result must clear."""
    up = F.interpolate(lr[None].clamp(0, 1), size=size, mode="bicubic",
                       align_corners=False)[0]
    return up.clamp(0, 1)


# ---------------------------------------------------------------------------
# Sources of unseen content
# ---------------------------------------------------------------------------

def load_gray(path: Path) -> torch.Tensor | None:
    """Load any supported file as a single-channel float32 tensor in [0,1]."""
    if path.suffix.lower() == ".npy":
        arr = np.load(path).astype(np.float32)
    else:
        try:
            from PIL import Image
        except ImportError:
            raise SystemExit("Pillow is required to read image files: "
                             "pip install pillow")
        arr = np.asarray(Image.open(path).convert("L"), dtype=np.float32) / 255.0

    if arr.ndim == 3:
        arr = arr.mean(axis=-1) if arr.shape[-1] in (3, 4) else arr[0]
    if arr.ndim != 2 or min(arr.shape) < 64:
        return None

    lo, hi = float(arr.min()), float(arr.max())
    if hi - lo > 1e-6 and (hi > 1.5 or lo < -0.5):
        arr = (arr - lo) / (hi - lo)

    h, w = arr.shape
    arr = arr[: h - h % 2, : w - w % 2]          # even dims for the 2x factor
    return torch.from_numpy(np.ascontiguousarray(arr)).clamp(0, 1)[None]


def builtin_images() -> list[tuple[str, torch.Tensor]]:
    """
    scikit-image's bundled photographs: camera, coins, moon, page, text,
    astronaut, coffee, chelsea. Real photographic content with no relation to
    the training set, and no download -- it ships with the package.
    """
    try:
        from skimage import data
    except ImportError:
        raise SystemExit(
            "--builtin needs scikit-image (pip install scikit-image), or pass "
            "--source_dir with your own images instead."
        )

    out = []
    for name in ("camera", "coins", "moon", "page", "text", "astronaut",
                 "coffee", "chelsea", "brick", "grass", "gravel"):
        fn = getattr(data, name, None)
        if fn is None:
            continue
        arr = np.asarray(fn(), dtype=np.float32)
        if arr.ndim == 3:
            arr = arr.mean(axis=-1)
        if arr.max() > 1.5:
            arr = arr / 255.0
        h, w = arr.shape
        arr = arr[: h - h % 2, : w - w % 2]
        out.append((name, torch.from_numpy(np.ascontiguousarray(arr))
                    .clamp(0, 1)[None]))
    return out


# ---------------------------------------------------------------------------

def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--source_dir", default=None,
                   help="Directory of clean images with content unlike training.")
    p.add_argument("--builtin", action="store_true",
                   help="Use scikit-image's bundled photographs (no download).")
    p.add_argument("--weights", default=str(DEFAULT_WEIGHTS))
    p.add_argument("--tta_views", type=int, default=4, choices=[1, 2, 4, 8])
    p.add_argument("--speckle", type=float, default=0.20,
                   help="Multiplicative speckle std (training recipe: 0.20).")
    p.add_argument("--sigma", type=float, default=0.02,
                   help="Additive Gaussian std (training recipe: 0.02).")
    p.add_argument("--limit", type=int, default=64)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    if not args.source_dir and not args.builtin:
        p.error("pass --source_dir or --builtin")

    if args.builtin:
        items = builtin_images()
    else:
        src = Path(args.source_dir)
        files = sorted(f for f in src.rglob("*")
                       if f.suffix.lower() in _EXT and not f.name.startswith("._"))
        if not files:
            print(f"No readable images in {src}")
            return 1
        items = []
        for f in files[: args.limit]:
            g = load_gray(f)
            if g is not None:
                items.append((f.name, g))

    if not items:
        print("No usable images.")
        return 1

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    runner = load_model(Path(args.weights), dev)
    gen = torch.Generator().manual_seed(args.seed)

    print(f"Device      : {dev}")
    print(f"Images      : {len(items)}  (content unseen in training)")
    print(f"Degradation : speckle {args.speckle}, sigma {args.sigma} "
          f"(the training recipe, unchanged)")
    print(f"TTA         : {args.tta_views} view(s)")
    print()
    print(f"  {'image':<22} {'bicubic':>9} {'model':>9} {'gain':>8} {'SSIM':>8}")
    print("  " + "-" * 60)

    rows = []
    with torch.no_grad():
        for name, gt in items:
            lr = degrade(gt, args.speckle, args.sigma, gen)
            x = lr[None].to(dev)

            pred = (tta_predict(runner, x, views=args.tta_views)
                    if args.tta_views > 1 else runner(x).float().clamp(0, 1))

            g = gt[None].to(dev)
            m = compute_metrics(pred.float(), g)
            base = compute_metrics(bicubic_baseline(lr, gt.shape[-2:])[None].to(dev), g)

            rows.append((name, base["psnr"], m["psnr"], m["ssim"], base["ssim"]))
            print(f"  {name[:22]:<22} {base['psnr']:>9.2f} {m['psnr']:>9.2f} "
                  f"{m['psnr'] - base['psnr']:>+8.2f} {m['ssim']:>8.4f}")

    bp = float(np.mean([r[1] for r in rows]))
    mp = float(np.mean([r[2] for r in rows]))
    ms = float(np.mean([r[3] for r in rows]))
    bs = float(np.mean([r[4] for r in rows]))
    wins = sum(1 for r in rows if r[2] > r[1])

    print("  " + "-" * 60)
    print(f"  {'MEAN':<22} {bp:>9.2f} {mp:>9.2f} {mp - bp:>+8.2f} {ms:>8.4f}")
    print()
    print(f"  PSNR  bicubic {bp:.4f} -> model {mp:.4f}   ({mp - bp:+.4f} dB)")
    print(f"  SSIM  bicubic {bs:.4f} -> model {ms:.4f}   ({ms - bs:+.4f})")
    print(f"  Beats bicubic on {wins}/{len(rows)} images.")
    print()
    print("  The model never saw content like this. Any gain over bicubic is")
    print("  learned restoration transferring, not memorised training content.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
