"""
utils/viz.py — Worst-sample visualisation for validation monitoring.

WHY: The overall PSNR average can hide systematic failure modes (e.g. a
particular speckle level the model never recovers from).  Saving the N
hardest examples as a PNG grid exposes these patterns immediately without
needing to write a separate analysis script.

Output: <save_dir>/viz/iter{N:07d}_worst{k}.png
  Each row: [noisy-upsampled | prediction | ground-truth]
  Rows sorted worst-PSNR first.
"""

import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from typing import List


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _to_uint8(t: torch.Tensor) -> np.ndarray:
    """[1, H, W] float32 → (H, W) uint8 for PIL."""
    return (t.squeeze(0).float().clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)


def _upsample_to(x, hw):
    """
    Resize an image/tensor to (H, W).

    Accepts:
        [H, W]
        [C, H, W]
        [N, C, H, W]

    Returns the same dimensionality as the input.
    """
    original_ndim = x.ndim

    if original_ndim == 2:
        x4 = x.unsqueeze(0).unsqueeze(0)
    elif original_ndim == 3:
        x4 = x.unsqueeze(0)
    elif original_ndim == 4:
        x4 = x
    else:
        raise ValueError(
            f"_upsample_to expected 2D, 3D, or 4D input, "
            f"got shape {tuple(x.shape)}"
        )

    y = F.interpolate(
        x4.float(),
        size=hw,
        mode="bilinear",
        align_corners=False,
    )

    if original_ndim == 2:
        return y.squeeze(0).squeeze(0).to(x.dtype)
    elif original_ndim == 3:
        return y.squeeze(0).to(x.dtype)
    else:
        return y.to(x.dtype)

def save_worst_samples(
    noisy_list,
    pred_list,
    gt_list,
    psnr_list,
    save_dir,
    iteration,
    n_worst=4,
):
    """
    Save the worst validation samples as a grayscale comparison grid.

    Each row contains:
        NoisyLR | Prediction | Ground Truth
    """
    try:
        from PIL import Image
    except ImportError:
        return

    if not psnr_list:
        return

    save_dir = Path(save_dir)

    viz_dir = save_dir / "viz"
    viz_dir.mkdir(parents=True, exist_ok=True)

    order = sorted(
        range(len(psnr_list)),
        key=lambda i: psnr_list[i],
    )[:n_worst]

    rows = []

    for i in order:
        hw = gt_list[i].shape[-2:]

        n_up = _upsample_to(
            noisy_list[i],
            hw,
        )

        noisy_img = _to_uint8(n_up)
        pred_img = _to_uint8(pred_list[i])
        gt_img = _to_uint8(gt_list[i])

        # Convert [1,H,W] -> [H,W]
        if noisy_img.ndim == 3:
            if noisy_img.shape[0] == 1:
                noisy_img = noisy_img[0]
            else:
                noisy_img = np.mean(
                    noisy_img,
                    axis=0,
                ).astype(np.uint8)

        if pred_img.ndim == 3:
            if pred_img.shape[0] == 1:
                pred_img = pred_img[0]
            else:
                pred_img = np.mean(
                    pred_img,
                    axis=0,
                ).astype(np.uint8)

        if gt_img.ndim == 3:
            if gt_img.shape[0] == 1:
                gt_img = gt_img[0]
            else:
                gt_img = np.mean(
                    gt_img,
                    axis=0,
                ).astype(np.uint8)

        if noisy_img.ndim != 2:
            raise ValueError(
                f"Invalid noisy visualization shape: "
                f"{noisy_img.shape}"
            )

        if pred_img.ndim != 2:
            raise ValueError(
                f"Invalid prediction visualization shape: "
                f"{pred_img.shape}"
            )

        if gt_img.ndim != 2:
            raise ValueError(
                f"Invalid GT visualization shape: "
                f"{gt_img.shape}"
            )

        height = min(
            noisy_img.shape[0],
            pred_img.shape[0],
            gt_img.shape[0],
        )

        noisy_img = noisy_img[:height]
        pred_img = pred_img[:height]
        gt_img = gt_img[:height]

        row = np.concatenate(
            [
                noisy_img,
                pred_img,
                gt_img,
            ],
            axis=1,
        )

        rows.append(row)

    if not rows:
        return

    grid = np.concatenate(
        rows,
        axis=0,
    )

    fname = viz_dir / (
        f"iter{iteration:07d}_worst{len(order)}.png"
    )

    Image.fromarray(
        grid.astype(np.uint8),
        mode="L",
    ).save(str(fname))