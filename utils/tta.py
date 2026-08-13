"""
Geometric self-ensemble (test-time augmentation) for the restoration model.

The model is equivariant-by-training, not by construction: it was trained with
random flips and rot90, so each of the 8 dihedral views of an input is an
equally valid presentation of the same image. Predicting all 8 and averaging
back in the original frame cancels a large part of the model's orientation-
specific error. This is the standard SR "self-ensemble" (EDSR x8).

The transforms are spatial symmetries, so they commute with the network's 2x
upscaling: transforming a 128x128 input and un-transforming the 256x256 output
lands in the same frame as the untransformed prediction.

Averaging is done in float32 over the *clamped* predictions, matching how single
-pass outputs are consumed elsewhere in the project.
"""

from __future__ import annotations

import torch

# The dihedral group D4: 4 rotations x {no flip, horizontal flip}.
# Listed explicitly so the ordering is stable and reviewable.
_D4 = tuple((k, f) for f in (False, True) for k in (0, 1, 2, 3))


def _forward(x: torch.Tensor, k: int, flip: bool) -> torch.Tensor:
    if flip:
        x = torch.flip(x, dims=[-1])
    if k:
        x = torch.rot90(x, k, dims=[-2, -1])
    return x


def _inverse(x: torch.Tensor, k: int, flip: bool) -> torch.Tensor:
    # Exact inverse of _forward: undo the rotation first, then the flip.
    if k:
        x = torch.rot90(x, -k, dims=[-2, -1])
    if flip:
        x = torch.flip(x, dims=[-1])
    return x


@torch.no_grad()
def tta_predict(model, x: torch.Tensor, clamp: bool = True) -> torch.Tensor:
    """
    Average the model's prediction over the 8 dihedral views of `x`.

    Views are evaluated one at a time and accumulated, so peak memory stays at
    single-pass levels rather than 8x. Cost is 8 forward passes.

    model : callable taking [B, 1, H, W] -> [B, 1, 2H, 2W]
    x     : input batch, NOT clipped (physical out-of-range values preserved)
    """
    total = None

    for k, flip in _D4:
        pred = model(_forward(x, k, flip))
        pred = _inverse(pred.float(), k, flip)
        if clamp:
            pred = pred.clamp(0.0, 1.0)
        total = pred if total is None else total + pred

    return total / len(_D4)


def self_consistency(model, x: torch.Tensor) -> float:
    """
    Mean absolute spread across the 8 views, in the original frame.

    A diagnostic, not a metric: it measures how much the model's answer depends
    on presentation. Large values mean TTA has real disagreement to average
    away; near-zero means the model is already orientation-consistent and TTA
    will change little.
    """
    preds = []
    with torch.no_grad():
        for k, flip in _D4:
            pred = model(_forward(x, k, flip))
            preds.append(_inverse(pred.float(), k, flip).clamp(0.0, 1.0))

    stacked = torch.stack(preds)
    return float((stacked - stacked.mean(dim=0, keepdim=True)).abs().mean())
