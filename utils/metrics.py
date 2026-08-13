"""
utils/metrics.py — Evaluation metrics for training monitoring.

TRAINING / EVALUATION ONLY — never imported by inference.py.

Computes per-batch:
  PSNR   (higher is better)
  SSIM   (higher is better, → 1.0)
  LPIPS  (lower is better) — requires the lpips package
  L1     (lower is better)
  L2     (lower is better)

Also provides composite_score() for best-model tracking (improvement #20).
"""

import math
import torch
import torch.nn.functional as F

try:
    import lpips as _lpips_lib
    _LPIPS_AVAILABLE = True
except ImportError:
    _LPIPS_AVAILABLE = False


# ---------------------------------------------------------------------------
# PSNR
# ---------------------------------------------------------------------------

def psnr(pred: torch.Tensor, target: torch.Tensor, max_val: float = 1.0) -> float:
    """Peak Signal-to-Noise Ratio in dB.  Both tensors in [0, max_val]."""
    mse = F.mse_loss(pred.float(), target.float()).item()
    if mse == 0.0:
        return float("inf")
    return 10.0 * math.log10(max_val ** 2 / mse)


# ---------------------------------------------------------------------------
# SSIM — pure PyTorch, no extra dependencies
# ---------------------------------------------------------------------------

def _gauss_kernel(k: int = 11, sigma: float = 1.5,
                  device=None, dtype=None) -> torch.Tensor:
    coords = torch.arange(k, dtype=dtype, device=device) - k // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    return g.outer(g).unsqueeze(0).unsqueeze(0)


def ssim(pred: torch.Tensor, target: torch.Tensor,
         kernel_size: int = 11, sigma: float = 1.5) -> float:
    """Windowed SSIM, returned as a Python float."""
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    k = _gauss_kernel(kernel_size, sigma, device=pred.device, dtype=pred.dtype)
    p = kernel_size // 2

    x = pred.clamp(0, 1);   y = target.clamp(0, 1)
    mu_x  = F.conv2d(x, k, padding=p)
    mu_y  = F.conv2d(y, k, padding=p)
    sig_x = F.conv2d(x*x, k, padding=p) - mu_x**2
    sig_y = F.conv2d(y*y, k, padding=p) - mu_y**2
    sig_xy= F.conv2d(x*y, k, padding=p) - mu_x*mu_y

    num = (2 * mu_x * mu_y + C1) * (2 * sig_xy + C2)
    den = (mu_x**2 + mu_y**2 + C1) * (sig_x + sig_y + C2)
    return (num / den).mean().item()


# ---------------------------------------------------------------------------
# LPIPS
# ---------------------------------------------------------------------------

class LPIPSMetric:
    """Wrapper around the lpips package.  Raises ImportError if not installed."""

    def __init__(self, net: str = "vgg"):
        if not _LPIPS_AVAILABLE:
            raise ImportError("lpips not installed.  Run: pip install lpips==0.1.4")
        self._fn = _lpips_lib.LPIPS(net=net, verbose=False)
        self._fn.eval()

    def __call__(self, pred: torch.Tensor, target: torch.Tensor) -> float:
        """pred, target: [B, 1, H, W] float32 in [0, 1]."""
        # The VGG backbone is built on CPU. Predictions arrive on whatever
        # device training/eval is using, and a CPU-vs-CUDA mismatch raises --
        # which compute_metrics swallows via `except Exception: pass`, so the
        # metric silently vanishes rather than failing loudly. Follow the input.
        if next(self._fn.parameters()).device != pred.device:
            self._fn = self._fn.to(pred.device)

        # LPIPS expects 3-channel input in [-1, 1]
        p = pred.clamp(0, 1).repeat(1, 3, 1, 1) * 2 - 1
        t = target.clamp(0, 1).repeat(1, 3, 1, 1) * 2 - 1
        with torch.no_grad():
            return self._fn(p, t).mean().item()


# ---------------------------------------------------------------------------
# Composite score for best-model selection (improvement #20)
# WHY: PSNR alone misses perceptual quality; LPIPS alone misses fidelity.
# Composite score = psnr/50 + ssim - 2*lpips treats each metric fairly.
# ---------------------------------------------------------------------------

def composite_score(metrics: dict) -> float:
    """
    Weighted composite of PSNR, SSIM, and LPIPS for best-model tracking.
    Higher is better.  PSNR is normalised by 50 dB (practical ceiling for SR).
    LPIPS is penalised 2× because it is the tiebreaker for close PSNR/SSIM.
    """
    psnr_v  = metrics.get("psnr",  0.0)
    ssim_v  = metrics.get("ssim",  0.0)
    lpips_v = metrics.get("lpips", 1.0)   # 1.0 = worst fallback if unavailable
    return (psnr_v / 50.0) + ssim_v - 2.0 * lpips_v


# ---------------------------------------------------------------------------
# Aggregate metrics helper
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_metrics(
    pred:     torch.Tensor,
    target:   torch.Tensor,
    lpips_fn: "LPIPSMetric | None" = None,
) -> dict:
    """
    Compute all metrics for a batch.
    Returns dict: l1, l2, psnr, ssim, (lpips if fn provided), composite.
    """
    p = pred.float().clamp(0, 1)
    t = target.float().clamp(0, 1)

    result = {
        "l1":   F.l1_loss(p, t).item(),
        "l2":   F.mse_loss(p, t).item(),
        "psnr": psnr(p, t),
        "ssim": ssim(p, t),
    }
    if lpips_fn is not None:
        try:
            result["lpips"] = lpips_fn(p, t)
        except Exception:
            pass

    result["composite"] = composite_score(result)
    return result
