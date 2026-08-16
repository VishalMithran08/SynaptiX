#!/usr/bin/env python3
"""
Deterministic checkpoint evaluator for the Phase-1 / Phase-2 comparison.

Evaluates any checkpoint (bare state_dict OR full training payload, RAW or EMA
weights) on the exact deterministic validation split produced by
data.dataset.build_splits, and reports:

    l1, l2, psnr, ssim, composite       (utils.metrics.compute_metrics)
    monitoring_score                    (train.monitoring_score)
    edge energy   (mean Sobel gradient magnitude of the prediction)
    hf energy     (fraction of FFT power above 0.5 * Nyquist radius)

Metrics are averaged per batch, exactly as train.validate does, so the numbers
are directly comparable with the Phase-1 training logs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from data.dataset import make_loaders
from models.nafnet import NAFNet, NAFNET_CONFIG, config_for_state
from utils.metrics import compute_metrics, LPIPSMetric
from utils.tta import tta_predict
from train import monitoring_score


# ---------------------------------------------------------------------------
# Spectral / edge diagnostics
# ---------------------------------------------------------------------------

_SOBEL_X = torch.tensor(
    [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]
)[None, None]

_SOBEL_Y = torch.tensor(
    [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]
)[None, None]


@torch.no_grad()
def edge_energy(x: torch.Tensor) -> float:
    """Mean Sobel gradient magnitude. Higher = sharper / more edge content."""
    x = x.float().clamp(0.0, 1.0)
    kx = _SOBEL_X.to(x.device)
    ky = _SOBEL_Y.to(x.device)
    gx = F.conv2d(x, kx, padding=1)
    gy = F.conv2d(x, ky, padding=1)
    return float(torch.sqrt(gx * gx + gy * gy + 1e-8).mean())


@torch.no_grad()
def hf_energy(x: torch.Tensor) -> float:
    """
    Fraction of total FFT power located above 0.5 * Nyquist radius.

    Higher = more high-frequency content, i.e. the quantity that rose
    monotonically through Phase 1 while PSNR fell.
    """
    x = x.float().clamp(0.0, 1.0)
    h, w = x.shape[-2:]

    spec = torch.fft.fftshift(
        torch.fft.fft2(x, norm="ortho"),
        dim=(-2, -1),
    )
    power = (spec.real ** 2 + spec.imag ** 2)

    fy = torch.fft.fftshift(torch.fft.fftfreq(h, device=x.device))
    fx = torch.fft.fftshift(torch.fft.fftfreq(w, device=x.device))
    radius = torch.sqrt(
        (fy[:, None] * 2.0) ** 2 + (fx[None, :] * 2.0) ** 2
    )

    mask = (radius >= 0.5).float()

    total = power.sum(dim=(-2, -1)).clamp_min(1e-12)
    high = (power * mask).sum(dim=(-2, -1))

    return float((high / total).mean())


# ---------------------------------------------------------------------------
# Checkpoint loading
# ---------------------------------------------------------------------------

def extract_state(payload, want: str):
    """
    want = "raw" -> canonical optimizer weights
    want = "ema" -> EMA shadow weights

    Bare state_dicts (model_best.pth / model.pth) are already EMA exports and
    are returned unchanged for both requests.
    """
    if not (isinstance(payload, dict) and "model" in payload):
        return payload, "bare_state_dict"

    if want == "raw":
        return payload["model"], "payload.model"

    ema = payload.get("ema")
    if not ema or "shadow" not in ema:
        return None, None

    raw = payload["model"]
    shadow = ema["shadow"]

    merged = {k: v.clone() for k, v in raw.items()}
    for name, tensor in shadow.items():
        if name not in merged:
            raise RuntimeError(f"EMA key absent from model state: {name}")
        merged[name] = tensor.to(merged[name].dtype)

    return merged, "payload.ema.shadow"


def load_model(path: Path, want: str, device):
    payload = torch.load(str(path), map_location="cpu", weights_only=False)
    state, origin = extract_state(payload, want)

    if state is None:
        return None, None

    model = NAFNet(**config_for_state(state))
    model.load_state_dict(state, strict=True)
    return model.to(device).eval(), origin


def checkpoint_info(path: Path) -> dict:
    payload = torch.load(str(path), map_location="cpu", weights_only=False)

    if not (isinstance(payload, dict) and "model" in payload):
        return {
            "kind": "bare state_dict (EMA export)",
            "n_tensors": len(payload),
            "has_optimizer": False,
            "has_scheduler": False,
            "has_scaler": False,
            "has_ema": False,
            "global_iter": None,
            "phase_idx": None,
            "phase_iter": None,
            "best_score": None,
        }

    return {
        "kind": f"full training payload (version={payload.get('version')})",
        "n_tensors": len(payload["model"]),
        "has_optimizer": payload.get("optimizer") is not None,
        "has_scheduler": payload.get("scheduler") is not None,
        "has_scaler": payload.get("scaler") is not None,
        "has_ema": payload.get("ema") is not None,
        "has_rng": payload.get("rng_state") is not None,
        "global_iter": payload.get("global_iter"),
        "phase_idx": payload.get("phase_idx"),
        "phase_iter": payload.get("phase_iter"),
        "best_score": payload.get("best_score"),
    }


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(model, loader, device, amp_dtype=None, tta=False, lpips_fn=None,
             tta_views=8):
    totals = {}
    batches = 0

    for noisy, gt in loader:
        noisy = noisy.to(device)
        gt = gt.to(device).float()

        def forward(x):
            if amp_dtype is None:
                return model(x)
            with torch.amp.autocast(device_type="cuda", dtype=amp_dtype):
                return model(x)

        pred = (tta_predict(forward, noisy, views=tta_views) if tta
                else forward(noisy))

        pred = pred.float()

        if not torch.isfinite(pred).all():
            raise RuntimeError("NaN/Inf in prediction.")

        metrics = compute_metrics(pred, gt, lpips_fn)
        metrics["edge_energy"] = edge_energy(pred)
        metrics["hf_energy"] = hf_energy(pred)

        for k, v in metrics.items():
            totals[k] = totals.get(k, 0.0) + float(v)

        batches += 1

    if batches == 0:
        raise RuntimeError("Empty validation loader.")

    means = {k: v / batches for k, v in totals.items()}
    means["monitoring_score"] = monitoring_score(means)
    return means


def build_loaders(data_root: Path, seed: int, batch_size: int):
    return make_loaders(
        noisy_dir=data_root / "train" / "NoisyLR",
        gt_dir=data_root / "train" / "GT",
        batch_size=batch_size,
        num_workers=0,
        val_frac=0.10,
        hard_frac=0.05,
        extra_degrade=False,
        seed=seed,
        difficulty=0.0,
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", default="d:/semicon/train")
    p.add_argument("--checkpoints", nargs="+", required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument(
        "--amp",
        default="off",
        choices=["off", "fp16", "bf16"],
        help="Evaluate under autocast (train.validate uses autocast).",
    )
    p.add_argument(
        "--tta",
        action="store_true",
        help="Average predictions over the 8 dihedral views (8x inference cost).",
    )
    p.add_argument(
        "--lpips",
        action="store_true",
        help="Also report LPIPS (requires the lpips package).",
    )
    p.add_argument("--tta_views", type=int, default=8, choices=[1, 2, 4, 8],
                   help="Self-ensemble size when --tta is set.")
    p.add_argument("--json_out", default=None)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_dtype = {
        "off": None,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }[args.amp]

    if device.type != "cuda":
        amp_dtype = None

    lpips_fn = LPIPSMetric() if args.lpips else None

    loaders = build_loaders(Path(args.data_root), args.seed, args.batch_size)

    val_ds = loaders["val"].dataset
    hard_ds = loaders["val_hard"].dataset
    train_ds = loaders["_train_ds"]

    print(f"Device        : {device}")
    print(f"AMP           : {args.amp}")
    print(f"Split (seed {args.seed}): train={len(train_ds)} "
          f"val={len(val_ds)} val_hard={len(hard_ds)}")
    print(f"val batches   : {len(loaders['val'])}   "
          f"hard batches: {len(loaders['val_hard'])}")
    print()

    results = {}

    for ckpt_str in args.checkpoints:
        ckpt = Path(ckpt_str)
        if not ckpt.exists():
            print(f"SKIP (missing): {ckpt}")
            continue

        info = checkpoint_info(ckpt)
        print("=" * 78)
        print(f"{ckpt.name}")
        print(f"  contents: {info}")

        entry = {"info": info, "variants": {}}

        for want in ("raw", "ema"):
            model, origin = load_model(ckpt, want, device)
            if model is None:
                continue

            val = evaluate(model, loaders["val"], device, amp_dtype,
                           args.tta, lpips_fn, args.tta_views)
            hard = evaluate(model, loaders["val_hard"], device, amp_dtype,
                            args.tta, lpips_fn, args.tta_views)

            label = "EMA" if origin != "payload.model" else "RAW"
            entry["variants"][label] = {"val": val, "hard": hard,
                                        "weight_source": origin}

            def _lp(m):
                return f"LPIPS={m['lpips']:.6f} " if "lpips" in m else ""

            print(
                f"  [{label:3s}] val : L1={val['l1']:.6f} "
                f"PSNR={val['psnr']:.4f} SSIM={val['ssim']:.6f} {_lp(val)}"
                f"edge={val['edge_energy']:.6f} hf={val['hf_energy']:.6f} "
                f"score={val['monitoring_score']:.6f}"
            )
            print(
                f"  [{label:3s}] hard: L1={hard['l1']:.6f} "
                f"PSNR={hard['psnr']:.4f} SSIM={hard['ssim']:.6f} {_lp(hard)}"
                f"edge={hard['edge_energy']:.6f} hf={hard['hf_energy']:.6f}"
            )

            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()

            if origin == "bare_state_dict":
                break

        results[str(Path(*ckpt.parts[-2:]))] = entry
        print()

    if args.json_out:
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.json_out, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Wrote {args.json_out}")


if __name__ == "__main__":
    main()
