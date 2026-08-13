#!/usr/bin/env python3
"""
Industrial-grade training pipeline for semiconductor image restoration.

Key guarantees:
- No clipping of degraded input.
- Canonical NAFNet remains uncompiled for checkpoint export.
- torch.compile is only a training execution wrapper.
- EMA tracks the canonical model.
- True per-sample hard-example weighting.
- Restoration losses are evaluated in FP32.
- Curriculum difficulty is shared with DataLoader workers.
- Hard validation samples are excluded from the training split.
- AMP, channels_last, gradient clipping and optional torch.compile.
- Robust checkpoint/resume including RNG state.
  RNG state is restored on resume, but exact DataLoader sample order across
  a resume is not guaranteed because sampler position is not checkpointed.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import logging
import math
import os
import random
import sys
import time
from collections import deque
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from models.nafnet import NAFNet, NAFNET_CONFIG
from data.dataset import make_loaders
from models.losses import RestorationLoss
from utils.metrics import compute_metrics, LPIPSMetric
from utils.ema import EMA
from utils.viz import save_worst_samples


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("semicon.train")


# ---------------------------------------------------------------------------
# Training phases.
# Keep L1 dominant because this is scientific/inspection restoration rather
# than photographic enhancement. Final weights should still be validated by
# ablation rather than assumed to be optimal.
# ---------------------------------------------------------------------------

# ITER_SCALE lets the whole 3-phase schedule be shrunk to fit a real compute
# budget (e.g. a single Colab T4 session/month) without hand-editing every
# phase below. 1.0 = full original schedule. Set e.g. ITER_SCALE=0.2 for a
# ~20% run. warmup_iters is scaled with it so the LR schedule stays sane.
#     ITER_SCALE=0.2 python train.py --data_root ... --amp_dtype fp16
_ITER_SCALE = float(os.environ.get("ITER_SCALE", "1.0"))

# Per-phase keys `hem`, `cutblur_p`, `freq_aug_p` and `extra_degrade` make the
# augmentation/HEM configuration explicit and per-phase instead of global, so a
# controlled experiment can switch one component at a time without editing the
# training loop. Defaults reproduce the historical behaviour.
PHASES = [
    dict(
        name="phase1_warmup",
        # WIDTH-160 SCHEDULE, batch 8.
        #
        # 22.5k x batch 8 = 180k sample-presentations, matching where the
        # width-64 run actually PEAKED (15k x batch 12 = 180k) rather than where
        # its schedule happened to end (27k x 12 = 324k). Width-64 declined over
        # the 144k presentations after that peak, so reproducing them would waste
        # ~4 GPU-hours going backwards.
        #
        # Compressing the schedule also means the cosine anneals to completion AT
        # the peak instead of past it -- the width-64 peak sat at lr ~9.5e-5,
        # mid-schedule, and never got to converge.
        iters=22_500,
        lr=2e-4,
        lr_min=2e-6,
        warmup_iters=2_000,
        w_l1=1.00,
        w_ssim=0.00,
        w_perceptual=0.00,
        w_fft=0.00,
        w_edge=0.00,
        batch_size=16,
        difficulty_start=0.20,
        difficulty_end=0.40,
        hem=True,
        cutblur_p=0.3,
        freq_aug_p=0.2,
        extra_degrade=False,
    ),
    # -----------------------------------------------------------------------
    # phase2_joint — HIGH-FREQUENCY RECOVERY.
    #
    # The earlier "over-sharpening" diagnosis was wrong. It was drawn from a
    # broken extraction holding 359 of 3200 pairs (307 train / 35 val), where a
    # 7.49M model saw each image ~1000 times. Re-running phase1_warmup unchanged
    # on the full 2720/320/160 split reverses the trend entirely: validation
    # improves monotonically 5k->20k (25.5326 -> 25.9484 dB EMA) with HEM ON,
    # freq_aug_p=0.2 and difficulty 0.20->0.40 all active. So difficulty, HEM and
    # frequency augmentation are exonerated and stay ON here.
    #
    # Measured against the ground truth itself, predictions are heavily OVER-
    # smoothed, not over-sharpened:
    #     GT    val  edge=0.309172  hf=0.010758
    #     20k   pred edge=0.232688  hf=0.002542   (75% of GT edge, 24% of GT hf)
    #
    # This phase therefore ADDS high-frequency pressure instead of removing it:
    #   w_fft  = 0.05  (was 0.00 — directly targets the measured hf deficit)
    #   w_edge = 0.05  (was 0.02 — targets the edge deficit)
    #   w_ssim = 0.10  (unchanged; cheap and aligned with the SSIM metric)
    #   w_perceptual = 0.00 (was 0.05). At 0.05 the VGG term measured 1.677624
    #   raw = 0.084 weighted = 63% of total loss, dwarfing L1 at 0.0345. It
    #   optimises feature similarity rather than pixel fidelity and cost 0.020
    #   SSIM in the previous pilot. Reintroduce it later as its own experiment.
    #
    # DIFFICULTY IS NOW CAPPED AT 0.45. Running this phase 0.40->0.60 measured a
    # peak at ~0.48 (phase_iter 25k, PSNR 26.0365) followed by decline at seven
    # consecutive evaluations down to 26.0034 at difficulty 0.60 — with hf energy
    # rising 0.003184 -> 0.003415 while edge stayed flat, i.e. genuine
    # high-frequency over-production. The cosine anneal did not rescue it. 0.45
    # stays below the observed turn. phase3_finetune's difficulty=1.00 is far
    # into the harmful regime and must not be run unchanged.
    #
    # DIFFICULTY IS NOW FIXED, NOT RAMPED. The threshold scales DOWN with model
    # capacity: width-32 turned over at ~0.48, width-64 at ~0.31. Width-64's
    # phase-1 peaked at 15k (difficulty 0.311, PSNR 26.1692) then declined at
    # 20k and 25k as difficulty rose to 0.39; a ramped 0.40->0.45 phase 2 then
    # declined further (0.510699 at 5k -> 0.510241 at 10k). A CPU-only
    # generalisation-gap test on the 15k checkpoint measured -1.0% (train vs
    # val), i.e. NO overfitting at 29.56M params on 2720 images -- so the decline
    # is the curriculum, not memorisation, and regularisation is the wrong lever.
    #
    # The peak also occurred at lr ~9.5e-5, mid-schedule: the model never got to
    # anneal at a difficulty it handles. This phase freezes difficulty at 0.30
    # (where it peaked, with freq_aug still active since that activates at 0.30)
    # and anneals lr fully to 1e-6. Both levers now point the same way.
    # -----------------------------------------------------------------------
    dict(
        # BALANCED VARIANT -- w_perceptual tuned, everything else held fixed.
        #
        # Three runs from the same width-64 15k EMA start decompose the cost of
        # each loss group (320-image validation):
        #
        #   L1 only                          PSNR 26.1692  SSIM 0.786949
        #   + ssim .10 / fft .05 / edge .05  PSNR 26.0915  SSIM 0.788398
        #                                    -> -0.078 dB, +0.0014 SSIM
        #   + perceptual 0.05                PSNR 25.9468  SSIM 0.771647
        #                                    -> -0.145 dB, -0.0168 SSIM
        #
        # So the ssim/fft/edge trio is nearly free and slightly helps SSIM; the
        # perceptual term carries essentially all the distortion cost. It is
        # therefore the only knob worth tuning, and the others stay as they are.
        #
        # LPIPS at w_perceptual=0.05 improved 19.2% (0.371722 -> 0.300230, 64-img
        # subset). Perceptual gains are sublinear in weight -- the first
        # increment buys most of the perceptual improvement, later increments
        # mostly buy distortion damage -- so 0.02 should retain most of the LPIPS
        # gain at ~40% of the PSNR/SSIM cost.
        #
        # Blau & Michaeli (CVPR 2018) proved distortion and perceptual quality
        # cannot both be optimised past a bound, so the aim is not "both best"
        # but the best point on the curve for a metric weighting all three.
        # WIDTH-160: 4.5k x batch 8 = 36k presentations, matching the 3k x batch
        # 12 = 36k that produced the best width-64 checkpoint.
        #
        # SCHEDULE COMPRESSED. All three phase-2 runs -- ramped, fixed
        # difficulty, and perceptual -- peaked at their FIRST validation (5k) and
        # declined thereafter, the perceptual one on all three metrics at once
        # (PSNR 25.9468->25.8694, SSIM 0.771647->0.769992, LPIPS 0.298987->
        # 0.300604) even while the LR annealed. With zero image-level overfitting
        # measured, the fit is to the DEGRADATION distribution: training runs at
        # difficulty 0.30 with HEM/CutBlur/freq-aug while validation is clean, so
        # converging harder onto the training corruptions costs clean accuracy.
        # A 15k schedule therefore spends 10k iterations going backwards and
        # anneals long past peak. 6k lets the cosine finish AT the peak; pair it
        # with --val_every 1000 to locate the true optimum rather than assuming
        # it sits exactly on a 5000 boundary.
        name="phase2_joint",
        iters=4_500,
        lr=1e-4,
        lr_min=1e-6,
        warmup_iters=300,
        w_l1=1.00,
        w_ssim=0.10,
        # 0.05 produced a strong PixelShuffle checkerboard in flat regions and
        # 0.02 a faint one -- both are "artificial patterns", which the brief
        # forbids. 0.01 is the last point worth testing before concluding that
        # perceptual loss needs an architectural fix rather than a weight.
        w_perceptual=0.01,
        w_fft=0.05,
        w_edge=0.05,
        batch_size=16,
        # DIFFICULTY 0.0 -- the one axis never tested. Every run so far trained at
        # difficulty 0.20-0.60, i.e. with synthetic corruption stacked on top of
        # the real degradation, while validation is clean. A model that expects
        # more noise than it receives hedges, and hedging looks like smoothing.
        # Setting 0.0 matches the training distribution to the validation
        # distribution exactly (cutblur_p and freq_aug_p are both scaled by
        # difficulty, so they switch off automatically). Hypothesis: sharper
        # output. Risk: the competition test set is explicitly out-of-
        # distribution, so removing augmentation may cost robustness -- which is
        # why this must be checked with tools/ood_probe.py, not just on the
        # clean split it is tuned for.
        # 0.30 is the winning setting. Difficulty 0.0 was tested and was a null
        # result (identical at 1k, marginally worse at 2k) because cutblur_p and
        # freq_aug_p already scale to ~0.09/0.06 at difficulty 0.30.
        difficulty_start=0.30,
        difficulty_end=0.30,
        hem=True,
        cutblur_p=0.3,
        freq_aug_p=0.2,
        extra_degrade=False,
    ),
    dict(
        name="phase3_finetune",
        iters=50_000,
        lr=5e-5,
        lr_min=5e-7,
        warmup_iters=1_000,
        w_l1=0.70,
        w_ssim=0.15,
        w_perceptual=0.10,
        w_fft=0.08,
        w_edge=0.08,
        batch_size=16,
        difficulty_start=1.00,
        difficulty_end=1.00,
        hem=True,
        cutblur_p=0.3,
        freq_aug_p=0.2,
        extra_degrade=True,
    ),
]

if _ITER_SCALE != 1.0:
    for _p in PHASES:
        _p["iters"] = max(1, int(_p["iters"] * _ITER_SCALE))
        _p["warmup_iters"] = max(1, int(_p["warmup_iters"] * _ITER_SCALE))

VAL_EVERY = max(500, int(5_000 * min(_ITER_SCALE, 1.0)))
SAVE_EVERY = max(500, int(5_000 * min(_ITER_SCALE, 1.0)))
GRAD_CLIP = 1.0

# Keep only the most recent periodic checkpoints PER PHASE.
# latest.pth, model_best.pth, model.pth and model_avg.pth are never rotated.
CHECKPOINTS_PER_PHASE = 3

EMA_DECAY = 0.999
EMA_WARMUP = 2_000

# Project-wide HEM default. Individual phases may override it with the
# per-phase "hem" key above; HEM_ENABLED remains the fallback for any phase
# that does not specify one.
HEM_ENABLED = True
HEM_HARD_FRACTION = 0.50
HEM_HARD_WEIGHT = 2.0


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def seed_everything(seed: int, deterministic: bool) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception as exc:
            log.warning("Deterministic algorithms unavailable: %s", exc)
        log.info("Best-effort deterministic mode enabled.")
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


def capture_rng_state() -> dict:
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: Optional[dict]) -> None:
    if not state:
        return
    try:
        random.setstate(state["python"])
        np.random.set_state(state["numpy"])
        torch.set_rng_state(state["torch"])
        if torch.cuda.is_available() and "cuda" in state:
            torch.cuda.set_rng_state_all(state["cuda"])
    except Exception as exc:
        log.warning("Could not fully restore RNG state: %s", exc)


# ---------------------------------------------------------------------------
# AMP
# ---------------------------------------------------------------------------

def amp_dtype_from_args(args, device: torch.device):
    if device.type != "cuda":
        return torch.float32

    if args.amp_dtype == "bf16":
        if not torch.cuda.is_bf16_supported():
            log.warning("BF16 unsupported; falling back to FP16.")
            return torch.float16
        return torch.bfloat16

    return torch.float16


def autocast_context(device: torch.device, dtype):
    if device.type == "cuda":
        return torch.amp.autocast(
            device_type="cuda",
            dtype=dtype,
            enabled=True,
        )
    return contextlib.nullcontext()


def make_scaler(device: torch.device, amp_dtype):
    if device.type != "cuda" or amp_dtype == torch.bfloat16:
        return torch.amp.GradScaler("cuda", enabled=False)
    return torch.amp.GradScaler("cuda", enabled=True)


# ---------------------------------------------------------------------------
# Scheduler
# ---------------------------------------------------------------------------

def make_scheduler(
    optimizer,
    warmup_iters: int,
    total_iters: int,
    base_lr: float,
    min_lr: float,
):
    min_ratio = min_lr / max(base_lr, 1e-12)

    def lr_lambda(step: int):
        if step < warmup_iters:
            return (step + 1) / max(1, warmup_iters)

        progress = (step - warmup_iters) / max(
            1, total_iters - warmup_iters
        )
        progress = min(max(progress, 0.0), 1.0)

        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_ratio + (1.0 - min_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda,
    )


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def cpu_state_dict(model: nn.Module) -> dict:
    return {
        k: v.detach().cpu().clone()
        for k, v in model.state_dict().items()
    }


def save_checkpoint(
    path: Path,
    raw_model: nn.Module,
    optimizer,
    scheduler,
    scaler,
    ema,
    global_iter: int,
    phase_idx: int,
    phase_iter: int,
    best_score: float,
) -> None:
    payload = {
        "version": 3,
        "model": cpu_state_dict(raw_model),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler else None,
        "scaler": scaler.state_dict() if scaler else None,
        "ema": ema.state_dict() if ema else None,
        "global_iter": global_iter,
        "phase_idx": phase_idx,
        "phase_iter": phase_iter,
        "best_score": best_score,
        "rng_state": capture_rng_state(),
    }

    path.parent.mkdir(parents=True, exist_ok=True)

    # Atomic checkpoint write:
    # write the complete file first, then atomically replace the destination.
    tmp_path = path.with_suffix(
        path.suffix + ".tmp"
    )

    try:
        torch.save(payload, str(tmp_path))
        os.replace(
            str(tmp_path),
            str(path),
        )
    except Exception:
        # Never leave a partial temporary checkpoint behind.
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        except OSError:
            pass
        raise

    # Rotate periodic phase checkpoints only.
    # Protected checkpoints are intentionally excluded.
    if (
        path.name not in {
            "latest.pth",
            "model_best.pth",
            "model.pth",
            "model_avg.pth",
        }
        and "_iter" in path.stem
    ):
        import re

        match = re.match(
            r"^(?P<phase>.+)_iter(?P<iteration>\\d+)$",
            path.stem,
        )

        if match:
            phase_name = match.group("phase")

            periodic = sorted(
                path.parent.glob(
                    f"{phase_name}_iter*.pth"
                ),
                key=lambda p: p.stat().st_mtime,
            )

            while len(periodic) > CHECKPOINTS_PER_PHASE:
                old = periodic.pop(0)
                try:
                    old.unlink()
                    log.info(
                        "Rotated old checkpoint: %s",
                        old,
                    )
                except FileNotFoundError:
                    pass

    log.info("Saved checkpoint: %s", path)


def load_checkpoint(
    path: Path,
    raw_model: nn.Module,
    optimizer=None,
    scaler=None,
    ema=None,
):
    # weights_only=False is required: the payload carries optimizer/scheduler
    # state and pickled RNG state, which the PyTorch >=2.6 default
    # (weights_only=True) refuses to unpickle. These checkpoints are produced
    # by this repository, so they are trusted.
    state = torch.load(
        str(path),
        map_location="cpu",
        weights_only=False,
    )

    if not isinstance(state, dict) or "model" not in state:
        raw_model.load_state_dict(state, strict=True)
        return {
            "global_iter": 0,
            "phase_idx": 0,
            "phase_iter": 0,
            "best_score": -float("inf"),
            "scheduler": None,
            "rng_state": None,
        }

    raw_model.load_state_dict(state["model"], strict=True)

    if optimizer is not None and state.get("optimizer") is not None:
        optimizer.load_state_dict(state["optimizer"])

    if scaler is not None and state.get("scaler") is not None:
        scaler.load_state_dict(state["scaler"])

    if ema is not None and state.get("ema") is not None:
        ema.load_state_dict(state["ema"])

    return {
        "global_iter": int(state.get("global_iter", 0)),
        "phase_idx": int(state.get("phase_idx", 0)),
        "phase_iter": int(state.get("phase_iter", 0)),
        "best_score": float(state.get("best_score", -float("inf"))),
        "scheduler": state.get("scheduler"),
        "rng_state": state.get("rng_state"),
    }


def average_model_checkpoints(checkpoint_paths):
    """Average canonical model weights from final-phase checkpoints only."""
    if not checkpoint_paths:
        raise ValueError("No checkpoints supplied.")

    states = []
    for path in checkpoint_paths:
        payload = torch.load(
            str(path),
            map_location="cpu",
            weights_only=False,
        )
        state = payload["model"] if (
            isinstance(payload, dict) and "model" in payload
        ) else payload
        states.append(state)

    keys = set(states[0].keys())
    for state in states[1:]:
        if set(state.keys()) != keys:
            raise RuntimeError(
                "Cannot average checkpoints with different state_dict keys."
            )

    averaged = {}
    for key in sorted(keys):
        values = [state[key] for state in states]
        if values[0].dtype.is_floating_point:
            averaged[key] = torch.stack(
                [v.float() for v in values],
                dim=0,
            ).mean(dim=0)
        else:
            averaged[key] = values[0]

    return averaged


def export_ema_model(
    raw_model: nn.Module,
    ema: EMA,
    path: Path,
) -> None:
    ema.apply_shadow()
    try:
        state = cpu_state_dict(raw_model)
    finally:
        ema.restore()

    torch.save(state, str(path))
    log.info("Exported canonical EMA model: %s", path)


# ---------------------------------------------------------------------------
# Hard-example mining
# ---------------------------------------------------------------------------

def hem_weights(
    per_sample_l1: torch.Tensor,
    hard_fraction: float = HEM_HARD_FRACTION,
    hard_weight: float = HEM_HARD_WEIGHT,
) -> torch.Tensor:
    b = per_sample_l1.shape[0]
    if b <= 1:
        return torch.ones_like(per_sample_l1)

    k = max(1, math.ceil(b * hard_fraction))
    _, idx = torch.topk(
        per_sample_l1.detach(),
        k=k,
        largest=True,
    )

    weights = torch.ones_like(per_sample_l1)
    weights.scatter_(0, idx, float(hard_weight))

    # Keep mean loss scale unchanged.
    return weights / weights.mean()


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

@torch.no_grad()
def validate(
    model,
    loader,
    device,
    amp_dtype,
    tag,
    lpips_fn,
    save_dir,
    iteration,
    n_viz=8,
):
    model.eval()

    totals = {}
    batches = 0

    viz_noisy, viz_pred, viz_gt, viz_psnr = [], [], [], []

    for noisy, gt in loader:
        noisy = noisy.to(device, non_blocking=True)
        gt = gt.to(device, non_blocking=True)

        with autocast_context(device, amp_dtype):
            pred = model(noisy)

        pred = pred.float()
        gt = gt.float()

        if not torch.isfinite(pred).all():
            raise FloatingPointError(f"NaN/Inf during {tag} validation.")

        metrics = compute_metrics(pred, gt, lpips_fn)

        for k, v in metrics.items():
            totals[k] = totals.get(k, 0.0) + float(v)

        if batches < 50:
            from utils.metrics import psnr

            for b in range(pred.shape[0]):
                p = float(
                    psnr(
                        pred[b:b + 1].cpu(),
                        gt[b:b + 1].cpu(),
                    )
                )
                viz_noisy.append(noisy[b:b + 1].cpu())
                viz_pred.append(pred[b:b + 1].cpu())
                viz_gt.append(gt[b:b + 1].cpu())
                viz_psnr.append(p)

        batches += 1

    if batches == 0:
        raise RuntimeError(f"Empty validation loader: {tag}")

    means = {k: v / batches for k, v in totals.items()}

    log.info(
        "[%s] %s",
        tag,
        "  ".join(f"{k}={v:.6f}" for k, v in means.items()),
    )

    if save_dir is not None and viz_psnr:
        save_worst_samples(
            viz_noisy,
            viz_pred,
            viz_gt,
            viz_psnr,
            save_dir,
            iteration,
            n_worst=n_viz,
        )

    return means


def monitoring_score(metrics: Dict[str, float]) -> float:
    """
    Internal monitoring score only.
    It is NOT claimed to be the official hackathon metric.
    """

    psnr = float(metrics.get("psnr", 0.0))
    ssim = float(metrics.get("ssim", 0.0))
    lpips = float(metrics.get("lpips", 1.0))

    psnr_norm = min(max(psnr / 50.0, 0.0), 1.0)
    lpips_score = 1.0 - min(max(lpips, 0.0), 1.0)

    return (
        0.45 * psnr_norm
        + 0.35 * ssim
        + 0.20 * lpips_score
    )


def benchmark_layouts(device, steps=25, batch_size=4):
    """Benchmark NCHW vs channels_last without changing NAFNet internals."""
    if device.type != "cuda":
        log.info("Layout benchmark skipped: CUDA unavailable.")
        return

    import copy

    model_nchw = NAFNet(**NAFNET_CONFIG).to(device).eval()
    model_cl = copy.deepcopy(model_nchw).to(
        device,
        memory_format=torch.channels_last,
    )

    x_nchw = torch.randn(
        batch_size, 1, 128, 128, device=device
    )
    x_cl = x_nchw.to(memory_format=torch.channels_last)

    def run(model, x):
        torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.no_grad():
            for _ in range(steps):
                model(x)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        speed = batch_size * steps / max(elapsed, 1e-9)
        memory = torch.cuda.max_memory_allocated() / (1024 ** 3)
        return speed, memory

    torch.cuda.reset_peak_memory_stats()
    nchw_speed, nchw_mem = run(model_nchw, x_nchw)

    torch.cuda.reset_peak_memory_stats()
    cl_speed, cl_mem = run(model_cl, x_cl)

    log.info(
        "Layout benchmark: NCHW %.2f img/s %.2f GB | "
        "channels_last %.2f img/s %.2f GB",
        nchw_speed,
        nchw_mem,
        cl_speed,
        cl_mem,
    )

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def train(args):
    if args.batch_size is not None:
        for _p in PHASES:
            _p["batch_size"] = args.batch_size

    seed_everything(args.seed, args.deterministic)

    # TF32 improves FP32 matmul/convolution throughput on supported
    # NVIDIA GPUs (e.g. A100/H100). Never enable it in deterministic mode.
    if (
        not args.deterministic
        and torch.cuda.is_available()
    ):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

        log.info(
            "TF32 enabled for non-deterministic CUDA training."
        )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    if device.type == "cuda":
        log.info("GPU: %s", torch.cuda.get_device_name(0))
    else:
        log.warning("CUDA unavailable; CPU training.")

    amp_dtype = amp_dtype_from_args(args, device)

    log.info("AMP dtype: %s", amp_dtype)

    if args.max_iters is not None:
        log.info("Global iteration cap: %d", args.max_iters)

    if args.max_iters is not None and args.max_iters <= 0:
        raise ValueError("--max_iters must be a positive integer.")

    if args.benchmark_layouts:
        benchmark_layouts(device)

        if args.benchmark_only:
            log.info("Benchmark-only mode requested; exiting before training.")
            return

    elif args.benchmark_only:
        raise ValueError("--benchmark_only requires --benchmark_layouts.")

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Every phase-2 run peaked at its first validation, so a fixed 5000-step
    # interval is too coarse to locate the real optimum on a short schedule.
    val_every = int(args.val_every) if args.val_every else VAL_EVERY
    save_every = int(args.val_every) if args.val_every else SAVE_EVERY

    # Mirror the console log into the run directory as UTF-8. Shell redirection
    # on Windows PowerShell rewrites the stream as UTF-16 and wraps stderr in
    # NativeCommandError noise, which makes the resulting file painful to read
    # back; writing it here keeps every run's log plain and self-contained.
    file_handler = logging.FileHandler(
        str(save_dir / "train.log"),
        mode="a",
        encoding="utf-8",
    )
    file_handler.setFormatter(
        logging.Formatter(
            "%(asctime)s  %(levelname)-7s  %(message)s",
            datefmt="%H:%M:%S",
        )
    )
    logging.getLogger().addHandler(file_handler)

    # ---------------------------------------------------------------
    # TensorBoard
    # ---------------------------------------------------------------
    writer = None
    try:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(str(save_dir / "tb"))
    except ImportError:
        log.warning("TensorBoard not installed.")

    # ---------------------------------------------------------------
    # Canonical raw model
    # ---------------------------------------------------------------
    # `--width` overrides only the base channel count; block layout is unchanged.
    # Every later NAFNet construction in this function reuses `model_config`, so
    # the verifiers and the summary describe the model actually trained.
    model_config = dict(NAFNET_CONFIG)
    model_config["width"] = int(args.width)
    if model_config["width"] != NAFNET_CONFIG["width"]:
        log.info(
            "Model width overridden: %d -> %d",
            NAFNET_CONFIG["width"],
            model_config["width"],
        )

    raw_model = NAFNet(**model_config).to(device)

    if args.grad_checkpoint:
        # Recompute block activations in backward instead of storing them.
        # Costs ~30% throughput and buys a large drop in activation memory,
        # which is what makes wide models trainable on a small card.
        raw_model.enable_gradient_checkpointing(True)
        log.info("Gradient checkpointing ENABLED (slower per step, less memory).")

    if device.type == "cuda":
        raw_model = raw_model.to(
            memory_format=torch.channels_last
        )

    log.info(
        "Parameters: %.2f M",
        sum(p.numel() for p in raw_model.parameters()) / 1e6,
    )

    # ---------------------------------------------------------------
    # EMA
    # ---------------------------------------------------------------
    ema = EMA(
        raw_model,
        decay=EMA_DECAY,
        warmup=EMA_WARMUP,
    )

    # ---------------------------------------------------------------
    # Optimizer / scaler
    # ---------------------------------------------------------------
    optimizer = torch.optim.AdamW(
        raw_model.parameters(),
        lr=PHASES[0]["lr"],
        weight_decay=1e-4,
        betas=(0.9, 0.999),
        eps=1e-8,
    )

    scaler = make_scaler(device, amp_dtype)

    # ---------------------------------------------------------------
    # Resume
    # ---------------------------------------------------------------
    global_iter = 0
    resume_phase = 0
    resume_phase_iter = 0
    best_score = -float("inf")
    resume_scheduler_state = None

    if args.resume:
        info = load_checkpoint(
            Path(args.resume),
            raw_model,
            optimizer=optimizer,
            scaler=scaler,
            ema=ema,
        )

        global_iter = info["global_iter"]
        resume_phase = info["phase_idx"]
        resume_phase_iter = info["phase_iter"]
        best_score = info["best_score"]
        resume_scheduler_state = info["scheduler"]

        restore_rng_state(info["rng_state"])

        log.info(
            "Resumed: global=%d phase=%d phase_iter=%d",
            global_iter,
            resume_phase,
            resume_phase_iter,
        )

        log.warning(
            "Resume restores Python/NumPy/Torch/CUDA RNG states, but exact "
            "DataLoader sample order across a resume is NOT guaranteed."
        )

    # ---------------------------------------------------------------
    # Optional compile wrapper.
    #
    # raw_model remains the canonical model for EMA/checkpoints.
    # ---------------------------------------------------------------
    train_model = raw_model

    if args.compile and device.type == "cuda":
        try:
            train_model = torch.compile(
                raw_model,
                mode=args.compile_mode,
            )
            log.info(
                "torch.compile enabled: %s",
                args.compile_mode,
            )
        except Exception as exc:
            log.warning(
                "torch.compile failed; using eager mode: %s",
                exc,
            )

    # ---------------------------------------------------------------
    # LPIPS
    # ---------------------------------------------------------------
    lpips_fn = None
    try:
        lpips_fn = LPIPSMetric()
        log.info("LPIPS validation enabled.")
    except Exception as exc:
        log.warning("LPIPS unavailable: %s", exc)

    # Only final-phase checkpoints are eligible for weight averaging.
    final_phase_checkpoints = deque(maxlen=5)

    training_start_time = time.perf_counter()
    val_loader = None
    val_hard_loader = None
    final_val_metrics = {}
    final_hard_metrics = {}

    # ===================================================================
    # PHASES
    # ===================================================================
    for phase_idx, phase in enumerate(PHASES):

        if phase_idx < resume_phase:
            continue

        phase_iter = (
            resume_phase_iter
            if phase_idx == resume_phase
            else 0
        )

        # ---------------------------------------------------------------
        # Loss
        # ---------------------------------------------------------------
        criterion = RestorationLoss(
            w_l1=phase["w_l1"],
            w_ssim=phase["w_ssim"],
            w_perceptual=phase["w_perceptual"],
            w_fft=phase["w_fft"],
            w_edge=phase["w_edge"],
            use_msssim=True,
        ).to(device)

        # ---------------------------------------------------------------
        # LR
        # ---------------------------------------------------------------
        # `initial_lr` must be overwritten, not just `lr`. LambdaLR seeds its
        # base LR with `group.setdefault("initial_lr", group["lr"])`, so a key
        # left over from an earlier phase silently wins: resuming a phase-2 run
        # from a phase-1 checkpoint restores initial_lr=2e-4 through the
        # optimizer state and the phase-2 schedule then scales 2e-4 instead of
        # its own phase["lr"]. Setting both keys makes the phase LR authoritative.
        for group in optimizer.param_groups:
            group["lr"] = phase["lr"]
            group["initial_lr"] = phase["lr"]

        scheduler = make_scheduler(
            optimizer,
            phase["warmup_iters"],
            phase["iters"],
            phase["lr"],
            phase["lr_min"],
        )

        # Restore scheduler only when resuming inside this phase.
        if phase_idx == resume_phase and resume_scheduler_state:
            scheduler.load_state_dict(
                resume_scheduler_state
            )

        # ---------------------------------------------------------------
        # Data
        # ---------------------------------------------------------------
        initial_difficulty = phase["difficulty_start"]

        hem_enabled = bool(phase.get("hem", HEM_ENABLED))
        cutblur_p = float(phase.get("cutblur_p", 0.3))
        freq_aug_p = float(phase.get("freq_aug_p", 0.2))
        extra_degrade = bool(
            phase.get("extra_degrade", phase_idx >= 1)
        )

        log.info(
            "[%s] HEM=%s  cutblur_p=%.3f  freq_aug_p=%.3f  "
            "extra_degrade=%s  difficulty=%.2f->%.2f",
            phase["name"],
            "ON" if hem_enabled else "OFF",
            cutblur_p,
            freq_aug_p,
            extra_degrade,
            phase["difficulty_start"],
            phase["difficulty_end"],
        )

        loaders = make_loaders(
            noisy_dir=Path(args.data_root) / "train" / "NoisyLR",
            gt_dir=Path(args.data_root) / "train" / "GT",
            batch_size=phase["batch_size"],
            num_workers=args.num_workers,
            val_frac=0.10,
            hard_frac=0.05,
            extra_degrade=extra_degrade,
            seed=args.seed,
            difficulty=initial_difficulty,
            cutblur_p=cutblur_p,
            freq_aug_p=freq_aug_p,
        )

        train_loader = loaders["train"]
        val_loader = loaders["val"]
        val_hard_loader = loaders["val_hard"]
        train_ds = loaders["_train_ds"]

        train_iter = iter(train_loader)

        train_model.train()

        t0 = time.perf_counter()

        log.info(
            "Starting %s: %d iterations",
            phase["name"],
            phase["iters"],
        )

        # =================================================================
        # ITERATIONS WITH LIVE ETA PROGRESS BAR
        # =================================================================
        pbar = tqdm(
            total=phase["iters"],
            initial=phase_iter,
            desc=f"[{phase['name']}]",
            unit="iter",
            dynamic_ncols=True,
            file=sys.stdout,
        )

        while (
            phase_iter < phase["iters"]
            and (
                args.max_iters is None
                or global_iter < args.max_iters
            )
        ):

            # -----------------------------------------------------------
            # Curriculum.
            # Shared-memory difficulty is visible inside DataLoader
            # workers, including persistent workers.
            # -----------------------------------------------------------
            if phase["iters"] > 1:
                p = phase_iter / float(phase["iters"] - 1)
            else:
                p = 1.0

            difficulty = (
                phase["difficulty_start"]
                + (
                    phase["difficulty_end"]
                    - phase["difficulty_start"]
                ) * p
            )

            train_ds.set_difficulty(difficulty)

            # -----------------------------------------------------------
            # Batch
            # -----------------------------------------------------------
            try:
                noisy, gt = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                noisy, gt = next(train_iter)

            if device.type == "cuda":
                noisy = noisy.to(
                    device,
                    non_blocking=True,
                    memory_format=torch.channels_last,
                )
                gt = gt.to(
                    device,
                    non_blocking=True,
                    memory_format=torch.channels_last,
                )
            else:
                noisy = noisy.to(device, non_blocking=True)
                gt = gt.to(device, non_blocking=True)

            if noisy.ndim != 4 or noisy.shape[1] != 1:
                raise ValueError(
                    f"Expected NoisyLR [B,1,H,W], got {tuple(noisy.shape)}"
                )

            if gt.ndim != 4 or gt.shape[1] != 1:
                raise ValueError(
                    f"Expected GT [B,1,H,W], got {tuple(gt.shape)}"
                )

            if noisy.shape[-2:] != (128, 128):
                raise ValueError(
                    f"Expected NoisyLR 128x128, got {tuple(noisy.shape[-2:])}"
                )

            if gt.shape[-2:] != (256, 256):
                raise ValueError(
                    f"Expected GT 256x256, got {tuple(gt.shape[-2:])}"
                )

            if not torch.isfinite(noisy).all():
                raise FloatingPointError("NaN/Inf found in NoisyLR.")

            if not torch.isfinite(gt).all():
                raise FloatingPointError("NaN/Inf found in GT.")

            # Do NOT clip NoisyLR. Values outside [0,1] are valid.
            optimizer.zero_grad(set_to_none=True)

            # -----------------------------------------------------------
            # Forward in AMP
            # -----------------------------------------------------------
            with autocast_context(device, amp_dtype):
                pred = train_model(noisy)

            if pred.ndim != 4 or pred.shape[1] != 1:
                raise ValueError(
                    f"Expected model output [B,1,H,W], got {tuple(pred.shape)}"
                )

            if pred.shape[-2:] != gt.shape[-2:]:
                raise ValueError(
                    "Model output must match GT spatial size: "
                    f"pred={tuple(pred.shape[-2:])}, "
                    f"gt={tuple(gt.shape[-2:])}"
                )

            if not torch.isfinite(pred).all():
                raise FloatingPointError(
                    f"NaN/Inf prediction at iteration {global_iter}."
                )

            # -----------------------------------------------------------
            # Losses in FP32.
            # -----------------------------------------------------------
            pred32 = pred.float()
            gt32 = gt.float()

            if hem_enabled:
                per_sample = criterion.per_sample_l1(
                    pred32,
                    gt32,
                )
                weights = hem_weights(
                    per_sample,
                    HEM_HARD_FRACTION,
                    HEM_HARD_WEIGHT,
                )
            else:
                per_sample = None
                weights = None

            loss, parts = criterion(
                pred32,
                gt32,
                l1_weights=weights,
                precomputed_l1=per_sample,
            )

            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"NaN/Inf loss at iteration {global_iter}."
                )

            # -----------------------------------------------------------
            # Backward
            # -----------------------------------------------------------
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)

            grad_norm = nn.utils.clip_grad_norm_(
                raw_model.parameters(),
                GRAD_CLIP,
            )

            if not torch.isfinite(
                torch.as_tensor(grad_norm)
            ):
                optimizer.zero_grad(set_to_none=True)
                scaler.update()
                log.warning(
                    "Non-finite gradient norm; skipped iteration."
                )
                continue

            scaler.step(optimizer)
            scaler.update()

            scheduler.step()

            # EMA after optimizer step.
            ema.update()

            phase_iter += 1
            global_iter += 1

            pbar.update(1)
            pbar.set_postfix(
                loss=f"{float(loss.detach()):.4f}",
                lr=f"{optimizer.param_groups[0]['lr']:.1e}",
                diff=f"{difficulty:.2f}",
                refresh=False,
            )

            # -----------------------------------------------------------
            # Logging
            # -----------------------------------------------------------
            if phase_iter % 500 == 0:
                elapsed = time.perf_counter() - t0
                speed = phase_iter / max(elapsed, 1e-6)
                lr = optimizer.param_groups[0]["lr"]

                # Peak VRAM is worth surfacing: on a small card the difference
                # between "fits" and "thrashes" is a few hundred MB, and the
                # allocator degrades badly rather than failing outright.
                vram = (torch.cuda.max_memory_allocated() / 1024 ** 3
                        if device.type == "cuda" else 0.0)

                log.info(
                    "[%s] %d/%d | loss=%.6f | lr=%.3e | "
                    "%.2f it/s | difficulty=%.3f | grad=%.3f | vram=%.2fG",
                    phase["name"],
                    phase_iter,
                    phase["iters"],
                    float(loss.detach()),
                    lr,
                    speed,
                    difficulty,
                    float(grad_norm),
                    vram,
                )

                if writer:
                    writer.add_scalar(
                        "train/loss",
                        float(loss.detach()),
                        global_iter,
                    )
                    writer.add_scalar(
                        "train/lr",
                        lr,
                        global_iter,
                    )
                    writer.add_scalar(
                        "train/difficulty",
                        difficulty,
                        global_iter,
                    )
                    writer.add_scalar(
                        "train/grad_norm",
                        float(grad_norm),
                        global_iter,
                    )

                    if weights is not None:
                        writer.add_scalar(
                            "train/hem_hard_fraction",
                            float(
                                (weights > 1.0).float().mean()
                            ),
                            global_iter,
                        )

                    for k, v in parts.items():
                        if k != "total":
                            writer.add_scalar(
                                f"loss/{k}",
                                float(v.detach()),
                                global_iter,
                            )

            # -----------------------------------------------------------
            # Validation
            # -----------------------------------------------------------
            if phase_iter % val_every == 0:

                ema.apply_shadow()

                try:
                    train_model.eval()

                    val_metrics = validate(
                        train_model,
                        val_loader,
                        device,
                        amp_dtype,
                        "val",
                        lpips_fn,
                        save_dir,
                        global_iter,
                    )

                    hard_metrics = validate(
                        train_model,
                        val_hard_loader,
                        device,
                        amp_dtype,
                        "val_hard",
                        lpips_fn,
                        save_dir,
                        global_iter,
                        n_viz=4,
                    )
                finally:
                    ema.restore()
                    train_model.train()

                if writer:
                    for k, v in val_metrics.items():
                        writer.add_scalar(
                            f"val/{k}",
                            v,
                            global_iter,
                        )
                    for k, v in hard_metrics.items():
                        writer.add_scalar(
                            f"val_hard/{k}",
                            v,
                            global_iter,
                        )

                score = monitoring_score(
                    val_metrics
                )

                log.info(
                    "Internal monitoring score: %.6f",
                    score,
                )

                if score > best_score:
                    best_score = score

                    export_ema_model(
                        raw_model,
                        ema,
                        save_dir / "model_best.pth",
                    )

                    log.info(
                        "NEW BEST MODEL: %.6f",
                        best_score,
                    )

            # -----------------------------------------------------------
            # Checkpoint
            # -----------------------------------------------------------
            if (
                phase_iter % save_every == 0
                or phase_iter == phase["iters"]
            ):
                ckpt = (
                    save_dir
                    / f"{phase['name']}_iter{phase_iter:06d}.pth"
                )

                save_checkpoint(
                    ckpt,
                    raw_model,
                    optimizer,
                    scheduler,
                    scaler,
                    ema,
                    global_iter,
                    phase_idx,
                    phase_iter,
                    best_score,
                )

                if phase_idx == len(PHASES) - 1:
                    final_phase_checkpoints.append(ckpt)

                save_checkpoint(
                    save_dir / "latest.pth",
                    raw_model,
                    optimizer,
                    scheduler,
                    scaler,
                    ema,
                    global_iter,
                    phase_idx,
                    phase_iter,
                    best_score,
                )

        pbar.close()
        log.info("Completed %s.", phase["name"])

        if args.max_iters is not None and global_iter >= args.max_iters:
            log.info(
                "Reached --max_iters=%d; stopping before the next phase.",
                args.max_iters,
            )
            break

    # ===================================================================
    # FINAL-PHASE CHECKPOINT WEIGHT AVERAGING
    #
    # This is an additional candidate only. It never replaces the EMA model
    # automatically.
    # ===================================================================
    if len(final_phase_checkpoints) >= 2:
        try:
            averaged_state = average_model_checkpoints(
                list(final_phase_checkpoints)
            )

            averaged_path = save_dir / "model_avg.pth"
            torch.save(
                averaged_state,
                str(averaged_path),
            )

            verifier_avg = NAFNet(
                **model_config
            ).cpu()

            verifier_avg.load_state_dict(
                averaged_state,
                strict=True,
            )

            log.info(
                "Final-phase model_avg.pth passed strict=True verification."
            )
            log.info(
                "model_avg.pth is an additional candidate; "
                "validate before shipping."
            )

        except Exception as exc:
            log.warning(
                "Final-phase checkpoint averaging failed: %s",
                exc,
            )

    # ===================================================================
    # FINAL EMA EXPORT
    # ===================================================================
    final_path = save_dir / "model.pth"

    export_ema_model(
        raw_model,
        ema,
        final_path,
    )

    # Fresh-model strict compatibility test.
    verifier = NAFNet(**model_config).cpu()
    final_state = torch.load(
        str(final_path),
        map_location="cpu",
    )
    verifier.load_state_dict(
        final_state,
        strict=True,
    )

    log.info(
        "FINAL CHECKPOINT VERIFIED with strict=True."
    )
    log.info(
        "Final model: %s",
        final_path,
    )

    if val_loader is not None and val_hard_loader is not None:
        ema.apply_shadow()
        try:
            raw_model.eval()
            final_val_metrics = validate(
                raw_model,
                val_loader,
                device,
                amp_dtype,
                "final_val",
                lpips_fn,
                save_dir,
                global_iter,
            )
            final_hard_metrics = validate(
                raw_model,
                val_hard_loader,
                device,
                amp_dtype,
                "final_val_hard",
                lpips_fn,
                save_dir,
                global_iter,
                n_viz=4,
            )
        finally:
            ema.restore()

    total_training_time_sec = time.perf_counter() - training_start_time

    generate_model_summary(
        raw_model=raw_model,
        final_path=final_path,
        save_dir=save_dir,
        val_metrics=final_val_metrics,
        hard_metrics=final_hard_metrics,
        best_score=best_score,
        global_iter=global_iter,
        total_training_time_sec=total_training_time_sec,
        device=device,
        amp_dtype=amp_dtype,
        model_config=model_config,
    )

    if writer:
        writer.flush()
        writer.close()


def _sanitize_for_json(obj):
    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_sanitize_for_json(v) for v in obj]
    elif hasattr(obj, "item"):
        return obj.item()
    elif isinstance(obj, (int, float, str, bool, type(None))):
        return obj
    else:
        return str(obj)


def generate_model_summary(
    raw_model: nn.Module,
    final_path: Path,
    save_dir: Path,
    val_metrics: Dict[str, float],
    hard_metrics: Dict[str, float],
    best_score: float,
    global_iter: int,
    total_training_time_sec: float,
    device: torch.device,
    amp_dtype: str,
    model_config: Optional[dict] = None,
):
    model_config = model_config or NAFNET_CONFIG
    total_params = sum(p.numel() for p in raw_model.parameters())
    trainable_params = sum(p.numel() for p in raw_model.parameters() if p.requires_grad)
    model_size_mb = final_path.stat().st_size / (1024 * 1024) if final_path.exists() else 0.0

    summary = {
        "model_architecture": {
            "name": f"NAFNet-{model_config['width']} (Semiconductor Image Restoration)",
            "total_parameters": total_params,
            "total_parameters_million": round(total_params / 1e6, 2),
            "trainable_parameters": trainable_params,
            "model_size_mb": round(model_size_mb, 2),
            "config": model_config,
        },
        "final_accuracy_metrics": {
            "validation_set": {
                "psnr_db": round(float(val_metrics.get("psnr", 0.0)), 4),
                "ssim": round(float(val_metrics.get("ssim", 0.0)), 4),
                "lpips": round(float(val_metrics.get("lpips", 0.0)), 4) if "lpips" in val_metrics else "N/A",
                "mae_l1": round(float(val_metrics.get("l1", 0.0)), 6),
                "mse_l2": round(float(val_metrics.get("l2", 0.0)), 6),
                "composite_monitoring_score": round(float(monitoring_score(val_metrics)), 6),
            },
            "hard_validation_set": {
                "psnr_db": round(float(hard_metrics.get("psnr", 0.0)), 4),
                "ssim": round(float(hard_metrics.get("ssim", 0.0)), 4),
                "lpips": round(float(hard_metrics.get("lpips", 0.0)), 4) if "lpips" in hard_metrics else "N/A",
                "mae_l1": round(float(hard_metrics.get("l1", 0.0)), 6),
            },
            "best_monitoring_score": round(float(best_score), 6),
        },
        "training_details": {
            "total_iterations": global_iter,
            "total_training_time_seconds": round(total_training_time_sec, 2),
            "formatted_training_time": time.strftime("%H:%M:%S", time.gmtime(total_training_time_sec)),
            "device": str(device),
            "amp_precision": amp_dtype,
        },
        "checkpoint_paths": {
            "final_model": str(final_path),
            "best_model": str(save_dir / "model_best.pth"),
            "summary_json": str(save_dir / "model_summary.json"),
            "summary_txt": str(save_dir / "model_summary.txt"),
        }
    }

    # Save JSON summary
    json_path = save_dir / "model_summary.json"
    with open(json_path, "w") as f:
        json.dump(_sanitize_for_json(summary), f, indent=2)

    # Format text report
    report_lines = [
        "=" * 70,
        "                 MODEL & TRAINING DETAILED REPORT              ",
        "=" * 70,
        f" Architecture       : NAFNet-{model_config['width']}"
        f" (Semiconductor Image Restoration)",
        f" Total Parameters   : {total_params:,} ({total_params / 1e6:.2f} M)",
        f" Trainable Params   : {trainable_params:,}",
        f" Model Size on Disk : {model_size_mb:.2f} MB",
        "-" * 70,
        " ACCURACY & EVALUATION METRICS:",
        "   [Validation Set]",
        f"     - PSNR (Peak Signal-to-Noise Ratio) : {val_metrics.get('psnr', 0.0):.4f} dB",
        f"     - SSIM (Structural Similarity)      : {val_metrics.get('ssim', 0.0):.4f}",
    ]
    if "lpips" in val_metrics:
        report_lines.append(f"     - LPIPS (Perceptual Distance)       : {val_metrics.get('lpips', 0.0):.4f}")
    report_lines.extend([
        f"     - MAE (L1 Error)                    : {val_metrics.get('l1', 0.0):.6f}",
        f"     - MSE (L2 Error)                    : {val_metrics.get('l2', 0.0):.6f}",
        f"     - Composite Score                   : {monitoring_score(val_metrics):.6f}",
        "   [Hard Validation Set]",
        f"     - PSNR                              : {hard_metrics.get('psnr', 0.0):.4f} dB",
        f"     - SSIM                              : {hard_metrics.get('ssim', 0.0):.4f}",
        f"     - MAE (L1 Error)                    : {hard_metrics.get('l1', 0.0):.6f}",
        "-" * 70,
        " TRAINING RUN DETAILS:",
        f"   - Total Iterations   : {global_iter}",
        f"   - Total Time Elapsed : {summary['training_details']['formatted_training_time']} ({total_training_time_sec:.2f} s)",
        f"   - Compute Device     : {device}",
        f"   - Precision (AMP)    : {amp_dtype}",
        f"   - Best Score Achieved: {best_score:.6f}",
        "-" * 70,
        " SAVED ARTIFACTS:",
        f"   - Final Checkpoint   : {final_path}",
        f"   - Best Checkpoint    : {save_dir / 'model_best.pth'}",
        f"   - Summary JSON       : {json_path}",
        f"   - Summary TXT        : {save_dir / 'model_summary.txt'}",
        "=" * 70,
    ])

    report_text = "\n".join(report_lines)

    # Save TXT summary
    txt_path = save_dir / "model_summary.txt"
    with open(txt_path, "w") as f:
        f.write(report_text + "\n")

    # Print report to logs
    for line in report_lines:
        log.info(line)

    return summary


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--data_root",
        required=True,
    )
    p.add_argument(
        "--save_dir",
        default="checkpoints",
    )
    p.add_argument(
        "--resume",
        default=None,
    )
    p.add_argument(
        "--num_workers",
        type=int,
        default=2 if sys.platform == "win32" else 8,
        help="Number of DataLoader worker processes (default: 2 on Windows to prevent paging file exhaustion, 8 on Linux).",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    p.add_argument(
        "--deterministic",
        action="store_true",
    )
    p.add_argument(
        "--compile",
        action="store_true",
    )
    p.add_argument(
        "--compile_mode",
        default="default",
        choices=[
            "default",
            "reduce-overhead",
            "max-autotune",
        ],
    )
    p.add_argument(
        "--amp_dtype",
        default="bf16",
        choices=["bf16", "fp16"],
    )
    p.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Override every phase's batch_size (e.g. if T4 OOMs at 16).",
    )
    p.add_argument(
        "--benchmark_layouts",
        action="store_true",
        help="Benchmark NCHW vs channels_last before training.",
    )
    p.add_argument(
        "--benchmark_only",
        action="store_true",
        help="Run --benchmark_layouts and exit without starting training.",
    )
    p.add_argument(
        "--grad_checkpoint",
        action="store_true",
        help="Trade ~30%% speed for much lower activation memory. Needed for "
             "wide models on small GPUs (width 160 @ batch 8 peaks at 5.5GB "
             "with this on, and does not fit without it).",
    )
    p.add_argument(
        "--val_every",
        type=int,
        default=None,
        help="Validate/checkpoint every N iterations (default 5000). Use a "
             "smaller value on short schedules to locate the true optimum.",
    )
    p.add_argument(
        "--width",
        type=int,
        default=NAFNET_CONFIG["width"],
        help="NAFNet base channel width (default 32). Block layout is unchanged; "
             "checkpoints record their own width and load back automatically.",
    )
    p.add_argument(
        "--max_iters",
        type=int,
        default=None,
        help="Cap total optimizer iterations across all phases.",
    )

    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())