# Semiconductor Image Restoration — Findings & Continuation Plan

**Prepared for external review.** Every number below is measured on this machine unless
explicitly marked as an estimate. Where an inference is uncertain, it says so.

---

## 1. Executive summary

| | Value |
|---|---|
| **Current best result** | **PSNR 26.1931 dB · SSIM 0.791754 · L1 0.029870** |
| How it is produced | `checkpoints_p2_pilot/model_best.pth` served with 8-view TTA |
| Measured on | 320-image deterministic validation split, no augmentation |
| Model | NAFNet-32, 7,492,836 params, 28.7 MB |

**Where the gains came from:**

| Change | Gain (val PSNR) | Cost |
|---|---|---|
| Fixing a broken dataset extraction (359 → 3200 pairs) | **+0.60 dB** | re-download |
| Phase-2 loss redesign + 25k iterations | +0.09 dB | ~4 h GPU |
| Test-time augmentation (8-view self-ensemble) | **+0.16 dB** | 0 training |

The two largest wins were a data-integrity fix and a free inference-time trick.
Training-recipe changes contributed comparatively little. **This is the central fact
we would like reviewed:** we may be near a task-intrinsic ceiling, or we may be
leaving something significant on the table.

---

## 2. Setup

**Task.** Restore a degraded 128×128 grayscale image to a clean 256×256 image
(joint denoise + 2× upscale). Degradation is stated by the organizers to be speckle
noise, Gaussian noise, and 2× downsampling. Gaussian blur is explicitly *not* part of it.

**Data.** 3200 matched training pairs; 400 test inputs with hidden GT.
- `NoisyLR`: 128×128 float32, range ≈ `[-0.278, 1.851]` — contains physical
  out-of-range values and is **never clipped** on input.
- `GT`: 256×256 float32, range `[0, 1]`.
- Split (seed 42, deterministic): **train 2720 / val 320 / val_hard 160**.
  `val_hard` = top 5% by max absolute magnitude, removed first; the remainder is
  split 90/10. Verified non-overlapping.
- Note: despite the project name, the imagery is natural grayscale photography
  (trees, bird feathers, fabric), not wafer imagery.

**Model.** NAFNet, `width=32`, `enc_blocks=[2,2,4,8]`, `middle_blocks=12`,
`dec_blocks=[2,2,2,2]`, 1→1 channel, 2× upscaling via PixelShuffle. Output clamped
to `[0,1]` — the only clamp in the forward pass.

**Environment.** Python 3.11.9 · torch 2.12.0.dev+cu128 · CUDA 12.8 · cuDNN 9.2 ·
NVIDIA RTX 5060 Laptop (8 GB, compute 12.0) · AMP **bf16** · seed 42.

**Metrics.** `PSNR = 10·log10(1/MSE)`; `SSIM` 11×11 Gaussian σ=1.5;
`L1 = mean|pred − gt|`. Batch-averaged.

> ⚠️ **Scoring caveat.** The internal `monitoring_score` is
> `0.45·(PSNR/50) + 0.35·SSIM + 0.20·(1−LPIPS)`. **LPIPS is not installed**, so it
> defaults to 1.0 and that term contributes a constant zero — best-checkpoint
> selection is effectively **PSNR + SSIM only**. The project's own docs state this
> score is *not* the official competition metric, and **the official metric is
> unknown to us**. This materially affects strategy (see §8, Q1).

---

## 3. The finding that invalidated all prior work

The training directory contained **359 matched pairs, not 3200**. A partial
extraction of a Mac-created zip left 2121 IDs missing from both `GT/` and
`NoisyLR/`, plus 352 GT-only and 368 NoisyLR-only orphans. The effective split was
**train 307 / val 35 / val_hard 17**.

Consequences:
- A 7.49M-parameter model was trained ~1000 epochs over 307 images.
- The previously reported baseline of **26.1175 dB was a 35-image measurement**,
  optimistic by **0.76 dB**. Re-measured on the full 320-image split, the same
  checkpoint scores **25.3527 dB**.
- A prior audit document concluded the model was "over-sharpening" due to the
  difficulty curriculum and hard-example mining. That conclusion was drawn entirely
  from this broken data and is wrong in its central claim (see §5).

The full dataset was re-downloaded and verified: 3200/3200 pairs, correct shapes and
dtypes, all finite, and the 359 previously-usable pairs are **byte-identical** to
their counterparts in the new extraction (i.e. a strict superset, same generation).

---

## 4. Phase 1 — same recipe, correct data

Re-ran `phase1_warmup` **bit-for-bit unchanged** (difficulty 0.20→0.40, HEM ON,
`freq_aug_p=0.2`, `cutblur_p=0.3`, lr 2e-4→2e-6, warmup 2k, L1-only, batch 16,
20k iters, bf16). Only the dataset size differed — a clean single-variable test.

| Iter | L1 | PSNR | SSIM | hard PSNR | edge | HF |
|---|---|---|---|---|---|---|
| 5k | 0.032187 | 25.5326 | 0.766532 | 26.5938 | 0.232916 | 0.002173 |
| 10k | 0.031004 | 25.8472 | 0.777154 | 26.9987 | 0.232009 | 0.002359 |
| 15k | 0.030779 | 25.9259 | 0.779981 | 27.1017 | 0.233951 | 0.002609 |
| **20k** | **0.030685** | **25.9484** | **0.780552** | **27.1290** | 0.232688 | 0.002542 |

*(EMA weights; RAW tracked EMA closely and was consistently marginally worse.)*

**Monotone improvement on every metric and both splits.** On the broken data, the
identical schedule had *degraded* monotonically (25.35 → 25.07 dB). The "degradation"
was an artifact of training on 307 images.

---

## 5. The over-sharpening diagnosis was inverted

We measured the **ground truth's own** spectral energy to give the prediction-side
numbers a reference target:

```
GT           edge = 0.309172    hf = 0.010758
P1 20k pred  edge = 0.232688    hf = 0.002542
             →  75% of GT edge,  24% of GT high-frequency energy
```

The model produces **less than a quarter of the ground truth's high-frequency
content**. It is chronically **over-smoothed**, not over-sharpened. Worst-case
visualizations confirm this directly: predictions are visibly smeared against GT,
with **no ringing, no halos, no edge overshoot, no artificial texture**.

Rising HF energy — the prior audit's key evidence of "over-sharpening" — was in fact
the model legitimately recovering detail it lacks. Phase 2 was therefore designed to
**add** high-frequency pressure rather than remove it.

`edge` = mean Sobel gradient magnitude. `hf` = fraction of FFT power above
0.5 × Nyquist radius. Both computed on predictions over the same 320-image split.

---

## 6. Phase 2 — loss redesign, and a measured difficulty threshold

**Config:** `w_l1=1.00, w_ssim=0.10, w_fft=0.05, w_edge=0.05, w_perceptual=0.00`;
lr 1e-4→1e-6, warmup 2k, 60k iters, batch 16, difficulty **0.40→0.60**, HEM ON,
`freq_aug_p=0.2`, resumed from Phase-1 20k.

**Why `w_perceptual` was set to 0:** at its previous value of 0.05, the VGG term
measured 1.677624 raw = 0.084 weighted = **63% of total loss**, dwarfing L1 at
0.0345. It optimizes feature similarity rather than pixel fidelity and cost 0.020
SSIM in an earlier run. Under the new weights, L1 is 67% of the budget.

| Phase iter | difficulty | LR | L1 | PSNR | SSIM |
|---|---|---|---|---|---|
| 5k | 0.417 | 1.0e-04 | 0.030782 | 25.9474 | 0.784324 |
| 10k | 0.433 | 9.5e-05 | 0.030657 | 25.9922 | 0.785549 |
| 15k | 0.450 | 8.9e-05 | 0.030575 | 26.0205 | 0.786247 |
| 20k | 0.467 | 7.8e-05 | 0.030559 | 26.0316 | 0.786271 |
| **25k** | **0.483** | 3.9e-05 | **0.030549** | **26.0365** | **0.786838** |
| 30k | 0.500 | 3.4e-05 | 0.030566 | 26.0367 | 0.786460 |
| 35k | 0.517 | 3.0e-05 | 0.030606 | 26.0281 | 0.786409 |
| 40k | 0.533 | 2.7e-05 | 0.030630 | 26.0216 | 0.786661 |
| 45k | 0.550 | 1.6e-05 | 0.030675 | 26.0129 | 0.786201 |
| 50k | 0.567 | 8.1e-06 | 0.030700 | 26.0072 | 0.786079 |
| 55k | 0.583 | 2.8e-06 | 0.030705 | 26.0062 | 0.785954 |
| 60k | 0.600 | 1.0e-06 | 0.030708 | 26.0034 | 0.786012 |

**A difficulty threshold at ≈ 0.48–0.50.** Phase 1 ran 0.20→0.40 and improved
throughout. Phase 2 improved to ~0.48, then declined at **seven consecutive
evaluations**, with L1 rising every time. The cosine anneal did not rescue it —
decline continued as LR fell from 3.9e-05 to 1e-6, flattening only because the
weights stopped moving.

**Mechanism, 25k → 60k:** `hf` rose 0.003184 → 0.003415 (**+7%**) while `edge`
stayed flat and PSNR fell. That *is* genuine high-frequency over-production — the
mechanism the original audit proposed — but it only appears **above difficulty ~0.5**.
The audit inferred it from a 0.20→0.40 run on broken data where it was not occurring.
Right mechanism, wrong regime, wrong evidence.

**Implication:** `phase3_finetune` as configured in `train.py` uses
`difficulty_start = difficulty_end = 1.00`, far into the harmful regime. **It should
not be run unchanged.**

---

## 7. Test-time augmentation — the largest single win

8-view geometric self-ensemble over the D4 dihedral group (4 rotations × {identity,
hflip}), averaged in the original frame. Same checkpoint, same split, same evaluator.

| | no TTA | with TTA | gain |
|---|---|---|---|
| val PSNR | 26.0365 | **26.1931** | **+0.157 dB** |
| val SSIM | 0.786838 | **0.791754** | **+0.004916** |
| val L1 | 0.030549 | **0.029870** | −0.000679 |
| hard PSNR | 27.2204 | **27.4841** | **+0.264 dB** |
| hard SSIM | 0.829088 | **0.836680** | **+0.007592** |

This one change exceeded the entire Phase-2 training gain (+0.09 dB) for zero
training cost. Hard validation gains most, consistent with orientation-specific error
being largest on difficult images.

**Note:** TTA slightly *reduces* edge (0.245154 → 0.242660) and HF (0.003184 →
0.003019) energy while improving every fidelity metric. Averaging smooths. For
PSNR/SSIM this is strictly good; under a perceptual metric the trade could reverse.

---

## 8. Diagnostics bearing on "what is the ceiling?"

### Generalization gap — essentially zero

Best model evaluated on **training** pairs through the *validation* pipeline
(no augmentation, difficulty 0), matched sample count:

```
TRAIN (clean, n=320)   L1=0.029888   PSNR=25.9010   SSIM=0.790969
VAL   (clean, n=320)   L1=0.030549   PSNR=26.0365   SSIM=0.786838
```

L1 differs by 2.2%. PSNR is *higher* on validation (that subset is easier). After
~353 epochs of exposure, the model performs the same on memorized and unseen data.
**No overfitting whatsoever.**

> **Interpretive caution (we flag this as a genuine weakness in our reasoning).**
> We initially read this as proof of *capacity limitation*. It is not conclusive.
> Two different ceilings produce an identical zero gap:
> 1. **Capacity-limited** — the network cannot represent the mapping.
> 2. **Noise-limited (Bayes error)** — the degraded input does not determine the GT,
>    and the model has converged to the conditional mean, which is the optimal L1
>    predictor. This *also* explains the over-smoothing, the fast convergence, and
>    the diminishing returns.

### Capacity probe — ambiguous, leaning capacity-bound

Attempted to overfit **32 training pairs** (no augmentation, pure L1, lr 2e-4,
batch 8, 1500 iters ≈ 375 epochs), starting from the best checkpoint:

```
start                 L1 0.0299
iter  750             L1 0.0142
iter 1500             L1 0.0161      ← plateaued, not still falling
final (eval mode)     L1 0.0189
```

- **Supports capacity limitation:** reached ~0.015 on 32 memorized pairs vs 0.0299
  on the full training set, so the model is *not* at the noise floor — there is real
  fitting headroom that width 32 cannot exploit across 2720 images.
- **Argues against:** it plateaued at ~0.015 rather than collapsing toward ~0.002.
  A network with ample capacity should memorize 32 fixed pairs almost exactly. Some
  residual is genuinely irreducible.

**We would particularly value an expert opinion on this diagnostic and its
interpretation.**

---

## 9. Bugs found and fixed

| # | Bug | Impact |
|---|---|---|
| 1 | **Broken dataset extraction** — 359/3200 pairs | Invalidated all prior results |
| 2 | **Stale `initial_lr` on resume** — `LambdaLR` seeds its base LR via `group.setdefault("initial_lr", ...)`. Resuming restores the previous phase's `initial_lr` through the optimizer state, so a new phase silently scaled the *old* LR. Observed: Phase 2 ran at 2e-4 instead of the configured 1e-4. | Every resumed run before the fix has an unverified LR |
| 3 | Smoke-test assertions hardcoded to one experiment's values | Blocked valid configs; tested constants rather than plumbing |
| 4 | No per-run log file; shell redirection produced UTF-16 with `NativeCommandError` wrapping | Logs hard to read back |

Bug 2 is the one most worth an outside eye — it is silent, and it affects any
multi-phase resume in this codebase.

### Code changes
- `train.py` — Phase-2 config; `initial_lr` fix; UTF-8 `train.log` per run.
- `utils/tta.py` *(new)* — D4 self-ensemble; accumulates one view at a time so peak
  memory stays at single-pass levels.
- `tests/test_tta.py` *(new)* — 8 tests guarding failure modes that produce **no
  error, only worse output**: exact `_inverse ∘ _forward = identity`, the 8 views
  being genuinely distinct, and averaging being exactly identity for a provably
  equivariant model.
- `eval_report.py`, `inference.py` — `--tta` flag.
- `verify_dataset.py` *(new)* — pair-count / shape / dtype / finiteness gate.
- Test suite: **34 passed, 0 failed.**

### Reproducing the headline number
```
python eval_report.py --data_root D:/semicon/train_new \
    --checkpoints checkpoints_p2_pilot/model_best.pth --tta
```

*Minor: in-training validation runs under autocast while `eval_report.py` defaults to
AMP off, so the two differ in the 4th decimal (e.g. 26.035945 vs 26.0365). All
comparisons in this document use one path consistently.*

---

## 10. Open questions for the reviewer

1. **What should we optimize?** The official competition metric is unknown to us.
   Everything above optimizes PSNR/SSIM. If scoring is perceptual (LPIPS, or human
   judgment), the strategy inverts: a blurry conditional-mean prediction is *optimal*
   for PSNR and *poor* perceptually, and perceptual/adversarial losses — which we
   deliberately removed — would become correct. **This single unknown changes the
   whole plan.**
2. **Is the capacity probe (§8) sound?** Does the plateau at ~0.015 indicate a
   representational limit, or is the probe underpowered (too few iterations, LR too
   low, or memorization genuinely impossible under this degradation)?
3. **Is ~26.2 dB reasonable** for joint 2× SR + speckle/Gaussian denoise at this
   noise level, or does it suggest something is being left on the table?
4. **Difficulty curriculum** — is a measured harm threshold at ~0.48 plausible, and
   is the curriculum worth keeping at all versus training at fixed low difficulty?
5. **Is the degradation model worth inverting explicitly?** Inputs carry physical
   out-of-range values (up to ~1.85). Is there value in a noise-model-aware
   preprocessing step rather than learning it end-to-end?

---

## 11. Continuation plan — all options considered

Gains marked *(est.)* are estimates from literature and our own trend data, **not
measured**. Costs assume the RTX 5060 Laptop (8 GB) at ~3.5–4 it/s for width 32.

### Tier 0 — Do regardless

| Option | Gain | Cost | Risk |
|---|---|---|---|
| **A. Generate submission with current model + TTA** | banked | 4 min | none |
| **B. Confirm the official competition metric** | decisive | minutes | none |

Nothing else should be started before A and B. A valid submission at 26.19 dB in
hand is worth more than any speculative gain; and B may invalidate the entire
optimization direction.

### Tier 1 — Cheap, low risk

| Option | Gain *(est.)* | Cost | Notes |
|---|---|---|---|
| **C. Checkpoint weight averaging** | +0.02–0.05 dB | minutes | Average 20k/25k/30k weights. `train.py` already has a `model_avg.pth` path. Free; near-zero risk. |
| **D. Multi-checkpoint prediction ensembling** | +0.03–0.08 dB | inference only | Average *predictions* (not weights) from 2–3 checkpoints. Stacks with TTA. Costs N× inference. |
| **E. Difficulty-capped short rerun (hold 0.45)** | +0.01–0.03 dB | ~2 h | Tests §6's threshold directly. Scientifically clean, small payoff. |
| **F. EMA decay sweep (0.999 → 0.9995/0.9999)** | +0.01–0.03 dB | ~2 h each | EMA consistently beat RAW; the decay was never tuned. |

### Tier 2 — Moderate cost, moderate payoff

| Option | Gain *(est.)* | Cost | Notes |
|---|---|---|---|
| **G. Width 48 retrain** | +0.10–0.20 dB | ~3–4 h | ~17M params, batch 12. Safer memory profile than width 64. |
| **H. Width 64 retrain** | +0.15–0.30 dB | ~6–7 h | ~30M params, batch 8–12 on 8 GB. Must train from scratch — weights do not transfer across widths. Our headline capacity recommendation, but see §8 caveat. |
| **I. Deeper instead of wider** (`middle_blocks` 12→20) | +0.05–0.15 dB | ~3 h | Cheaper in memory than widening; unclear whether depth or width binds. |
| **J. Cosine warm restarts / longer schedule at capped difficulty** | +0.05–0.10 dB | ~4 h | Phase 2 converged; restarts may escape the basin. |
| **K. Charbonnier loss instead of pure L1** | ±0.05 dB | ~2 h | Smooth-L1 variant, standard in SR. Cheap to test. |
| **L. Larger effective batch via gradient accumulation** | +0.02–0.08 dB | ~same | Not currently implemented. Would decouple batch size from VRAM for Tier-2 runs. |

### Tier 3 — High cost or high uncertainty

| Option | Gain *(est.)* | Cost | Notes |
|---|---|---|---|
| **M. Alternative architecture** (SwinIR / Restormer / HAT) | unknown, possibly large | 1–3 days | Transformer restoration models typically outperform NAFNet at equal params, at higher training cost. Biggest potential step change; biggest time risk. |
| **N. Pretrained initialization / transfer** | unknown | ~1 day | Initialize from a public denoising/SR checkpoint. Domain differs (grayscale, unusual value range) — may not transfer. |
| **O. Perceptual or adversarial training** | metric-dependent | ~1 day | **Only if §10 Q1 says perceptual.** Would *reduce* PSNR/SSIM while improving perceptual quality. We measured perceptual loss costing 0.020 SSIM. |
| **P. Explicit degradation-model preprocessing** | unknown | ~1 day | Variance-stabilizing transform for speckle (e.g. log or Anscombe-like) before the network, exploiting the known noise model rather than learning it. |
| **Q. Fine-tune on the hard subset** | unclear | ~2 h | Hard-set metrics are *better* than normal-set ones, so "hard" may be mislabelled — it selects by max absolute magnitude, not by difficulty. Worth auditing before acting. |

### Explicitly not recommended

- **`phase3_finetune` as configured** — `difficulty=1.00` is deep in the measured
  harmful regime (§6).
- **200k-iteration training** — at batch 16 over 2720 images that is ~1176 epochs,
  the same regime that produced degradation on the broken dataset.
- **More data / stronger regularization / heavier augmentation** — there is zero
  generalization gap (§8); these attack a problem that does not exist here.

### Our recommended sequence

1. **A + B** (submission banked, metric confirmed) — before anything else.
2. **C + D** — free/cheap ensembling wins.
3. **H** (width 64) overnight *if* time permits and Q1 confirms PSNR/SSIM scoring.
4. **M** only if there is substantial time remaining and Tier 2 disappoints.

**Honest expectation:** Tier 1 + Tier 2 realistically lands ~**26.4–26.6 dB**.
A step change beyond that likely requires Tier 3 (architecture), and we are not
confident it is available at all — §8 leaves open that we are near a task-intrinsic
ceiling.

---

## 12. Artifacts

| Path | Contents |
|---|---|
| `semi/checkpoints_p2_pilot/model_best.pth` | **Best model** (Phase-2 25k EMA) |
| `semi/checkpoints_p1_fulldata/` | Phase-1 full-data run + `train.log` |
| `semi/checkpoints_p2_pilot/` | Phase-2 run, checkpoints every 5k, `viz/` worst-case images |
| `train_new/` | Verified 3200-pair dataset |
| `p1_fulldata_eval.json`, `p2_pilot_eval.json`, `p2_final_eval.json`, `tta_eval.json` | Full RAW/EMA + spectral evaluations |
| `verify_dataset.py` | Dataset integrity gate |
| `audit_report.md` | **Superseded** — conclusions drawn from the broken dataset |
