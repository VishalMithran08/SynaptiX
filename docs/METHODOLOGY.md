# Experimental Record

**How to read this document.** It is a *chronological* record of the
investigation, written as the work happened, and deliberately not rewritten to
look tidy in hindsight. Sections 2–9 describe the width-32 and width-64 era,
when the best model scored 26.19 dB; those numbers are the numbers *as measured
at that time* and are left intact, because the reasoning only makes sense
against what was known then. Sections 10 and 11 were open questions and a plan;
both now carry their resolutions.

**The shipped model is described in section 1 and in `docs/MODEL_CARD.md`.**
Where this document and the model card disagree, the model card is current.

---

## 1. Executive summary — the shipped model

| | Value |
|---|---|
| **Result** | **PSNR 28.9058 dB · SSIM 0.793831 · LPIPS 0.311227** |
| How it is produced | `weights/nafnet160_final.pth` (EMA) served with 4-view TTA |
| Measured on | 320-image deterministic validation split, no augmentation |
| Model | NAFNet width 160, 183,158,884 params, 349.5 MB |
| Inference | 95 ms/image; 400 images in 38.2 s |

**PSNR is computed per image and then averaged**, the convention used by EDSR,
RCAN, SwinIR and NAFNet. Sections 2–9 quote a *pooled* figure (MSE averaged
across a batch before conversion to dB), which understates by ~2.5 dB and
depends on batch size. See `tests/test_psnr_convention.py`.

**Where the gains came from:**

| Change | Gain (val PSNR, pooled metric) | Cost |
|---|---|---|
| Fixing a broken dataset extraction (359 → 3200 pairs) | **+0.60 dB** | re-download |
| Capacity: width 32 → 64 → 160 | **+0.24 dB** | ~6 h GPU |
| Phase-2 loss redesign | +0.09 dB | ~1 h GPU |
| Test-time augmentation | **+0.30 dB** | 0 training |

The largest single win was a data-integrity fix, not a modelling change. That
remains the headline finding of this project.

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

## 7. Test-time augmentation — the largest single inference-time win

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
| 5 | **`psnr()` pooled MSE across the batch** before converting to decibels. The convention in the restoration literature is to convert per image, then average. `-log` is convex, so by Jensen's inequality pooling can only *understate*, and it makes the value depend on batch size — which a reported metric must never do. | **Our headline understated by 2.54 dB** (26.3635 pooled at batch 16 vs 28.9058 per-image). SSIM was already per-image and was unaffected, which is what isolated the cause. Fixed by `psnr_per_image()`; `tests/test_psnr_convention.py` pins the distinction, including that the per-image figure is invariant to batch grouping and the pooled one is not. |

Bug 2 is the one most worth an outside eye — it is silent, and it affects any
multi-phase resume in this codebase. Bug 5 is the one that cost us the most:
it was found on the last day, by checking a reported *range* against a fresh
measurement rather than by any test, and every number in this document's
sections 2–9 is stated on the pooled metric because of it.

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

The shipped model, end to end from a clean clone:
```
pip install -r requirements.txt
python evaluate.py --input_dir <test images> --output_dir <results>
```

The width-32 number quoted throughout sections 2–9:
```
python tools/eval_report.py --data_root D:/semicon/train_new \
    --checkpoints checkpoints_p2_pilot/model_best.pth --tta
```

*Minor: in-training validation runs under autocast while `eval_report.py` defaults to
AMP off, so the two differ in the 4th decimal (e.g. 26.035945 vs 26.0365). All
comparisons in this document use one path consistently.*

---

## 10. Open questions — and how they resolved

These were genuinely open when written. Each now carries what we found.

1. **What should we optimize?** *Still formally unknown* — the official metric was
   never published. We optimise PSNR/SSIM because the submission template names
   them first, and we expose `--tta_views` so the perception–distortion balance
   can be moved at inference time without retraining: 1 view gives the best
   LPIPS, 8 views the best PSNR/SSIM. This is the one question that never
   closed, so we made the model configurable along that axis instead of guessing.
2. **Is the capacity probe sound?** *It was underpowered.* The plateau did not
   indicate a representational limit. Scaling to 183M parameters improved the
   top frequency band 13.3% → 21.0%, and the train/val gap stayed at −1.5% —
   capacity never saturated even at 24× the parameters.
3. **Is ~26.2 dB reasonable?** *Yes, and it was conservative on two counts.*
   Capacity was still binding (Q2), and our own PSNR helper pooled MSE across
   batches, understating the figure by 2.5 dB (§9).
4. **Difficulty curriculum — is the ~0.48 threshold plausible?** *Confirmed, and
   it moves with capacity:* ~0.48 at width 32, ~0.31 at width 64. It is a
   capacity-dependent threshold, not a constant — which is why a single global
   difficulty setting was the wrong shape for the problem.
5. **Is the degradation model worth inverting explicitly?** *Not pursued.* The
   end-to-end model already handles out-of-range inputs without clipping, and
   the measured bottleneck proved to be capacity, then information — not
   preprocessing.

---

## 11. What was actually done, and what it measured

The plan below was written speculatively, with estimated gains. This is the
outcome, with measured numbers replacing the estimates.

| Option (as planned) | Status | Measured outcome |
|---|---|---|
| **A.** Bank a submission with the current model | done | width-32 submission produced and kept as a fallback |
| **B.** Confirm the official metric | **never resolved** | not published; handled by making TTA configurable (§10 Q1) |
| **C/D.** Weight averaging / prediction ensembling | built, not shipped | `tools/ensemble.py`; gains sat inside run-to-run noise |
| **H.** Width-64 retrain | done | 26.2934 pooled (+0.10 dB over width 32) |
| — Width-160 (beyond the plan) | done | **the shipped model**; required gradient checkpointing to fit 8 GB |
| **M.** Alternative architecture (SwinIR / transformer) | done | **negative.** SwinIR's flat topology fits only 13.8M params in 8 GB and needs 15.4 h, against our 183M in 5.5 h. A U-shaped NAFNet+attention hybrid (248.8M) trained, then diverged, and its best checkpoint measured behind the CNN. `models/swinir.py` ships as the SwinIR ablation; the hybrid was a configuration of the NAFNet code and neither its variant nor its checkpoint is shipped, since it lost. |
| **O.** Perceptual / adversarial training | swept; adversarial rejected | perceptual weight tuned 0.05 / 0.02 / 0.01 → 0.01 best. Adversarial rejected on principle: synthesised texture is uncorrelated with the truth and *doubles* the error in the bands it fills. |
| **Q.** Fine-tune on the hard subset | audited, not done | the audit was right — "hard" selects by maximum absolute magnitude, not difficulty, and scores *better* than the normal split. Mislabelled. |

**Two experiments that failed, reported because they were informative:**

- **FFT loss reweighting.** Per-band recovery falls from 99.6% at DC to 21.0% at
  Nyquist, so we doubled the high-frequency loss share (40.3% → 63.0%).
  High-frequency recovery got **worse** (21.0% → 19.9%). An FFT *magnitude* loss
  constrains how much high-frequency energy exists but not *where* it goes, so
  misplaced detail costs more in L1 than it earns in FFT and the optimiser emits
  less. Fine stochastic texture is not recoverable by loss engineering — the
  model does not lack the capacity to produce it, it lacks the information to
  place it.
- **The NAFNet + window-attention hybrid.** It trained stably, then a gradient
  spike saturated the outputs; `clamp(0,1)` has zero gradient outside its range,
  so gradients read exactly 0.000 for 25,000 iterations before anyone noticed.
  Six and a half GPU-hours lost to a failure mode with an obvious detector we
  had not built.

**Honest expectation, as written at the time:** *"Tier 1 + Tier 2 realistically
lands ~26.4–6.6 dB."* On the pooled metric the shipped model reached 26.41 dB —
inside that range, and reached by capacity scaling (H, extended past the plan)
rather than by the architecture change (M) that was expected to deliver the step
change.

---

## 12. Artifacts

| Path | Contents |
|---|---|
| `weights/nafnet160_final.pth.part{000..003}` | **The shipped model** — width 160, EMA, four checksummed parts |
| `evaluate.py` | Inference entry point; reassembles the weights in memory |
| `docs/MODEL_CARD.md` | Full specification of the shipped model — **current** |
| `train.py` | Training pipeline, all phases and losses |
| `tools/verify_dataset.py` | Dataset integrity gate — the check behind the largest single gain |
| `tools/eval_report.py` | Checkpoint evaluation (RAW/EMA, TTA, LPIPS, spectral) |
| `tools/content_shift.py` | Generalisation to unseen content |
| `tools/ood_probe.py` | Robustness under degradation shift |
| `tools/spectral_bands.py` | Per-frequency-band recovery |
| `tools/checkerboard.py` | Quantifies PixelShuffle artifacts |
| `outputs/` | 400 restored test images |
| `models/swinir.py` | SwinIR implementation — the ablation behind §11 option M |

Historical artifacts referenced in sections 2–9 (`checkpoints_p2_pilot/`,
`p*_eval.json`, `audit_report.md`) live in the training workspace and are not
part of this repository. `audit_report.md` in particular is **superseded** — its
conclusions were drawn from the broken dataset.
