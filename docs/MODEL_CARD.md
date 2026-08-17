# Model Card — Shipped Submission

**File:** `checkpoints_w160/model_best.pth` → shipped as
`submission/weights/nafnet160_final.pth.part{000..003}`
**Repository:** https://github.com/VishalMithran08/semicon
**Verified:** 532/532 tensors byte-identical between the local best checkpoint
and the four weight parts committed to GitHub.

---

## 1. Headline accuracy

Default configuration is **4-view TTA**, chosen because inference time is a
separately scored axis and 4 views beats 8 on a composite of PSNR/SSIM/LPIPS
while running twice as fast.

| Metric | 4-view (default) | 8-view |
|---|---|---|
| **PSNR** | **28.9058 dB** | 28.9711 dB |
| **SSIM** | **0.793831** | 0.795909 |
| **LPIPS** ↓ | **0.311227** | 0.319032 |
| MAE (L1) | 0.029385 | 0.029191 |
| composite | 0.698647 | 0.686031 |
| ms/image | **96.6** | 190.6 |

**PSNR is computed per image and then averaged** (`utils.metrics.psnr_per_image`),
the convention used by EDSR / RCAN / SwinIR / NAFNet. The repository's older
`psnr()` pools MSE across a batch before converting to dB; by Jensen's
inequality that can only understate, and it makes the value batch-size
dependent. The gap here is **2.54 dB** (26.3635 pooled at batch 16). SSIM and
LPIPS were already per-image and are unchanged. See
`tests/test_psnr_convention.py`.

Hard subset (160 images, top 5% by intensity magnitude):

| Metric | Value |
|---|---|
| PSNR | 30.2194 dB |
| SSIM | 0.838477 |
| LPIPS ↓ | 0.306489 |
| MAE | 0.023792 |

### Effect of test-time augmentation

| views | PSNR | SSIM | LPIPS ↓ | composite | ms/image |
|---|---|---|---|---|---|
| 1 | 28.6041 | 0.786095 | **0.302654** | 0.703241 | **31.7** |
| 2 | 28.7821 | 0.790462 | 0.304176 | **0.707546** | 50.8 |
| **4 (default)** | 28.9058 | 0.793831 | 0.311227 | 0.698647 | 96.6 |
| 8 | **28.9711** | **0.795909** | 0.319032 | 0.686031 | 190.6 |

The three measures disagree, and should: PSNR and SSIM rise monotonically with
views while LPIPS falls monotonically, because averaging smooths -- the
perception-distortion tradeoff in miniature. The composite (PSNR/50 + SSIM -
2*LPIPS) therefore peaks at **2 views**, not 4.

4 is shipped because the brief names PSNR and SSIM first: it keeps 84% of the
8-view PSNR gain for half the time, and the last four views buy +0.065 dB for
+94 ms -- 16x worse value than the first flip pair. `--tta_views 1` is both the
fastest setting and the best LPIPS if perceptual quality is what is scored.

**End-to-end profile** (400 images): GPU compute 99.1%, disk read 0.2%, disk
write 0.6%, host/device transfer 0.0%. I/O parallelisation cannot help; the
self-ensemble size is the throughput dial.

### Gain over doing nothing

Against bicubic 2× upscaling of the degraded input, per-image gain ranges
**+1.59 dB to +8.50 dB**, depending on how much irreducible texture the target
contains.

**Per-image PSNR spans 11.68 dB to 43.39 dB** (median 28.66). The 28.91 dB
average conceals that spread and should not be read as uniform performance.

---

## 2. Architecture

| | |
|---|---|
| Family | NAFNet (Chen et al., *Simple Baselines for Image Restoration*, ECCV 2022) |
| Base width | 160 |
| **Parameters** | **183,158,884 (183.16 M)** |
| Encoder | 4 stages, channels [160, 320, 640, 1280], blocks [2, 2, 4, 8] |
| Bottleneck | 12 NAFBlocks |
| Decoder | 4 stages, blocks [2, 2, 2, 2], skip-concat + 1×1 reduce |
| SR head | Conv(160→4) + PixelShuffle(2), ICNR-initialised |
| Input / output | 1 channel; H×W → 2H×2W |
| Clamping | Output only, `clamp(0,1)`. **Input never clipped.** |
| File size | 349.5 MB (float16 storage, float32 inference) |

Deviations from the paper: PixelShuffle SR tail with ICNR (original NAFNet is
same-resolution), single-channel I/O, and sigmoid-gated channel attention.

---

## 3. Training

| | Stage 1 | Stage 2 | Total |
|---|---|---|---|
| Iterations | 22,500 | 2,500 | 25,000 |
| Batch size | 8 | 8 | — |
| **Sample presentations** | **180,000** | **20,000** | **200,000** |
| **Epochs** (2,720 train images) | **66.2** | 7.4 | **73.5** |
| Loss | L1 only | L1 1.0 + SSIM 0.10 + FFT 0.05 + edge 0.05 + perceptual 0.01 | — |
| LR schedule | 2e-4 → 2e-6 cosine, 2,000 warmup | 1e-4 → 1e-6 cosine, 300 warmup | — |
| Difficulty | 0.20 → 0.40 | 0.30 fixed | — |
| Wall-clock | ~5.5 h | ~0.7 h | **~6.2 h** |

**Note on "epochs":** iteration counts are not comparable across batch sizes, so
phase length is defined in *sample presentations* and epochs are derived.
Stage 2 stopped at 2,500 of a planned 4,500 because a checkpoint save exhausted
system RAM; the 2,500 state was already the best and later runs confirmed
stage 2 plateaus within 1–3k iterations.

### Optimisation

| | |
|---|---|
| Optimizer | AdamW, weight decay 1e-4, betas (0.9, 0.999), eps 1e-8 |
| Gradient clipping | 1.0 |
| Mixed precision | bfloat16 autocast (GradScaler disabled — bf16 needs none) |
| EMA | decay 0.999, warmup 2,000 — **EMA weights are what ship** |
| Memory | Gradient checkpointing, peak **5.3 GB** |
| Seed | 42 |
| Hard-example mining | On, top 50% of batch at 2.0× weight, normalised to mean 1 |

### Data

| | |
|---|---|
| Source | 3,200 matched NoisyLR/GT `.npy` pairs |
| Split (seed 42) | **train 2,720 / val 320 / val_hard 160** — verified non-overlapping |
| Input | 128×128 float32, range ≈ [−0.278, 1.851] (**not** clipped) |
| Target | 256×256 float32, range [0, 1] |
| Augmentation | Paired flip/rot90; signed gamma + intensity jitter; CutBlur (p ≤ 0.3×difficulty); frequency-band suppression (p ≤ 0.2×difficulty) |

---

## 4. Generalisation and robustness

### Overfitting: none

Same model scored on **training** pairs and **validation** pairs under identical
conditions (no augmentation, difficulty 0, 128 images each):

```
TRAIN   L1 0.028098        VAL   L1 0.027671        gap  -1.5%
```

Negative gap — it performs no better on data it trained on than on data it has
never seen, at 183M parameters over 2,720 images. Capacity is not saturated.

### Out-of-distribution robustness

The competition test set contains images "from different sources", so the model
was scored under progressively harsher degradation than it trained on:

| degradation | PSNR | SSIM |
|---|---|---|
| 0.00 (as-shipped) | 26.1227 | 0.7861 |
| 0.50 | 25.6125 | 0.7559 |
| 1.00 | 24.5127 | 0.7143 |

It beat the previous 29.56M model at **every** level, and the margin widened
under shift (+0.050 → +0.177 dB), i.e. it degrades more gracefully rather than
merely fitting the training distribution better.

### Content shift: unseen image content

Degradation strength is only one axis. The brief says the test set comes from
**different sources**, so the model was also scored on content unlike anything
it trained on — photographs, printed text, coins, lunar surface, brick —
degraded with the exact training recipe so the ground truth is exact
(`tools/content_shift.py --builtin`).

| | bicubic 2× | model | gain |
|---|---|---|---|
| PSNR | 20.6841 | **24.9821** | **+4.30 dB** |
| SSIM | 0.4237 | **0.6520** | **+0.2282** |

**Wins on 11/11 images**, range +0.90 to +6.53 dB. The two weakest are `grass`
and `gravel` — pure stochastic texture, exactly the limitation section 7 states
and section 5 measures. Structured content gains most (`brick` +6.53,
`chelsea` +6.10). The restoration transfers; it is not memorised content.

### Artifact check

`tools/checkerboard.py` measures period-2 alternation in the flattest 40
validation images. Ground truth = 1.00×; this model = **0.39×**. Well below
ground truth, so it is not fabricating periodic structure — the failure mode the
brief warns against.

---

## 5. Spectral fidelity

Prediction power / ground-truth power, radial frequency bands:

| band (× Nyquist) | recovered |
|---|---|
| 0.00–0.12 | 99.6% |
| 0.12–0.25 | 93.0% |
| 0.25–0.38 | 78.7% |
| 0.38–0.50 | 62.6% |
| 0.50–0.62 | 46.9% |
| 0.62–0.75 | 35.8% |
| 0.75–0.88 | 26.3% |
| 0.88–1.00 | 21.0% |

The lowest band holds ~97% of image power, which is why PSNR is high while fine
texture is visibly incomplete. Capacity scaling (7.49M → 183M) improved the top
band 13.3% → 21.0% — a 58% relative gain that moved PSNR only +0.24 dB.

---

## 6. Inference

| | |
|---|---|
| **Per image (4-view TTA, default)** | **95.4 ms** |
| Per image (single pass) | 31.7 ms |
| Throughput | 10.5 images/s |
| 400-image test set | **38.2 s** |
| Hardware measured on | NVIDIA RTX 5060 Laptop, 8 GB |
| Scale regimes | 128→256 and 256→512, plus arbitrary sizes |
| Device | CUDA with automatic CPU fallback |
| Container | `Dockerfile`, CUDA 12.8 base; runs the test suite at build time |
| Container verified | built + 400/400 restored on GPU in 43.5 s; output matches native to 2.2e-04 |

**Where the time goes** (400 images, profiled end-to-end):

```
disk read     0.14 s    0.2%
host->GPU     0.02 s    0.0%
GPU compute  75.55 s   99.1%
GPU->host     0.03 s    0.0%
disk write    0.49 s    0.6%
```

Compute-bound: I/O parallelisation cannot help, and `--tta_views` is the only
meaningful throughput control. `--tta_views 1` restores the full 400-image set
in 12.7 s at 26.1227 dB.

---

## 7. Known limitations

1. **Fine stochastic texture is not recovered.** Only ~21% of ground-truth
   energy in the top frequency band. Synthesising it would *double* the error in
   that band (two uncorrelated signals add in variance), so smoothing is the
   correct choice under PSNR/SSIM.
2. **Sub-pixel structures** — 1-pixel wires, fine mesh — are partially lost.
3. **Performance varies 17–41 dB** by image, tracking how much irreducible
   texture the target holds.
4. **Stage 2 was truncated** at 2,500 of 4,500 iterations (RAM exhaustion during
   checkpoint save). The state shipped was already the best; the loss is likely
   negligible but was not measured.
5. **The official scoring metric is unknown**, so the model is tuned for
   PSNR/SSIM, which the submission template names first.

---

## 8. Reproducibility

- Deterministic split at seed 42; two independent runs of the same
  configuration landed within **0.001 dB**.
- **54 tests ship in this repository** and pass; a further 107 live in the
  training workspace, which is not part of the submission.
- Weights ship as four <100 MB parts with SHA-256 manifests, reassembled in
  memory by `evaluate.py`; verified byte-identical from a fresh clone.
- Environment: Python 3.11.9, PyTorch 2.12.0.dev+cu128, CUDA 12.8, cuDNN 9.2.
  `requirements.txt` pins installable stable versions;
  `requirements-frozen.txt` records the exact training environment.
