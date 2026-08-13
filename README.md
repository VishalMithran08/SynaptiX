# Image Restoration — SemiCon AI Hackathon

Restores degraded grayscale images: removes speckle and Gaussian noise **and**
upscales 2× in a single pass. NAFNet-based, 29.56M parameters, 54 ms per image.

| Metric (320-image held-out validation, 8-view TTA) | |
|---|---|
| **PSNR** | **26.2934 dB** |
| **SSIM** | **0.792988** |
| **LPIPS** | **0.331840** |
| Inference | **53.8 ms/image** (RTX 5060 Laptop, incl. TTA) |
| Model size | 56.5 MB (float16 storage, float32 inference) |

---

## Quick start

```bash
git clone <this-repo>
cd <this-repo>
pip install -r requirements.txt

python evaluate.py --input_dir /path/to/test_images --output_dir /path/to/results
```

That is the whole procedure. The trained weights are committed in `weights/`
(56.5 MB, no Git LFS or external download needed) and no paths need editing.

> Weights are **stored** as float16 to stay under GitHub's 100 MB per-file limit
> and are cast to float32 at load; inference runs in float32 throughout. The
> measured cost of that rounding is **0.0000 dB PSNR and 0.000004 SSIM** — i.e.
> nothing.

### Inference options

| Flag | Default | Notes |
|---|---|---|
| `--input_dir` | *(required)* | Directory of degraded images |
| `--output_dir` | *(required)* | Created if absent |
| `--weights` | `weights/nafnet64_final.pth` | |
| `--batch_size` | `8` | Halves automatically on CUDA OOM |
| `--device` | auto | `cuda` if available, else `cpu` |
| `--no_tta` | off | Disables the 8-view self-ensemble (~8× faster, −0.16 dB) |

**Input/output behaviour**

* `.npy` in → `.npy` out, float32, full precision. Images (`.png/.jpg/.tif/…`)
  in → `.png` out. Output basenames match input basenames.
* Handles **both** scale regimes in the brief — 128→256 and 256→512 — plus any
  other size, including dimensions not divisible by 16.
* A directory containing mixed sizes is handled in one run.
* Input is **never clipped**. Speckle pushes values outside `[0,1]` and that is
  signal; only the output is clamped to `[0,1]`, matching the ground truth.

---

## Model

**NAFNet** (Chen et al., *Simple Baselines for Image Restoration*, ECCV 2022),
adapted for 2× super-resolution.

```
input  [B, 1, H, W]        float32, may exceed [0,1] — never clipped
  → 3×3 conv  (1 → 64)
  → encoder   4 stages, ch [64,128,256,512], blocks [2,2,4,8]
  → bottleneck 12 NAFBlocks
  → decoder   4 stages, blocks [2,2,2,2], skip-concat + 1×1 reduce
  → SR tail   conv(64→4, ICNR-init) + PixelShuffle(2)
  → clamp(0,1)                       ← the only clamp in the forward pass
output [B, 1, 2H, 2W]
```

Each NAFBlock uses **SimpleGate** (channel-split multiply) instead of a
nonlinear activation, plus channel attention and channel-wise LayerNorm.

Two deliberate departures from the paper: a **PixelShuffle SR tail with ICNR
initialisation** (the original NAFNet is same-resolution; ICNR prevents
checkerboard artifacts from the upsampler), and single-channel I/O for
grayscale.

**Why NAFNet.** It handles all three degradations in one pass — the U-Net's
multi-scale receptive field captures global noise statistics while skip
connections preserve the fine structure super-resolution must reconstruct, so
no error-compounding denoise-then-upscale pipeline. It is also fast: no
self-attention, only depthwise and 1×1 convolutions, which keeps inference at
54 ms/image where a transformer of similar quality would cost several times
more.

---

## Reproducing training

```bash
# 0. Verify the dataset is complete before training on it
python tools/verify_dataset.py /path/to/train_root

# 1. Stage 1 — L1 warmup, 27k iterations (~2.7 h on RTX 5060)
python train.py --data_root /path/to/train_root \
    --save_dir checkpoints_stage1 --width 64 --batch_size 12 \
    --amp_dtype bf16 --seed 42 --max_iters 27000

# 2. Stage 2 — fine-tune with SSIM/FFT/edge/perceptual, 6k iterations (~50 min)
python train.py --data_root /path/to/train_root \
    --save_dir checkpoints_stage2 --width 64 --batch_size 12 \
    --amp_dtype bf16 --seed 42 --val_every 1000 \
    --resume checkpoints_stage1/<15k-checkpoint>.pth --max_iters 21000
```

`--data_root` must contain `train/NoisyLR/` and `train/GT/` as matched `.npy`
pairs. Split is deterministic at seed 42: **train 2720 / val 320 / val_hard 160**.

**Stage 1** — pure L1, difficulty 0.20→0.40, lr 2e-4→2e-6.
**Stage 2** — `L1 1.0 + SSIM 0.10 + FFT 0.05 + edge 0.05 + perceptual 0.01`,
difficulty fixed 0.30, lr 1e-4→1e-6. The shipped weights are stage 2 at 3k.

The `--width` flag sets base channels; checkpoints record their own width and
load back automatically, so width-32 and width-64 models both work everywhere.

---

## Repository layout

```
evaluate.py              inference entry point  ← the script to run
train.py                 training pipeline (all phases/losses)
models/
  nafnet.py              architecture + PaddedInference (arbitrary sizes)
  losses.py              L1, MS-SSIM, FFT, Sobel edge, VGG perceptual
data/dataset.py          .npy loader, augmentation, deterministic splits
utils/
  tta.py                 8-view dihedral self-ensemble
  metrics.py             PSNR, SSIM, LPIPS
  ema.py                 exponential moving average of weights
tools/
  verify_dataset.py      dataset integrity gate
  eval_report.py         checkpoint evaluation (RAW/EMA, TTA, LPIPS, spectral)
  ood_probe.py           robustness under degradation shift
  ensemble.py            weight averaging / prediction ensembling
  checkerboard.py        quantifies PixelShuffle artifacts
tests/                   34 unit tests — `python -m pytest tests -q`
weights/nafnet64_final.pth
outputs/                 restored test-set images
docs/METHODOLOGY.md      full experimental record
```

---

## Method summary

Selected from **nine trained models** across a controlled ablation. Full record
in [`docs/METHODOLOGY.md`](docs/METHODOLOGY.md).

| Change | Gain |
|---|---|
| Dataset integrity fix (359 → 3200 usable pairs) | **+0.60 dB** |
| Capacity: width 32 → 64 | **+0.13 dB** |
| 8-view test-time augmentation | **+0.16 dB** |
| Perceptual weight sweep → 0.01 | LPIPS −10%, SSIM +0.002 |

Findings worth noting:

* **The largest single win was a data-integrity check, not a model change.** The
  supplied extraction contained 359 of 3200 matched pairs; `tools/verify_dataset.py`
  now gates against that.
* **Predictions were over-smoothed, not over-sharpened.** Measured against the
  ground truth's own spectral statistics (edge 0.309, HF 0.0108) the model sat at
  ~30% of GT high-frequency energy, so training added frequency pressure rather
  than removing it.
* **The difficulty curriculum has a capacity-dependent harm threshold** — ~0.48
  at width 32, ~0.31 at width 64 — found by controlled ablation.
* **Perceptual loss was tuned, not assumed.** Weights 0.05 / 0.02 / 0.01 were each
  trained and measured; 0.01 gave the best composite. `tools/checkerboard.py`
  confirmed no model exceeds ground-truth high-frequency content, i.e. none
  fabricates periodic structure.

---

## Limitations

* Fine stochastic texture and sub-pixel structures (thin wires, fine grain) are
  not recovered. After 2× downsampling plus speckle that information is absent
  from the input; L1-family objectives return the conditional mean, which is
  smooth. Closing this gap requires adversarial training, which fabricates
  plausible-but-invented detail — inappropriate for an inspection task and
  explicitly cautioned against in the brief.
* Per-image PSNR ranges from ~17 dB to ~41 dB depending on how much irreducible
  texture the target contains. The 26.29 dB average conceals that spread.

---

## References

1. Chen et al. **Simple Baselines for Image Restoration.** ECCV 2022. *(NAFNet)*
2. Blau & Michaeli. **The Perception-Distortion Tradeoff.** CVPR 2018.
3. Lim et al. **Enhanced Deep Residual Networks for Single Image Super-Resolution.**
   CVPRW 2017. *(geometric self-ensemble / TTA)*
4. Aitken et al. **Checkerboard artifact free sub-pixel convolution.** 2017. *(ICNR)*
5. Zhang et al. **The Unreasonable Effectiveness of Deep Features as a Perceptual
   Metric.** CVPR 2018. *(LPIPS)*
