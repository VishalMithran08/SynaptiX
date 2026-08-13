# Image Restoration — SemiCon AI Hackathon

Restores degraded grayscale images: removes speckle and Gaussian noise **and**
upscales 2× in a single pass. NAFNet-based, 183.16M parameters, 168 ms per image.

| Metric (320-image held-out validation, 8-view TTA) | |
|---|---|
| **PSNR** | **26.4093 dB** |
| **SSIM** | **0.795909** |
| **LPIPS** | **0.319032** |
| Hard subset (160 images) | 27.8010 dB / 0.840699 SSIM |
| Inference | **168.4 ms/image** (RTX 5060 Laptop, incl. TTA) |
| Model size | 349.5 MB (float16 storage, float32 inference) |

---

## Quick start

```bash
git clone <this-repo>
cd <this-repo>
pip install -r requirements.txt

python evaluate.py --input_dir /path/to/test_images --output_dir /path/to/results
```

That is the whole procedure. Everything needed is committed to this repository
— **no Git LFS, no external download, no manual steps.**

> **How the weights are packaged.** At 183M parameters the model is 349.5 MB,
> over GitHub's 100 MB per-file limit. Rather than depend on Git LFS (which
> silently degrades to a pointer file if the reviewer lacks `git-lfs`) or an
> external download, the weights ship as four `<100 MB` parts with a SHA-256
> manifest. `evaluate.py` reassembles them **in memory** at load time — nothing
> is written to disk, so it works on a read-only checkout, and every part plus
> the whole file is checksum-verified so a truncated clone fails loudly instead
> of loading garbage. Verified byte-identical to the unsplit file.
>
> Weights are stored float16 and cast to float32 at load; inference runs in
> float32 throughout. Measured cost of that rounding: **0.0000 dB PSNR**.

### Inference options

| Flag | Default | Notes |
|---|---|---|
| `--input_dir` | *(required)* | Directory of degraded images |
| `--output_dir` | *(required)* | Created if absent |
| `--weights` | `weights/nafnet160_final.pth` | |
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
adapted for 2× super-resolution. Width 160, 183.16M parameters — selected by a
measured capacity study across 7.49M / 29.56M / 183.16M.

```
input  [B, 1, H, W]        float32, may exceed [0,1] — never clipped
  → 3×3 conv  (1 → 160)
  → encoder   4 stages, ch [160,320,640,1280], blocks [2,2,4,8]
  → bottleneck 12 NAFBlocks
  → decoder   4 stages, blocks [2,2,2,2], skip-concat + 1×1 reduce
  → SR tail   conv(160→4, ICNR-init) + PixelShuffle(2)
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
168 ms/image where a transformer of similar quality would cost several times
more.

---

## Reproducing training

```bash
# 0. Verify the dataset is complete before training on it
python tools/verify_dataset.py /path/to/train_root

# 1. Stage 1 — L1 warmup, 22.5k iterations (~5.5 h on an 8GB RTX 5060)
python train.py --data_root /path/to/train_root \
    --save_dir checkpoints_w160 --width 160 --batch_size 8 \
    --amp_dtype bf16 --seed 42 --grad_checkpoint \
    --max_iters 22500 --val_every 2500

# 2. Stage 2 — fine-tune with SSIM/FFT/edge/perceptual (~1.1 h)
#    Continues in the same directory; the phase advances automatically.
python train.py --data_root /path/to/train_root \
    --save_dir checkpoints_w160 --width 160 --batch_size 8 \
    --amp_dtype bf16 --seed 42 --grad_checkpoint --val_every 1000 \
    --resume checkpoints_w160/latest.pth --max_iters 27000
```

`--grad_checkpoint` is **required** at width 160 on an 8 GB card: it recomputes
block activations during backward rather than storing them, cutting peak VRAM
from over 8 GB to 5.3 GB. It affects memory only — inference output is
bit-identical and gradients match to 1e-5 (`tests/test_gradcheckpoint.py`).

> **Note on batch size and iteration counts.** Iteration counts are not
> comparable across batch sizes; sample presentations are. The schedule above is
> 22,500 x 8 = 180k presentations for stage 1 and 4,500 x 8 = 36k for stage 2.

`--data_root` must contain `train/NoisyLR/` and `train/GT/` as matched `.npy`
pairs. Split is deterministic at seed 42: **train 2720 / val 320 / val_hard 160**.

**Stage 1** — pure L1, difficulty 0.20→0.40, lr 2e-4→2e-6.
**Stage 2** — `L1 1.0 + SSIM 0.10 + FFT 0.05 + edge 0.05 + perceptual 0.01`,
difficulty fixed 0.30, lr 1e-4→1e-6. The shipped weights are stage 2 at 2.5k.

The `--width` flag sets base channels; checkpoints record their own width and
load back automatically, so models of any width work everywhere.

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
  gen_gap.py             train/val generalisation gap
  split_weights.py       splits weights into <100MB parts
tests/                   41 unit tests — `python -m pytest tests -q`
weights/                 model in 4 checksum-verified parts + manifest
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
| Capacity: width 32 → 64 → 160 | **+0.24 dB** |
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
* **Capacity was the binding constraint, and it never saturated.** At 183M
  parameters on 2,720 images the train/val gap is still **-1.5%** — the model does
  no better on data it trained on than on data it has never seen.
* **The difficulty curriculum has a capacity-dependent harm threshold** — ~0.48
  at width 32, ~0.31 at width 64 — found by controlled ablation.
* **Gradient checkpointing made the 183M model trainable on 8 GB.** Peak VRAM
  5.3 GB instead of over 8; measured that throughput collapses past ~6 GB peak on
  this card, so the batch size targets ~80% utilisation rather than filling it.
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
