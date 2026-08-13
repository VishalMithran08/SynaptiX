#!/usr/bin/env python3
"""
Image restoration inference — SemiCon AI Hackathon.

Reads every image in an input directory, restores it (denoise + 2x upscale),
and writes the result to an output directory.

    python evaluate.py --input_dir <test images> --output_dir <results>

Design notes
------------
* Runs on GPU when available and falls back to CPU automatically. No edits.
* Handles BOTH scale regimes in the brief (128->256 and 256->512) and any
  other size: the network is fully convolutional and 2x, and PaddedInference
  pads to the /16 stride the encoder requires, then crops back.
* Input is NEVER clipped. Speckle pushes values outside [0,1] and that is
  signal, not corruption. Only the output is clamped, to [0,1] like the GT.
* .npy in -> .npy out at float32, preserving full precision. Image formats in
  -> .png out. Output basenames match input basenames.
* Images are grouped by shape so a batch can be stacked; mixed-size
  directories are handled without falling back to one-at-a-time.
* Reports wall-clock and per-image inference time, since the benchmark scores
  speed as well as quality.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

# Allow running from any working directory.
_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from models.nafnet import NAFNet, PaddedInference, config_for_state  # noqa: E402
from utils.tta import tta_predict  # noqa: E402

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
DEFAULT_WEIGHTS = _ROOT / "weights" / "nafnet160_final.pth"


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def load_image(path: Path) -> torch.Tensor | None:
    """Return [1, H, W] float32, or None if the file cannot be read."""
    try:
        if path.suffix.lower() == ".npy":
            arr = np.load(str(path))
        else:
            from PIL import Image
            img = Image.open(str(path)).convert("L")
            # 8-bit images are stored 0-255; the model works in [0,1].
            arr = np.asarray(img, dtype=np.float32) / 255.0

        arr = np.asarray(arr, dtype=np.float32)
        if arr.ndim == 3:
            arr = arr[..., 0] if arr.shape[-1] <= 4 else arr[0]
        if arr.ndim != 2:
            return None
        return torch.from_numpy(np.ascontiguousarray(arr))[None]
    except Exception as exc:  # noqa: BLE001
        print(f"  [skip] {path.name}: {exc}")
        return None


def save_image(tensor: torch.Tensor, src: Path, out_dir: Path) -> None:
    """.npy -> float32 .npy (lossless). Anything else -> 8-bit .png."""
    arr = tensor.squeeze(0).cpu().numpy()
    if src.suffix.lower() == ".npy":
        np.save(str(out_dir / f"{src.stem}.npy"), arr.astype(np.float32))
    else:
        from PIL import Image
        Image.fromarray(
            (arr * 255.0).clip(0, 255).astype(np.uint8)
        ).save(str(out_dir / f"{src.stem}.png"))


def _read_weights(weights: Path) -> io.BytesIO | str:
    """
    Return something torch.load can read, reassembling split parts if needed.

    The trained model exceeds GitHub's 100 MB file limit, so it is committed as
    `<name>.pth.partNNN` alongside a manifest. Parts are concatenated IN MEMORY
    -- nothing is written to disk, so this works on a read-only checkout and
    leaves no artifacts. Each part and the whole are checksum-verified, because
    a silently truncated reassembly would load as garbage weights rather than
    fail.
    """
    if weights.is_file():
        return str(weights)

    manifest_path = weights.with_suffix(weights.suffix + ".manifest.json")
    if not manifest_path.is_file():
        print(f"ERROR: weights not found: {weights}")
        print(f"       and no manifest at: {manifest_path}")
        print("Hint: the trained model ships in weights/. Pass --weights to "
              "point at a different file.")
        raise SystemExit(1)

    manifest = json.loads(manifest_path.read_text())
    parts = manifest["parts"]
    print(f"Weights  : reassembling {len(parts)} parts "
          f"({manifest['total_bytes'] / 1024**2:.1f} MB) in memory")

    buf = io.BytesIO()
    digest = hashlib.sha256()
    for entry in parts:
        part = weights.parent / entry["name"]
        if not part.is_file():
            print(f"ERROR: missing weight part: {part}")
            print("Hint: ensure the full repository was cloned.")
            raise SystemExit(1)
        data = part.read_bytes()
        if hashlib.sha256(data).hexdigest() != entry["sha256"]:
            print(f"ERROR: checksum mismatch in {part.name} — file is corrupt.")
            raise SystemExit(1)
        buf.write(data)
        digest.update(data)

    if digest.hexdigest() != manifest["sha256"]:
        print("ERROR: reassembled weights failed the whole-file checksum.")
        raise SystemExit(1)

    buf.seek(0)
    return buf


def load_model(weights: Path, device: torch.device) -> PaddedInference:
    # Fail with a readable diagnostic rather than a traceback: this script is
    # run unattended by the benchmarking team, and a bare stack trace gives
    # them nothing actionable.
    weights = Path(weights)
    source = _read_weights(weights)

    try:
        state = torch.load(source, map_location="cpu", weights_only=False)
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: could not read checkpoint {weights}: {exc}")
        raise SystemExit(1) from exc

    if isinstance(state, dict) and "model" in state:
        # Full training payload: prefer the EMA weights, which validate better.
        merged = {k: v.clone() for k, v in state["model"].items()}
        shadow = (state.get("ema") or {}).get("shadow")
        if shadow:
            for k, v in shadow.items():
                merged[k] = v.to(merged[k].dtype)
        state = merged
    elif isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]

    # Weights ship as float16 to keep the file under GitHub's 100MB limit.
    # Measured cost of that rounding: 0.0000 dB PSNR, 0.000004 SSIM. Inference
    # itself still runs in float32, so cast on load.
    state = {k: (v.float() if v.is_floating_point() else v)
             for k, v in state.items()}

    # Width is recovered from the checkpoint itself, so any trained width loads.
    cfg = config_for_state(state)
    model = NAFNet(**cfg)
    try:
        model.load_state_dict(state, strict=True)
    except (RuntimeError, AttributeError, TypeError) as exc:
        print(f"ERROR: checkpoint does not match the architecture: {exc}")
        print("Hint: this usually means models/nafnet.py was changed after "
              "training, or the file is not a NAFNet checkpoint.")
        raise SystemExit(1) from exc

    n = sum(p.numel() for p in model.parameters())
    print(f"Model    : NAFNet width {cfg['width']}  ({n / 1e6:.2f}M params)")
    return PaddedInference(model).to(device).eval()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    p = argparse.ArgumentParser(
        description="Restore degraded images (denoise + 2x super-resolution)."
    )
    p.add_argument("--input_dir", required=True, help="Directory of degraded images.")
    p.add_argument("--output_dir", required=True, help="Where to write results.")
    p.add_argument("--weights", default=str(DEFAULT_WEIGHTS))
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--device", default=None, choices=["cuda", "cpu"])
    p.add_argument(
        "--no_tta",
        action="store_true",
        help="Disable the 8-view self-ensemble. TTA is ON by default: it is "
             "worth +0.16 dB PSNR / +0.005 SSIM and costs 8 forward passes.",
    )
    args = p.parse_args()

    in_dir, out_dir = Path(args.input_dir), Path(args.output_dir)
    if not in_dir.is_dir():
        print(f"ERROR: input directory not found: {in_dir}")
        return 1
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    use_tta = not args.no_tta

    print(f"Device   : {device}"
          + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))
    runner = load_model(Path(args.weights), device)
    print(f"TTA      : {'on (8 views)' if use_tta else 'off'}")

    files = sorted(
        f for f in in_dir.iterdir()
        if f.is_file()
        and not f.name.startswith("._")            # __MACOSX resource forks
        and (f.suffix.lower() == ".npy" or f.suffix.lower() in IMAGE_SUFFIXES)
    )
    if not files:
        print(f"ERROR: no readable images in {in_dir}")
        return 1
    print(f"Found    : {len(files)} images\n")

    # Group by shape so same-sized images can be stacked into one batch.
    buckets: dict[tuple, list[tuple[Path, torch.Tensor]]] = {}
    for f in files:
        t = load_image(f)
        if t is not None:
            buckets.setdefault(tuple(t.shape[-2:]), []).append((f, t))

    done, failed = 0, len(files) - sum(len(v) for v in buckets.values())
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()

    with torch.no_grad():
        for shape, items in sorted(buckets.items()):
            print(f"  {len(items):>4} image(s) at {shape[0]}x{shape[1]} "
                  f"-> {shape[0] * 2}x{shape[1] * 2}")
            bs = args.batch_size
            i = 0
            while i < len(items):
                chunk = items[i:i + bs]
                batch = torch.stack([t for _, t in chunk]).to(device)
                try:
                    out = (tta_predict(runner, batch) if use_tta
                           else runner(batch).float().clamp(0.0, 1.0))
                except torch.cuda.OutOfMemoryError:
                    # Halve the batch and retry rather than aborting the run.
                    torch.cuda.empty_cache()
                    if bs == 1:
                        raise
                    bs = max(1, bs // 2)
                    print(f"       OOM -> batch_size={bs}")
                    continue
                for (src, _), pred in zip(chunk, out):
                    save_image(pred, src, out_dir)
                done += len(chunk)
                i += len(chunk)

    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    print(f"\n{'=' * 58}")
    print(f"Restored          : {done} / {len(files)}")
    if failed:
        print(f"Unreadable        : {failed}")
    print(f"Total time        : {elapsed:.2f} s")
    print(f"Per image         : {elapsed / max(done, 1) * 1000:.1f} ms")
    print(f"Throughput        : {done / max(elapsed, 1e-9):.2f} images/s")
    print(f"Output directory  : {out_dir.resolve()}")
    print(f"{'=' * 58}")
    return 0 if done else 1


if __name__ == "__main__":
    raise SystemExit(main())
