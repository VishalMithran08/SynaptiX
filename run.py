#!/usr/bin/env python3
"""
Required competition entry point.

    python run.py <input-dir> <output-dir>

Reads every `.npy` in <input-dir>, restores it (speckle + Gaussian denoise and
2x super-resolution in a single pass), and writes one `.npy` per input to
<output-dir> under the same filename.

Runs entirely offline: the weights are committed to this repository as four
checksum-verified parts and reassembled in memory at load time. No download,
no API key, no interactive prompt, no manual configuration. CUDA is used when
available and the CPU path is an automatic fallback.

Every written array is validated before the run is reported as successful:
    * dtype float32, shape (H, W)
    * exactly 2x the input height and width
    * finite -- no NaN, no Inf
    * inside [0, 1]

A non-zero exit status means at least one file failed; the reason is printed.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from evaluate import load_model, save_image  # noqa: E402
from utils.tta import tta_predict  # noqa: E402

# Weights normally live in weights/. models/ is accepted too, so the layout can
# be rearranged to match a grader's expected tree without editing this file.
_WEIGHT_CANDIDATES = (
    _ROOT / "weights" / "nafnet160_final.pth",
    _ROOT / "models" / "nafnet160_final.pth",
)


def find_weights() -> Path:
    """First candidate that exists either whole or as split parts."""
    for cand in _WEIGHT_CANDIDATES:
        if cand.exists() or sorted(cand.parent.glob(cand.name + ".part*")):
            return cand
    searched = "\n".join(f"    {c}" for c in _WEIGHT_CANDIDATES)
    raise SystemExit(
        "ERROR: model weights not found. Looked for these, whole or as "
        f".partNNN pieces:\n{searched}\n"
        "The weights ship with this repository -- if they are missing the "
        "clone is incomplete."
    )


def load_npy(path: Path) -> np.ndarray | None:
    """Load one input as a 2-D float32 array, or None if unreadable."""
    try:
        arr = np.asarray(np.load(str(path)), dtype=np.float32)
    except Exception as exc:  # noqa: BLE001
        print(f"  [SKIP] {path.name}: cannot read ({exc})")
        return None

    # Accept (H,W), (H,W,1) and (1,H,W); anything else is not a grayscale image.
    if arr.ndim == 3:
        if arr.shape[-1] == 1:
            arr = arr[..., 0]
        elif arr.shape[0] == 1:
            arr = arr[0]
    if arr.ndim != 2:
        print(f"  [SKIP] {path.name}: expected a 2-D grayscale array, got "
              f"shape {arr.shape}")
        return None

    # Input is deliberately NOT clipped: speckle pushes values outside [0,1]
    # and that is signal, not corruption. Only the output is clamped.
    return arr


def check_output(out_path: Path, src_shape: tuple[int, int]) -> str | None:
    """Re-read a written file and verify it. Returns an error string or None."""
    try:
        arr = np.load(str(out_path))
    except Exception as exc:  # noqa: BLE001
        return f"cannot be read back ({exc})"

    if arr.dtype != np.float32:
        return f"dtype is {arr.dtype}, expected float32"
    if arr.ndim != 2:
        return f"shape is {arr.shape}, expected 2-D (H, W)"

    want = (src_shape[0] * 2, src_shape[1] * 2)
    if arr.shape != want:
        return f"shape is {arr.shape}, expected {want} (2x the input)"
    if not np.isfinite(arr).all():
        n = int((~np.isfinite(arr)).sum())
        return f"contains {n} non-finite value(s) (NaN or Inf)"
    if arr.min() < 0.0 or arr.max() > 1.0:
        return f"values span [{arr.min():.4f}, {arr.max():.4f}], outside [0, 1]"
    return None


def main() -> int:
    p = argparse.ArgumentParser(
        description="Restore degraded .npy images (denoise + 2x super-resolution).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="example:\n  python run.py ./test_images ./results",
    )
    p.add_argument("input_dir", help="Directory containing degraded .npy files")
    p.add_argument("output_dir", help="Directory for restored .npy files "
                                      "(created if it does not exist)")
    p.add_argument("--batch_size", type=int, default=8,
                   help="Halves automatically on CUDA out-of-memory (default 8)")
    p.add_argument("--tta_views", type=int, default=4, choices=[1, 2, 4, 8],
                   help="Self-ensemble size; 1 is fastest, 8 is highest PSNR "
                        "(default 4)")
    p.add_argument("--device", default=None, help="cuda or cpu (default: auto)")
    args = p.parse_args()

    in_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir)

    if not in_dir.is_dir():
        print(f"ERROR: input directory does not exist: {in_dir}")
        return 1

    # Requirement: create the output directory if it is not already there.
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(f for f in in_dir.iterdir()
                   if f.suffix.lower() == ".npy" and not f.name.startswith("._"))
    if not files:
        print(f"ERROR: no .npy files found in {in_dir}")
        return 1

    device = torch.device(args.device) if args.device else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 58)
    print("  Image Restoration  |  SynaptiX")
    print("=" * 58)
    print(f"Input    : {in_dir}")
    print(f"Output   : {out_dir}")
    print(f"Device   : {device}"
          + (f"  ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))

    runner = load_model(find_weights(), device)
    print(f"TTA      : {args.tta_views} view(s)")
    print(f"Found    : {len(files)} .npy file(s)")
    print()

    written, failures, skipped = 0, [], []
    batch_size = max(1, args.batch_size)
    t0 = time.perf_counter()
    i = 0

    with torch.no_grad():
        while i < len(files):
            chunk = files[i:i + batch_size]
            loaded = [(f, load_npy(f)) for f in chunk]
            usable = [(f, a) for f, a in loaded if a is not None]
            skipped.extend(f.name for f, a in loaded if a is None)
            i += len(chunk)
            if not usable:
                continue

            # Group by shape so a directory of mixed sizes still batches safely.
            groups: dict[tuple[int, int], list] = {}
            for f, a in usable:
                groups.setdefault(a.shape, []).append((f, a))

            for shape, items in groups.items():
                batch = torch.from_numpy(
                    np.stack([a for _, a in items])).unsqueeze(1).to(device)
                try:
                    out = (tta_predict(runner, batch, views=args.tta_views)
                           if args.tta_views > 1
                           else runner(batch).float().clamp(0.0, 1.0))
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    if batch_size > 1:
                        batch_size = max(1, batch_size // 2)
                        print(f"  [note] CUDA OOM -- batch size -> {batch_size}, retrying")
                        i -= len(chunk)
                        break
                    raise
                out = out.float().clamp(0.0, 1.0).cpu()

                for (f, arr), pred in zip(items, out):
                    save_image(pred, f, out_dir)
                    dest = out_dir / f"{f.stem}.npy"
                    err = check_output(dest, shape)
                    if err:
                        failures.append(f"{f.name}: {err}")
                    else:
                        written += 1

            done = min(i, len(files))
            print(f"\r  restored {done}/{len(files)}", end="", flush=True)

    dt = time.perf_counter() - t0
    print("\n")
    print("=" * 58)
    print(f"Inputs found      : {len(files)}")
    print(f"Restored + checked: {written}")
    if skipped:
        print(f"Unreadable        : {len(skipped)}  ({', '.join(skipped[:5])}"
              + (" ..." if len(skipped) > 5 else "") + ")")
    if failures:
        print(f"FAILED validation : {len(failures)}")
        for msg in failures[:10]:
            print(f"    - {msg}")
    print(f"Total time        : {dt:.2f} s")
    if written:
        print(f"Per image         : {dt / written * 1000:.1f} ms")
    print("=" * 58)

    if failures or skipped or written != len(files):
        print("\nRESULT: FAILED -- not every input produced a valid output.")
        return 1
    print("\nRESULT: OK -- one validated .npy written for every input.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
