#!/usr/bin/env python3
"""
Verify a semiconductor training-data extraction before it is used for training.

Usage:
    python verify_dataset.py D:/semicon/train_new

Expects <root>/train/GT and <root>/train/NoisyLR (the layout the competition
archive produces).  Checks pair count, ID coverage, shapes and dtypes, and
reports exactly which IDs are missing so a partial extraction cannot slip
through unnoticed.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np

EXPECTED_PAIRS = 3200
GT_SHAPE = (256, 256)
LR_SHAPE = (128, 128)

_ID = re.compile(r"^(\d+)")


def ids(d: Path) -> dict[int, Path]:
    out: dict[int, Path] = {}
    for f in d.glob("*.npy"):
        if f.name.startswith("._"):        # __MACOSX resource forks
            continue
        m = _ID.match(f.stem)
        if m:
            out[int(m.group(1))] = f
    return out


def summarize(label: str, missing: list[int]) -> None:
    if not missing:
        return
    head = ", ".join(str(i) for i in missing[:10])
    tail = " ..." if len(missing) > 10 else ""
    print(f"    {label}: {len(missing)} -> {head}{tail}")


def main() -> int:
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "D:/semicon/train")
    gt_dir, lr_dir = root / "train" / "GT", root / "train" / "NoisyLR"

    for d in (gt_dir, lr_dir):
        if not d.is_dir():
            print(f"FAIL: missing directory {d}")
            return 1

    gt, lr = ids(gt_dir), ids(lr_dir)
    paired = sorted(set(gt) & set(lr))

    print(f"root            : {root}")
    print(f"GT files        : {len(gt)}")
    print(f"NoisyLR files   : {len(lr)}")
    print(f"matched pairs   : {len(paired)}  (expected {EXPECTED_PAIRS})")

    ok = True

    if len(paired) != EXPECTED_PAIRS:
        ok = False
        print("\nFAIL: pair count mismatch. Extraction is incomplete.")
        expected = set(range(EXPECTED_PAIRS))
        summarize("IDs absent from both dirs", sorted(expected - set(gt) - set(lr)))
        summarize("IDs with GT but no NoisyLR", sorted(set(gt) - set(lr)))
        summarize("IDs with NoisyLR but no GT", sorted(set(lr) - set(gt)))

    # Shape/dtype spot-check across the full ID range, not just the first few.
    if paired:
        step = max(1, len(paired) // 50)
        bad = []
        for i in paired[::step]:
            a, b = np.load(gt[i]), np.load(lr[i])
            if a.shape != GT_SHAPE or a.dtype != np.float32:
                bad.append(f"GT {i}: {a.shape} {a.dtype}")
            if b.shape != LR_SHAPE or b.dtype != np.float32:
                bad.append(f"LR {i}: {b.shape} {b.dtype}")
            if not np.isfinite(a).all() or not np.isfinite(b).all():
                bad.append(f"non-finite values in pair {i}")
        print(f"\nsampled {len(paired[::step])} pairs for shape/dtype/finiteness")
        if bad:
            ok = False
            print("FAIL: malformed samples")
            for line in bad[:20]:
                print(f"    {line}")
        else:
            print(f"    all sampled GT {GT_SHAPE} float32, LR {LR_SHAPE} float32, finite")

    print("\n" + ("PASS - safe to train on." if ok else "DO NOT TRAIN on this extraction."))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
