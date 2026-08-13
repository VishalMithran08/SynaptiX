#!/usr/bin/env python3
"""
Split a checkpoint into GitHub-sized parts, with a checksum manifest.

GitHub rejects files over 100 MB. The trained model is ~350 MB at float16, so
it ships as parts that `evaluate.py` reassembles in memory at load time. This
keeps the repository self-contained: no Git LFS (which fails silently to a
pointer file if the reviewer lacks git-lfs) and no external download.

    python tools/split_weights.py --input model.pth --out_dir weights/ \
        --name nafnet160_final --chunk_mb 90

Produces:
    weights/nafnet160_final.pth.part000, .part001, ...
    weights/nafnet160_final.pth.manifest.json   (sha256 + sizes)
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256_of(path: Path, buf: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(buf):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--name", required=True,
                   help="Base name, e.g. nafnet160_final ('.pth' is appended).")
    p.add_argument("--chunk_mb", type=int, default=90,
                   help="Part size in MB. Must stay under GitHub's 100 MB limit.")
    args = p.parse_args()

    src = Path(args.input)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    size = src.stat().st_size
    chunk = args.chunk_mb * 1024 * 1024
    digest = sha256_of(src)

    parts = []
    with open(src, "rb") as f:
        i = 0
        while True:
            data = f.read(chunk)
            if not data:
                break
            part = out / f"{args.name}.pth.part{i:03d}"
            part.write_bytes(data)
            parts.append({"name": part.name, "bytes": len(data),
                          "sha256": hashlib.sha256(data).hexdigest()})
            print(f"  {part.name}  {len(data) / 1024**2:.1f} MB")
            i += 1

    manifest = {
        "target": f"{args.name}.pth",
        "total_bytes": size,
        "sha256": digest,
        "parts": parts,
    }
    mpath = out / f"{args.name}.pth.manifest.json"
    mpath.write_text(json.dumps(manifest, indent=2))

    print(f"\nsource      : {src}  ({size / 1024**2:.1f} MB)")
    print(f"parts       : {len(parts)}")
    print(f"sha256      : {digest}")
    print(f"manifest    : {mpath}")
    biggest = max(p["bytes"] for p in parts) / 1024**2
    print(f"largest part: {biggest:.1f} MB  "
          f"({'OK' if biggest < 100 else 'TOO BIG for GitHub'})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
