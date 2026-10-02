"""Resize the extracted pill crops to a uniform 300x300.

Writes two mirrored sets so the originals stay untouched:
  <root>/extracted300/<archive>/pill_XXX.png
  <root>/all300/<archive>__pill_XXX.png
along with a manifest recording original and new sizes.

Note: upscaling cannot recover detail that is not in the source file (the small
063258 crops are 104 px), so the 300 px canvases are not equally sharp.
"""
from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path

from PIL import Image

RAW = Path(r"F:\Code\SCI\Nanjing_Children_Hospital\抑菌区域分析_返修\药片分类数据")
SRC_GROUPED = RAW / "extracted"
SRC_FLAT = RAW / "all"
DST_GROUPED = RAW / "extracted300"
DST_FLAT = RAW / "all300"
SIZE = (300, 300)
EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

for d in (DST_GROUPED, DST_FLAT):
    d.mkdir(parents=True, exist_ok=True)

rows = []
for arch in sorted(p for p in SRC_GROUPED.iterdir() if p.is_dir()):
    out_dir = DST_GROUPED / arch.name
    out_dir.mkdir(parents=True, exist_ok=True)
    for img in sorted(p for p in arch.iterdir() if p.suffix.lower() in EXTS):
        with Image.open(img) as im:
            orig = im.size
            rgb = im.convert("RGB")
            resized = rgb.resize(SIZE, Image.LANCZOS)
        dst = out_dir / img.name
        resized.save(dst, format="PNG", optimize=True)
        flat = DST_FLAT / f"{arch.name}__{img.name}"
        resized.save(flat, format="PNG", optimize=True)
        rows.append(dict(archive=arch.name, file=img.name,
                         orig_w=orig[0], orig_h=orig[1],
                         new_w=SIZE[0], new_h=SIZE[1],
                         upscaled=int(SIZE[0] > max(orig)),
                         grouped=str(dst.relative_to(RAW)),
                         flat=str(flat.relative_to(RAW)),
                         bytes=dst.stat().st_size))

with open(RAW / "resize300_manifest.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)

print(f"resized {len(rows)} images to {SIZE[0]}x{SIZE[1]}")
print(f"  grouped -> {DST_GROUPED}")
print(f"  flat    -> {DST_FLAT}  ({len(list(DST_FLAT.glob('*.png')))} files)")

ws = [r["orig_w"] for r in rows]
up = sum(r["upscaled"] for r in rows)
print(f"\noriginal side length: min {min(ws)}  median {int(sorted(ws)[len(ws)//2])} "
      f"max {max(ws)}")
print(f"upscaled (original smaller than 300): {up}/{len(rows)} "
      f"({up/len(rows):.0%})  <- these gain no real detail")

buckets = Counter()
for w in ws:
    for lo, hi in ((0, 150), (150, 200), (200, 260), (260, 400)):
        if lo <= w < hi:
            buckets[f"{lo}-{hi}px"] += 1
print("source size buckets:", dict(buckets))

# verify every output
bad = 0
for r in rows[:20] + rows[-20:]:
    with Image.open(RAW / r["grouped"]) as im:
        if im.size != SIZE or im.mode != "RGB":
            bad += 1
print(f"\nverification (40 sampled): mismatches {bad}")
print(f"wrote {RAW/'resize300_manifest.csv'}")
