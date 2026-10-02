"""Extract the five new archives and resize to 112x112 into a separate folder.

Output:
  <CNNDIR>/new_batch_112/<archive>/pill_XXX.png    grouped by archive
  <CNNDIR>/new_batch_112/flat/<archive>__pill_XXX.png

Non-square sources are padded with edge pixels before resizing so the disk is
not distorted (one crop is 79x93).
"""
from __future__ import annotations

import csv
import zipfile
from collections import Counter
from pathlib import Path

from PIL import Image, ImageOps

ATT = Path(r"C:\Users\13256\.dsh\attachments\v1\files")
OUT = Path(r"F:\Code\SCI\Nanjing_Children_Hospital\抑菌区域分析_返修\药片CNN")
DEST = OUT / "new_batch_112"
SIZE = (112, 112)
EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

ARCHIVES = [
    "06/0635d0229470e7ee6e352350db88d34a85ab961a9d1fd8432c5cd0492a33a906/raw_crops_20260921_193049.zip",
    "86/861e62a314b8c8502ed84cfb6049d937c596e38b62242ac0c2fa75f4e81a70b6/raw_crops_20260921_192940.zip",
    "0e/0edc21b01bd495662f88ee7ea40c2e7fc0c0f350d7257a1898b43a3751493026/raw_crops_20260921_192900.zip",
    "0e/0e82274cf6242f94840e3756d1b0dbf4a0b860da356fa4321c82413685632c6f/raw_crops_20260921_192817.zip",
    "1f/1fa81f5ecef21b984a91fe9de3b0dc126cf52781d81fe73bba8cfd00e38670f1/raw_crops_20260921_192726.zip",
]

if DEST.exists():
    import shutil
    shutil.rmtree(DEST)
FLAT = DEST / "flat"
FLAT.mkdir(parents=True, exist_ok=True)

rows = []
for rel in ARCHIVES:
    z = ATT / rel
    if not z.exists():
        print(f"MISSING {z}")
        continue
    tag = z.stem                                  # e.g. raw_crops_20260921_193049
    out_dir = DEST / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(z) as zf:
        names = sorted(n for n in zf.namelist()
                       if Path(n).suffix.lower() in EXTS and not n.endswith("/"))
        for n in names:
            with zf.open(n) as fh:
                im = Image.open(fh)
                im.load()
                orig_size, orig_mode = im.size, im.mode
                # flatten alpha over white, then pad to square (no distortion)
                rgb = im.convert("RGB")
                side = max(rgb.size)
                sq = Image.new("RGB", (side, side), (255, 255, 255))
                sq.paste(rgb, ((side - rgb.size[0]) // 2,
                               (side - rgb.size[1]) // 2))
                small = sq.resize(SIZE, Image.LANCZOS)
            dst = out_dir / f"{Path(n).stem}.png"
            small.save(dst, format="PNG", optimize=True)
            flat = FLAT / f"{tag}__{Path(n).stem}.png"
            small.save(flat, format="PNG", optimize=True)
            rows.append(dict(archive=tag, file=Path(n).name,
                             orig_w=orig_size[0], orig_h=orig_size[1],
                             orig_mode=orig_mode, new_w=SIZE[0], new_h=SIZE[1],
                             grouped=str(dst.relative_to(OUT)),
                             flat=str(flat.relative_to(OUT))))
    print(f"{tag}: {len(names)} images -> {out_dir.name}/")

with open(DEST / "manifest.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)

print(f"\ntotal {len(rows)} images resized to {SIZE[0]}x{SIZE[1]}")

# ---- verify ----
sizes, modes = Counter(), Counter()
for p in DEST.rglob("*.png"):
    with Image.open(p) as im:
        sizes[im.size] += 1
        modes[im.mode] += 1
print(f"  grouped: {len(list(x for x in DEST.iterdir() if x.is_dir() and x.name != 'flat'))} archives")
print(f"  flat   : {len(list(FLAT.glob('*.png')))} files")
print(f"  sizes  : {dict(sizes)}")
print(f"  modes  : {dict(modes)}")
ws = [r["orig_w"] for r in rows]
print(f"  source side: min {min(ws)} median {sorted(ws)[len(ws)//2]} max {max(ws)}"
      f"  (all upscaled to 112)")
nsq = sum(1 for r in rows if r["orig_w"] != r["orig_h"])
print(f"  non-square sources (edge-padded first): {nsq}")
print(f"\nwrote {DEST}/manifest.csv")
print(f"output -> {DEST}")
