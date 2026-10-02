"""Create 112x112 versions of every disk crop.

Sets produced:
  labelstudio/files112/<plate>/<crop>.png      621 crops, for CNN input
  <raw>/extracted112/<archive>/pill_XXX.png    137 crops, grouped
  <raw>/all112/<archive>__pill_XXX.png         137 crops, flat

The 621 crops are resized from the ORIGINAL crops (not the 300 px upscales) so
no detail is lost twice.  The 300x300 sets are left in place: the annotation UI
keeps those, because at 112 px the printed abbreviations are hard to read.
"""
from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path

from PIL import Image

SIZE = (112, 112)
LS = Path(r"F:\Code\SCI\Nanjing_Children_Hospital\抑菌区域分析_返修\药片CNN\labelstudio")
RAW = Path(r"F:\Code\SCI\Nanjing_Children_Hospital\抑菌区域分析_返修\药片分类数据")
EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
rows = []

# ---------------------------------------------------------------- 621 crops
src = LS / "files"                     # originals (~205 px)
dst = LS / "files112"
dst.mkdir(parents=True, exist_ok=True)
n = 0
for plate_dir in sorted(p for p in src.iterdir() if p.is_dir()):
    out = dst / plate_dir.name
    out.mkdir(parents=True, exist_ok=True)
    for img in sorted(p for p in plate_dir.iterdir() if p.suffix.lower() in EXTS):
        with Image.open(img) as im:
            orig = im.size
            im.convert("RGB").resize(SIZE, Image.LANCZOS).save(
                out / (img.stem + ".png"), format="PNG", optimize=True)
        rows.append(dict(set="labelstudio", plate=plate_dir.name, file=img.name,
                         orig_w=orig[0], orig_h=orig[1], new=112))
        n += 1
print(f"labelstudio: {n} crops -> {dst}")

# ---------------------------------------------------------------- 137 crops
g_src = RAW / "extracted"
g_dst = RAW / "extracted112"
f_dst = RAW / "all112"
g_dst.mkdir(parents=True, exist_ok=True)
f_dst.mkdir(parents=True, exist_ok=True)
m = 0
for arch in sorted(p for p in g_src.iterdir() if p.is_dir()):
    out = g_dst / arch.name
    out.mkdir(parents=True, exist_ok=True)
    for img in sorted(p for p in arch.iterdir() if p.suffix.lower() in EXTS):
        with Image.open(img) as im:
            orig = im.size
            small = im.convert("RGB").resize(SIZE, Image.LANCZOS)
        small.save(out / (img.stem + ".png"), format="PNG", optimize=True)
        small.save(f_dst / f"{arch.name}__{img.stem}.png", format="PNG",
                   optimize=True)
        rows.append(dict(set="raw_crops", plate=arch.name, file=img.name,
                         orig_w=orig[0], orig_h=orig[1], new=112))
        m += 1
print(f"raw crops : {m} crops -> {g_dst} and {f_dst}")

# ---------------------------------------------------------------- verify
def check(d):
    sizes, modes = Counter(), Counter()
    for p in Path(d).rglob("*.png"):
        with Image.open(p) as im:
            sizes[im.size] += 1
            modes[im.mode] += 1
    return sizes, modes

for d in (LS / "files112", RAW / "extracted112", RAW / "all112"):
    s, mo = check(d)
    print(f"  {d.name}: sizes={dict(s)} modes={dict(mo)}")

with open(RAW / "resize112_manifest.csv", "w", newline="", encoding="utf-8") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)
print(f"\ntotal {len(rows)} crops resized; manifest -> {RAW/'resize112_manifest.csv'}")
