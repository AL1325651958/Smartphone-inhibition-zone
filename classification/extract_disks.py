"""Extract disk crops from the high-resolution plate photos.

The class dataset holds only 60 independent disks from 13 acquisition sessions,
which is why most batches have 1-3 disks.  These 4032x3024 plate photos contain
7 disks each and can supply far more independent disks at much higher resolution.

Output: crops/<photo>_<i>.png  plus a CSV of detection metadata.
"""
from __future__ import annotations

import csv
import re
import sys
from collections import deque
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

HERE = Path(r"F:\Code\SCI\Nanjing_Children_Hospital\抑菌区域分析_返修\药片CNN")
PLATES = Path(r"F:\Code\SCI\Nanjing_Children_Hospital\抑菌区域分析_返修\链球菌检测")
OUT = HERE / "expanded"
CROPS = OUT / "crops"


def detect_disks(im: Image.Image, max_side=900, keep=8):
    scale = max_side / max(im.size)
    small = im.convert("L").resize((int(im.size[0] * scale), int(im.size[1] * scale)),
                                   Image.BILINEAR)
    a = np.asarray(small, dtype=np.float32)
    bg = np.asarray(small.filter(ImageFilter.GaussianBlur(31)), dtype=np.float32)
    hi = a - bg
    mask = hi >= np.percentile(hi, 97)

    H, W = a.shape
    lab = np.zeros(a.shape, dtype=np.int32)
    comps, cur = [], 0
    ys, xs = np.nonzero(mask)
    for y0, x0 in zip(ys, xs):
        if lab[y0, x0]:
            continue
        cur += 1
        q = deque([(y0, x0)])
        lab[y0, x0] = cur
        pix = []
        while q:
            y, x = q.popleft()
            pix.append((y, x))
            for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                yy, xx = y + dy, x + dx
                if 0 <= yy < H and 0 <= xx < W and mask[yy, xx] and not lab[yy, xx]:
                    lab[yy, xx] = cur
                    q.append((yy, xx))
        pix = np.array(pix)
        if len(pix) < 60:
            continue
        h = int(np.ptp(pix[:, 0])) + 1
        w = int(np.ptp(pix[:, 1])) + 1
        if not (0.55 < h / max(w, 1) < 1.8):
            continue
        if len(pix) / (h * w) < 0.35:
            continue
        # disks are similar in size; keep the dominant radius cluster
        comps.append(dict(cx=pix[:, 1].mean() / scale, cy=pix[:, 0].mean() / scale,
                          r=(h + w) / 4 / scale, area=len(pix)))
    comps.sort(key=lambda c: -c["area"])
    comps = comps[:keep]
    if comps:
        rmed = np.median([c["r"] for c in comps])
        comps = [c for c in comps if 0.6 * rmed < c["r"] < 1.6 * rmed]
    return comps


def main():
    CROPS.mkdir(parents=True, exist_ok=True)
    files = sorted([p for p in PLATES.iterdir()
                    if p.suffix.lower() in (".jpg", ".jpeg", ".png")
                    and not p.stem.lower().startswith("r")])
    print(f"plate photos to process: {len(files)}")

    rows = []
    for n, p in enumerate(files, 1):
        with Image.open(p) as im:
            im = im.convert("RGB")
            cands = detect_disks(im)
            W, H = im.size
            for i, c in enumerate(cands):
                pad = c["r"] * 1.25
                box = (int(c["cx"] - pad), int(c["cy"] - pad),
                       int(c["cx"] + pad), int(c["cy"] + pad))
                box = (max(0, box[0]), max(0, box[1]),
                       min(W, box[2]), min(H, box[3]))
                crop = im.crop(box).convert("L")
                # target ~205 px of disk content, matching the readable batches
                side = int(round(crop.size[0] * (205 / (c["r"] * 2 * 1.25))))
                side = max(64, min(512, side))
                crop = crop.resize((side, side), Image.LANCZOS)
                name = f"{p.stem}_{i:02d}.png"
                crop.save(CROPS / name)
                rows.append(dict(photo=p.name, plate=p.stem, idx=i, crop=name,
                                 side=side, cx=round(c["cx"], 1), cy=round(c["cy"], 1),
                                 r=round(c["r"], 1)))
        if n % 10 == 0 or n == len(files):
            print(f"  {n}/{len(files)} photos, {len(rows)} crops")

    with open(OUT / "detections.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    from collections import Counter
    per_photo = Counter(r["plate"] for r in rows)
    print(f"\ncrops {len(rows)} from {len(per_photo)} photos")
    print("disks per photo:", Counter(per_photo.values()))
    sides = [r["side"] for r in rows]
    print(f"crop side px: min {min(sides)} median {int(np.median(sides))} max {max(sides)}")
    print(f"\nwrote {OUT/'detections.csv'}")
    print(f"crops -> {CROPS}")


if __name__ == "__main__":
    main()
