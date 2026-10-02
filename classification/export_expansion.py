"""Export the expanded disk set: verified labels + a montage for manual labelling.

The spreadsheet pins the drug set for 12 plates only.  For the remaining 20
plates the crops are correct but the labels are unknown, so they are laid out
per plate (7 crops per row) at readable size for a human to label in a few
minutes.
"""
import csv
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image, ImageDraw

HERE = Path(r"F:\Code\SCI\Nanjing_Children_Hospital\抑菌区域分析_返修\药片CNN")
OUT = HERE / "expanded"
CLASSES = ["CRO", "DA", "E", "LEV", "LZD", "P", "VA"]
XL = ['002', '003', '005', '006', '009', '011', '012', '013', '016', '017', '018', '019']

det = list(csv.DictReader(open(OUT / "detections.csv", encoding="utf-8")))
lab = list(csv.DictReader(open(OUT / "labelled_crops.csv", encoding="utf-8")))
lab_by_crop = {r["crop"]: r for r in lab}

by_plate = defaultdict(list)
for r in det:
    by_plate[r["plate"]].append(r)

matched = sorted(p for p in by_plate if p.split("_")[0] in XL)
unmatched = sorted(p for p in by_plate if p.split("_")[0] not in XL)
print(f"plates with verified labels : {len(matched)}  "
      f"({sum(len(by_plate[p]) for p in matched)} crops)")
print(f"plates needing manual labels: {len(unmatched)}  "
      f"({sum(len(by_plate[p]) for p in unmatched)} crops)")

# ---------------------------------------------------------------- summary
print("\n=== verified expanded set ===")
c = Counter(r["label"] for r in lab)
disk = defaultdict(set)
for r in lab:
    disk[r["label"]].add(r["photo"])
print(f"{'class':6} {'crops':>6} {'independent disks':>18}")
for k in CLASSES:
    print(f"{k:6} {c.get(k,0):>6} {len(disk[k]):>18}")
print(f"{'total':6} {len(lab):>6} {len({r['photo'] for r in lab}):>18}")

# ---------------------------------------------------------------- montage
CELL = 165
LABEL_H = 30
ROWS_PER_SHEET = 10
sheets = 0
items = unmatched
for s in range(0, len(items), ROWS_PER_SHEET):
    chunk = items[s:s + ROWS_PER_SHEET]
    nmax = max(len(by_plate[p]) for p in chunk)
    canvas = Image.new("RGB", (nmax * CELL, len(chunk) * (CELL + LABEL_H)), "white")
    d = ImageDraw.Draw(canvas)
    for ri, p in enumerate(chunk):
        rs = sorted(by_plate[p], key=lambda r: int(r["idx"]))
        y = ri * (CELL + LABEL_H)
        d.text((4, y + 2), f"plate {p}", fill=(0, 0, 0))
        for ci, r in enumerate(rs):
            im = Image.open(OUT / "crops" / r["crop"]).convert("L").resize((CELL, CELL))
            canvas.paste(im.convert("RGB"), (ci * CELL, y + LABEL_H - 8))
    out = OUT / f"manual_label_sheet_{s//ROWS_PER_SHEET + 1}.png"
    canvas.save(out)
    sheets += 1
    print(f"  wrote {out.name}: {len(chunk)} plates "
          f"({chunk[0]} .. {chunk[-1]})")
print(f"\n{sheets} labelling sheets -> {OUT}")
print("Each row is one plate's detected disks, in the order they were detected.")
print("Write the drug for each disk; the plate has 6-8 disks drawn from the")
print("7 drugs CRO, DA, E, LEV, LZD, P, VA.")
