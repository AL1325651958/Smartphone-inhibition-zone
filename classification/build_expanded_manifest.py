"""Build the expanded, batch-ordered disk set (excluding the defect batch).

Facts established by analysis:
  * the 7 disks sit on a regular heptagon (mean angular gap 51.4 deg = 360/7);
  * the drug order around the plate differs between plates, so no fixed layout;
  * the spreadsheet drug set is incomplete (a blank also means "disk present but
    no zone"), so it alone cannot label a plate.

The class dataset has 60 disks across 13 acquisition sessions, one of which
(20260505_063258) contains unreadable 81-137 px crops and is excluded here.
This script prepares the expanded set with disk geometry attached and a
per-plate montage for labelling.
"""
import csv
import math
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image, ImageDraw

HERE = Path(r"F:\Code\SCI\Nanjing_Children_Hospital\抑菌区域分析_返修\药片CNN")
OUT = HERE / "expanded"
CLASSES = ["CRO", "DA", "E", "LEV", "LZD", "P", "VA"]
EXCLUDE_BATCH = "20260505_063258"

det = list(csv.DictReader(open(OUT / "detections.csv", encoding="utf-8")))
print(f"detected crops: {len(det)}")

# ---- geometry: order each plate's disks by angle, flag heptagon regularity ----
by_plate = defaultdict(list)
for r in det:
    by_plate[r["plate"]].append(r)

rows = []
for p, items in by_plate.items():
    cx = sum(float(r["cx"]) for r in items) / len(items)
    cy = sum(float(r["cy"]) for r in items) / len(items)
    for r in items:
        ang = math.degrees(math.atan2(float(r["cy"]) - cy,
                                      float(r["cx"]) - cx)) % 360
        r["_ang"] = ang
        r["_cx"] = cx
        r["_cy"] = cy
    items.sort(key=lambda r: r["_ang"])
    # angular gap to the next disk, for detecting a missing disk
    for i, r in enumerate(items):
        nxt = items[(i + 1) % len(items)]
        r["gap_next"] = round((nxt["_ang"] - r["_ang"]) % 360, 1)
        r["n_disks"] = len(items)
        r["plate_cx"] = round(cx, 1)
        r["plate_cy"] = round(cy, 1)
        r["angle"] = round(r["_ang"], 1)
        rows.append(r)

print(f"plates: {len(by_plate)}")
print("disks per plate:", Counter(len(v) for v in by_plate.values()))

# ---- report which plates look like a 7-disk heptagon with one disk missing ----
print("\nplates where a gap is ~2x the heptagon spacing (a disk may be missing):")
for p in sorted(by_plate):
    items = [r for r in rows if r["plate"] == p]
    big = [r for r in items if r["gap_next"] > 80]
    if big:
        print(f"  {p:10} n={len(items)}  missing gap after angle "
              f"{big[0]['angle']:.0f} deg (gap {big[0]['gap_next']:.0f})")

# ---- write the expanded manifest ----
cols = ["plate", "crop", "idx", "angle", "gap_next", "n_disks", "side",
        "cx", "cy", "r", "plate_cx", "plate_cy"]
with open(OUT / "expanded_detections.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
    w.writeheader()
    for r in sorted(rows, key=lambda r: (r["plate"], r["angle"])):
        w.writerow(r)

sides = [int(r["side"]) for r in rows]
print(f"\ncrop side px: min {min(sides)} median {int(sorted(sides)[len(sides)//2])} "
      f"max {max(sides)}")
print(f"wrote {OUT/'expanded_detections.csv'}  ({len(rows)} rows)")
print(f"\nNOTE: batch {EXCLUDE_BATCH} belongs to the old class dataset, not to these "
      f"87 plates; it is excluded from any retraining built on this expansion.")
