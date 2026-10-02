"""Align plates by the angular gaps between consecutive disks.

Because the disks sit on a regular heptagon, the sequence of drugs around a plate
is defined only up to rotation.  If all plates share the same physical layout
(disks put on in the same order, then photographed at different rotations), then
their gap-sequences are rotations of one another, and identifying one plate fixes
all of them.
"""
import csv
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

HERE = Path(r"F:\Code\SCI\Nanjing_Children_Hospital\抑菌区域分析_返修\药片CNN")
CLASSES = ["CRO", "DA", "E", "LEV", "LZD", "P", "VA"]
det = {r["crop"]: r for r in csv.DictReader(
    open(HERE / "expanded" / "detections.csv", encoding="utf-8"))}
lab = list(csv.DictReader(open(HERE / "expanded" / "labelled_crops.csv",
                                encoding="utf-8")))

seq = defaultdict(list)
for r in lab:
    d = det[r["crop"]]
    seq[r["plate"]].append((float(d["cx"]), float(d["cy"]), r["label"]))

ordered = {}
for p, items in seq.items():
    cx = np.mean([x for x, _, _ in items])
    cy = np.mean([y for _, y, _ in items])
    pts = sorted(((math.degrees(math.atan2(y - cy, x - cx)) % 360, l)
                  for x, y, l in items))
    ordered[p] = [l for _, l in pts]

def canonical(s):
    """Lexicographically smallest rotation, for grouping."""
    return min(tuple(s[i:] + s[:i]) for i in range(len(s)))

print(f"{'plate':8} {'n':>3}  sequence (by angle)                canonical")
groups = Counter()
for p in sorted(ordered):
    s = ordered[p]
    c = canonical(s)
    groups[c] += 1
    print(f"{p:8} {len(s):>3}  {','.join(s):34} {'|'.join(c)}")

print(f"\ndistinct canonical layouts: {len(groups)}")
for c, n in groups.most_common():
    print(f"  {n}x  {'|'.join(c)}")

# rotational offset of each plate relative to the most common layout
if groups:
    base = groups.most_common(1)[0][0]
    print(f"\nreference layout: {'|'.join(base)}")
    print(f"{'plate':8} {'rotation needed to align':>25}")
    for p in sorted(ordered):
        s = ordered[p]
        off = None
        for k in range(len(s)):
            if tuple(s[k:] + s[:k]) == base:
                off = k
                break
        print(f"{p:8} {str(off):>25}")
