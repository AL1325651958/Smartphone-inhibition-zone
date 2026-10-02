"""Verify the enhanced output tree."""
import csv
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

CNN = Path(r"F:\Code\SCI\Nanjing_Children_Hospital\抑菌区域分析_返修\药片CNN")
OUT = CNN / "enhanced"

print("top level:", sorted(p.name for p in OUT.iterdir() if p.is_dir()))

rows = list(csv.DictReader(open(OUT / "enhance_manifest.csv", encoding="utf-8")))
print(f"manifest rows: {len(rows)}")

by_set = Counter(r["set"] for r in rows)
print(f"\n{'set':16} {'images':>7}   size distribution")
for s in sorted(by_set):
    ss = [r for r in rows if r["set"] == s]
    sizes = Counter(f"{r['w']}x{r['h']}" for r in ss)
    top = ", ".join(f"{k}({v})" for k, v in sizes.most_common(4))
    print(f"{s:16} {by_set[s]:>7}   {top}")

# structural check: does the tree mirror the sources?
print("\ntree shape per set (dirs / files):")
for s in sorted(by_set):
    d = OUT / s
    nd = sum(1 for p in d.rglob("*") if p.is_dir())
    nf = sum(1 for p in d.rglob("*.jpg"))
    print(f"  {s:16} {nd:4} dirs  {nf:5} jpg")

# decode check across all outputs
bad, checked = [], 0
for p in OUT.rglob("*.jpg"):
    img = cv2.imdecode(np.fromfile(str(p), dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    checked += 1
    if img is None or img.ndim != 2:
        bad.append(p.name)
print(f"\ndecoded {checked} jpgs; grayscale-ok {checked - len(bad)}; problems {len(bad)}")
if bad:
    print("  examples:", bad[:5])

# sample a few enhanced files for the record
print("\nsample outputs:")
for s in sorted(by_set):
    fs = sorted((OUT / s).rglob("*_enhanced.jpg"))[:2]
    for f in fs:
        print(f"  {f.relative_to(OUT)}")
