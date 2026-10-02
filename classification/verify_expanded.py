"""Verify the expanded dataset folder."""
import csv
from collections import Counter
from pathlib import Path

from PIL import Image

OUT = Path(r"F:\Code\SCI\Nanjing_Children_Hospital\抑菌区域分析_返修\药片CNN"
           r"\dataset_expanded")
CLASSES = ["CRO", "DA", "E", "LEV", "LZD", "P", "VA"]

print("top level:", sorted(p.name for p in OUT.iterdir()))

rows = list(csv.DictReader(open(OUT / "manifest.csv", encoding="utf-8")))
print(f"manifest rows: {len(rows)}")

# images tree
print(f"\n{'split':7} {'files':>6}  per class")
tot = 0
for s in ("train", "val", "test"):
    d = OUT / "images" / s
    if not d.exists():
        continue
    per = {}
    n = 0
    for c in CLASSES:
        k = len(list((d / c).glob("*.png"))) if (d / c).exists() else 0
        per[c] = k
        n += k
    tot += n
    print(f"{s:7} {n:>6}  {per}")
print(f"{'total':7} {tot:>6}")

# by_class
d = OUT / "by_class"
if d.exists():
    per = {c: len(list((d / c).glob("*.png"))) for c in CLASSES}
    print(f"\nby_class: {per}  total {sum(per.values())}")

# every file must be 112x112 RGB
bad = []
for p in (OUT / "images").rglob("*.png"):
    with Image.open(p) as im:
        if im.size != (112, 112) or im.mode != "RGB":
            bad.append((p.name, im.size, im.mode))
print(f"\nsize/mode violations: {len(bad)}")

# batch leakage re-check from the manifest
bs = {}
for r in rows:
    bs.setdefault(r["batch"], set()).add(r["split"])
print(f"batches: {len(bs)}  spanning >1 split: "
      f"{sum(1 for v in bs.values() if len(v) > 1)}")

# duplicates?
names = [r["file"] for r in rows]
print(f"duplicate filenames: {len(names) - len(set(names))}")

print(f"\ndata.yaml:\n{(OUT/'data.yaml').read_text(encoding='utf-8')}")
