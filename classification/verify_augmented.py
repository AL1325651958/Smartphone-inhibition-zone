"""Verify the augmented dataset: counts, sizes, and split integrity."""
import csv
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image

CNN = Path(r"F:\Code\SCI\Nanjing_Children_Hospital\抑菌区域分析_返修\药片CNN")
OUT = CNN / "dataset_augmented"
CLASSES = ["CRO", "DA", "E", "LEV", "LZD", "P", "VA"]

rows = list(csv.DictReader(open(OUT / "manifest.csv", encoding="utf-8")))
print(f"manifest rows: {len(rows)}")

print(f"\n{'split/class':14} {'orig':>5} {'aug':>5} {'total':>6}")
for split in ("train", "val", "test"):
    st = 0
    for c in CLASSES:
        o = sum(1 for r in rows if r["split"] == split and r["label"] == c
                and r["aug_id"] == "-1")
        a = sum(1 for r in rows if r["split"] == split and r["label"] == c
                and r["aug_id"] != "-1")
        if o + a:
            print(f"{split + '/' + c:14} {o:>5} {a:>5} {o + a:>6}")
            st += o + a
    print(f"{'  subtotal':14} {'':>5} {'':>5} {st:>6}")

# integrity
bs = defaultdict(set)
for r in rows:
    bs[r["batch"]].add(r["split"])
print(f"\nbatches: {len(bs)}   spanning >1 split: "
      f"{sum(1 for v in bs.values() if len(v) > 1)}")

src_split = {}
for r in rows:
    if r["aug_id"] == "-1":
        src_split[r["file"]] = r["split"]
bad = [r for r in rows if r["aug_id"] != "-1"
       and src_split.get(r["source_file"]) != r["split"]]
print(f"variants in a different split than their source: {len(bad)}")

# no variant named twice
print(f"duplicate filenames within a split: "
      f"{len(rows) - len({(r['split'], r['file']) for r in rows})}")

# every image decodes at 112x112
bad_img, checked = [], 0
for p in (OUT / "images").rglob("*.png"):
    checked += 1
    with Image.open(p) as im:
        if im.size != (112, 112):
            bad_img.append((p.name, im.size))
print(f"\nchecked {checked} images; wrong size: {len(bad_img)}")

by_class = OUT / "by_class"
if by_class.exists():
    per = {c: len(list((by_class / c).glob("*.png"))) for c in CLASSES}
    print(f"by_class totals: {per}  sum {sum(per.values())}")

print(f"\ndata.yaml:\n{(OUT/'data.yaml').read_text(encoding='utf-8')}")
