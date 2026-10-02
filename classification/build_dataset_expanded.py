"""Build the labelled dataset with a stratified, batch-disjoint 3-way split.

Two requirements pull against each other:
  * no acquisition batch may span two splits (otherwise the same plate leaks);
  * the sparse classes must appear in train, val and test if possible
    (E occurs in only 8 batches, DA in 11).
This uses StratifiedGroupKFold with the batch as the group and the *rarest class
of the batch* as its stratum, which spreads E and DA across all three splits.
"""
from __future__ import annotations

import csv
import json
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image
from sklearn.model_selection import StratifiedGroupKFold

CNN = Path(r"F:\Code\SCI\Nanjing_Children_Hospital\抑菌区域分析_返修\药片CNN")
LS = CNN / "labelstudio"
RAW = Path(r"F:\Code\SCI\Nanjing_Children_Hospital\抑菌区域分析_返修\药片分类数据")
OUT = CNN / "dataset_expanded"
SIZE = (112, 112)
CLASSES = ["CRO", "DA", "E", "LEV", "LZD", "P", "VA"]

SRC_112 = RAW / "all112"
SRC_300 = RAW / "all300"
SRC_ORIG = RAW / "all"
UPLOADS = LS / "data" / "media" / "upload" / "2"
NEW_112 = CNN / "new_batch_112" / "flat"

rows = json.loads((LS / "annotations_resolved.json").read_text(encoding="utf-8"))
labelled = [r for r in rows if r["label"] in CLASSES]


def core_name(u: str) -> str:
    n = Path(u).name
    m = re.match(r"^[0-9a-f]{8}-(.+)$", n)
    return m.group(1) if m else n


def find_source(core: str):
    for d, tag in ((SRC_112, "all112"), (NEW_112, "new_batch112"),
                   (SRC_300, "all300"), (SRC_ORIG, "all")):
        p = d / core
        if p.exists():
            return p, tag
    hits = list(UPLOADS.glob("*" + core)) if UPLOADS.exists() else []
    return (hits[0], "upload") if hits else (None, None)


items = []
for r in labelled:
    core = core_name(r["image"])
    src, tag = find_source(core)
    if src is None:
        continue
    m = re.match(r"^(raw_crops_\d{8}_\d{6})", core)
    items.append(dict(core=core, label=r["label"], task=r["task"],
                      src=src, source=tag, batch=m.group(1) if m else "unknown"))

print(f"labelled {len(labelled)}, resolved {len(items)}")

# ---- batch level table ----
by_batch = defaultdict(list)
for it in items:
    by_batch[it["batch"]].append(it)
batches = sorted(by_batch)
bcnt = {b: Counter(x["label"] for x in by_batch[b]) for b in batches}

# stratum = the rarest class present in the batch, so E/DA batches get spread
rarity = {c: sum(1 for b in batches if bcnt[b].get(c)) for c in CLASSES}
def stratum(b):
    present = [c for c in CLASSES if bcnt[b].get(c)]
    return min(present, key=lambda c: rarity[c])

X = batches
y = [stratum(b) for b in batches]
groups = batches                            # groups must be a flat array here
print(f"batches {len(batches)}; stratum counts: {dict(Counter(y))}")

sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=0)
folds = list(sgkf.split(X, y, groups))
# fold 3 -> test, fold 4 -> val, the rest -> train  (60/20/20 roughly)
assign = {}
for fi, (_, idx) in enumerate(folds):
    for i in idx:
        assign[X[i]] = {3: "test", 4: "val"}.get(fi, "train")

for it in items:
    it["split"] = assign[it["batch"]]

# ---- write ----
if OUT.exists():
    shutil.rmtree(OUT)
for it in items:
    img = Image.open(it["src"]).convert("RGB")
    if img.size != SIZE:
        img = img.resize(SIZE, Image.LANCZOS)
    for sub in (Path("images") / it["split"] / it["label"],
                Path("by_class") / it["label"]):
        d = OUT / sub
        d.mkdir(parents=True, exist_ok=True)
        img.save(d / it["core"], format="PNG", optimize=True)

with open(OUT / "manifest.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["file", "label", "batch", "split", "task", "source"])
    for it in sorted(items, key=lambda x: (x["split"], x["label"], x["core"])):
        w.writerow([it["core"], it["label"], it["batch"], it["split"],
                    it["task"], it["source"]])

(OUT / "data.yaml").write_text(json.dumps({
    "image_size": "112x112", "classes": CLASSES, "n_images": len(items),
    "n_batches": len(batches),
    "split": "batch-disjoint, stratified by rarest class (StratifiedGroupKFold)",
    "source": "Label Studio project 2, full API fetch",
}, ensure_ascii=False, indent=1), encoding="utf-8")

# ---- report ----
print(f"\n{'class':6} {'total':>6} {'train':>6} {'val':>6} {'test':>6}")
for c in CLASSES:
    t = Counter(x["split"] for x in items if x["label"] == c)
    print(f"{c:6} {sum(1 for x in items if x['label']==c):>6} "
          f"{t.get('train',0):>6} {t.get('val',0):>6} {t.get('test',0):>6}")
print(f"{'ALL':6} {len(items):>6} "
      f"{sum(1 for x in items if x['split']=='train'):>6} "
      f"{sum(1 for x in items if x['split']=='val'):>6} "
      f"{sum(1 for x in items if x['split']=='test'):>6}")

bs = defaultdict(set)
for it in items:
    bs[it["batch"]].add(it["split"])
print(f"\nbatches per split: {dict(Counter(assign.values()))}")
print(f"batches spanning >1 split: {sum(1 for v in bs.values() if len(v) > 1)} (0 = clean)")
zero = [f"{s}/{c}" for c in CLASSES for s in ("train", "val", "test")
        if not any(x["split"] == s and x["label"] == c for x in items)]
print(f"empty (split,class) cells: {zero if zero else 'none'}")
print(f"\nwrote {OUT}")
