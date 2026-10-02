"""Build the pill-classification manifest with a leakage-free grouped split.

Original dataset split (train/val) is a *within-pill augmentation split*: every
one of the 60 source pills contributes images to both train and val, so its
validation accuracy is optimistically biased.  Here we construct an additional
`split_group` assignment that keeps a whole pill (base key) on one side only.
"""
from __future__ import annotations

import collections
import csv
import hashlib
import re
from pathlib import Path

from PIL import Image

ROOT = Path(r"F:\Code\SCI\Nanjing_Children_Hospital\抑菌区域分析_返修\Antibacterial zone mask_Class\dataset")
OUT = Path(r"F:\Code\SCI\Nanjing_Children_Hospital\抑菌区域分析_返修\药片CNN")
CLASSES = ["CRO", "DA", "E", "LEV", "LZD", "P", "VA"]
EXTS = {".jpg", ".jpeg", ".png"}


def base_key(stem: str) -> str:
    s = stem.lower()
    s = re.sub(r"^aug[_\-]?\d+[_\-]?", "", s)
    s = re.sub(r"[_\-]?(enhanced)$", "", s)
    return s


def num_of(stem: str) -> int:
    m = re.match(r"^aug[_\-]?(\d+)", stem.lower())
    return int(m.group(1)) if m else -1


def md5_head(p: Path, cap: int = 1 << 18) -> str:
    h = hashlib.md5()
    with open(p, "rb") as f:
        h.update(f.read(cap))
    return h.hexdigest()


def main() -> None:
    rows = []
    for split in ["train", "val"]:
        for ci, c in enumerate(CLASSES):
            for p in sorted((ROOT / split / c).iterdir()):
                if p.suffix.lower() not in EXTS:
                    continue
                rows.append({
                    "path": str(p),
                    "class": c,
                    "class_idx": ci,
                    "orig_split": split,
                    "base_key": base_key(p.stem),
                    "aug_id": num_of(p.stem),
                    "md5": md5_head(p),
                })
    print(f"total images: {len(rows)}")

    # ---- honest grouped split at pill level -------------------------------
    # split each class's base keys into a validation fraction, deterministically
    key_counts = collections.Counter(r["base_key"] for r in rows)
    keys = sorted(key_counts)
    print(f"distinct pills: {len(keys)}  images/pill: {sorted(set(key_counts.values()))[:5]}...")

    # deterministic assignment: sort by hash so it is reproducible and unbiased
    keys_sorted = sorted(keys, key=lambda k: hashlib.md5(k.encode()).hexdigest())
    val_keys = set(keys_sorted[:15])          # 15 of 60 pills -> ~25%
    for r in rows:
        r["split_group"] = "val" if r["base_key"] in val_keys else "train"

    # sanity: no key straddles the grouped split
    straddle = {k for k in keys
                if len({r["split_group"] for r in rows if r["base_key"] == k}) > 1}
    assert not straddle, straddle

    g = collections.Counter((r["split_group"], r["class"]) for r in rows)
    print("\ngrouped split (pill-disjoint):")
    for c in CLASSES:
        print(f"  {c:4} train={g[('train', c)]:5}  val={g[('val', c)]:5}")
    print("  img-level val fraction "
          f"{sum(v for (s, _), v in g.items() if s == 'val') / len(rows):.3f}")
    nvt = len({r['base_key'] for r in rows if r['split_group'] == 'train'})
    nvv = len({r['base_key'] for r in rows if r['split_group'] == 'val'})
    print(f"  pills train={nvt} val={nvv}")

    # leakage audit of the original split
    tr_keys = {r["base_key"] for r in rows if r["orig_split"] == "train"}
    va_rows = [r for r in rows if r["orig_split"] == "val"]
    leaked = sum(1 for r in va_rows if r["base_key"] in tr_keys)
    print(f"\noriginal split leakage: {leaked}/{len(va_rows)} = {leaked/len(va_rows):.1%} "
          "of val images come from a pill also present in train")

    # what fraction of pixels are duplicated across splits (augmentation check)
    print("unique md5 per orig split:",
          {s: len({r['md5'] for r in rows if r['orig_split'] == s}) for s in ['train', 'val']})

    OUT.mkdir(parents=True, exist_ok=True)
    man = OUT / "manifest.csv"
    with open(man, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {man}")

    with open(OUT / "classes.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(CLASSES))
    with open(OUT / "val_keys.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(sorted(val_keys)))

    # image size distribution for choosing the training resolution
    sizes = collections.Counter()
    for r in rows[::7]:
        with Image.open(r["path"]) as im:
            sizes[im.size] += 1
    side = sorted(s[0] for s in sizes.elements())
    import statistics
    print(f"image side length: min={side[0]} p25={side[len(side)//4]} "
          f"median={statistics.median(side)} p75={side[3*len(side)//4]} max={side[-1]}")


if __name__ == "__main__":
    main()
