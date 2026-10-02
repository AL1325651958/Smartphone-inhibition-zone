"""Augment dataset_expanded with mild rotation / contrast / distortion.

Key rules
  * variants are generated WITHIN each split, so a plate never appears in two
    splits (the batch leakage that inflated earlier results);
  * the test set is left untouched - otherwise it stops being a held-out batch;
  * the manifest records every variant's source, batch and split so the origin
    can be reconstructed.

Output:
  dataset_augmented/
    images/<split>/<class>/<stem>.png          originals (copied)
    images/<split>/<class>/<stem>__augNN.png   variants
    by_class/<class>/*.png
    manifest.csv                               file, label, batch, split, source_file, aug_id
    data.yaml
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from augment_lib import mild_combo  # noqa: E402

CNN = Path(r"F:\Code\SCI\Nanjing_Children_Hospital\抑菌区域分析_返修\药片CNN")
RAW = Path(r"F:\Code\SCI\Nanjing_Children_Hospital\抑菌区域分析_返修\药片分类数据")
CNNSRC = CNN
SRC = CNN / "dataset_expanded"
OUT = CNN / "dataset_augmented"
CLASSES = ["CRO", "DA", "E", "LEV", "LZD", "P", "VA"]
SIZE = (112, 112)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-train", type=int, default=8, help="variants per train image")
    ap.add_argument("--n-val", type=int, default=2, help="variants per val image")
    ap.add_argument("--n-test", type=int, default=0, help="variants per test image")
    ap.add_argument("--strength", choices=["gentle", "mild"], default="mild")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--img-size", type=int, default=112,
                    help="output resolution; sources are resized to this")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    out = Path(args.out)
    SIZE = (args.img_size, args.img_size)
    rows = list(csv.DictReader(open(SRC / "manifest.csv", encoding="utf-8")))
    print(f"source images: {len(rows)}")

    plan = {"train": args.n_train, "val": args.n_val, "test": args.n_test}
    rng = random.Random(args.seed)

    if out.exists():
        shutil.rmtree(out)

    made = []
    for r in rows:
        split, cls = r["split"], r["label"]
        # prefer the highest-fidelity source available, then resize to SIZE once
        src_img = None
        for d in (CNNSRC / "new_batch_112" / "flat", RAW / "all",
                  RAW / "all300", SRC / "images" / split / cls):
            cand = d / r["file"]
            if cand.exists():
                src_img = cand
                break
        if src_img is None:
            print(f"  missing source for {r['file']}")
            continue
        with Image.open(src_img) as im:
            base = im.convert("L")
            if base.size != SIZE:
                base = base.resize(SIZE, Image.LANCZOS)
        # originals at the target resolution
        for sub in (Path("images") / split / cls, Path("by_class") / cls):
            d = out / sub
            d.mkdir(parents=True, exist_ok=True)
            base.save(d / r["file"], format="PNG", optimize=True)
        made.append(dict(file=r["file"], label=cls, batch=r["batch"],
                         split=split, source_file=r["file"], aug_id=-1,
                         params=""))

        n = plan.get(split, 0)
        if n <= 0:
            continue
        for k in range(n):
            aug, params = mild_combo(base, rng, strength=args.strength)
            if aug.size != SIZE:
                aug = aug.resize(SIZE, Image.LANCZOS)
            stem = Path(r["file"]).stem
            name = f"{stem}__aug{k:02d}.png"
            for sub in (Path("images") / split / cls, Path("by_class") / cls):
                d = out / sub
                d.mkdir(parents=True, exist_ok=True)
                aug.save(d / name, format="PNG", optimize=True)
            made.append(dict(file=name, label=cls, batch=r["batch"],
                             split=split, source_file=r["file"], aug_id=k,
                             params=json.dumps(
                                 {kk: round(vv, 4) if isinstance(vv, float) else vv
                                  for kk, vv in params.items()})))

    with open(out / "manifest.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["file", "label", "batch", "split",
                                          "source_file", "aug_id", "params"])
        w.writeheader()
        w.writerows(made)

    # ---- integrity: a batch must still live in exactly one split ----
    bs = defaultdict(set)
    for m in made:
        bs[m["batch"]].add(m["split"])
    straddle = sum(1 for v in bs.values() if len(v) > 1)

    # ---- integrity: every variant shares its source's split ----
    src_split = {r["file"]: r["split"] for r in rows}
    mism = sum(1 for m in made if src_split.get(m["source_file"]) != m["split"])

    (out / "data.yaml").write_text(json.dumps({
        "image_size": "112x112",
        "classes": CLASSES,
        "n_images": len(made),
        "n_original": len(rows),
        "variants_per_image": plan,
        "strength": args.strength,
        "augmentations": ["rotation <=15deg", "contrast <=12%",
                          "brightness <=10/255", "mild keystone/barrel"],
        "test_set": "not augmented (kept as held-out batches)",
    }, ensure_ascii=False, indent=1), encoding="utf-8")

    # ---- report ----
    print(f"\n{'split/class':22} {'orig':>5} {'aug':>6} {'total':>6}")
    tot = Counter()
    for split in ("train", "val", "test"):
        for c in CLASSES:
            o = sum(1 for m in made if m["split"] == split and m["label"] == c
                    and m["aug_id"] == -1)
            a = sum(1 for m in made if m["split"] == split and m["label"] == c
                    and m["aug_id"] >= 0)
            if o + a:
                print(f"{split + '/' + c:22} {o:>5} {a:>6} {o + a:>6}")
            tot[split] += o + a
    print(f"\n{'train':22} {'':>5} {'':>6} {tot['train']:>6}")
    print(f"{'val':22} {'':>5} {'':>6} {tot['val']:>6}")
    print(f"{'test':22} {'':>5} {'':>6} {tot['test']:>6}")
    print(f"{'TOTAL':22} {len(rows):>5} {len(made) - len(rows):>6} {len(made):>6}")

    print(f"\nbatches spanning >1 split: {straddle} (0 = clean)")
    print(f"variants whose split differs from their source: {mism} (0 = clean)")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
