"""Organise the labelled disk crops into a training-ready dataset.

Input : a Label Studio export (project 2, files named
        raw_crops_<batch>__pill_XXX.png)
Output: <out>/images/<CLASS>/<name>.png     class-per-folder layout
        <out>/manifest.csv                  image, label, batch, split
        <out>/data.yaml                     summary for downstream training

Splitting is done at ACQUISITION BATCH level so no crop from one plate lands in
two splits (the batch leakage that inflated the earlier results).
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image

CLASSES = ["CRO", "DA", "E", "LEV", "LZD", "P", "VA"]
FULL = {"CRO": "Ceftriaxone", "DA": "Clindamycin", "E": "Erythromycin",
        "LEV": "Levofloxacin", "LZD": "Linezolid", "P": "Penicillin",
        "VA": "Vancomycin"}
RAW = Path(r"F:\Code\SCI\Nanjing_Children_Hospital\抑菌区域分析_返修\药片分类数据")


def load_export(path: Path):
    d = json.loads(path.read_text(encoding="utf-8"))
    rows = []
    for r in d:
        anns = [a for a in (r.get("annotations") or []) if not a.get("was_cancelled")]
        label = None
        for a in anns:
            for res in (a.get("result") or []):
                if res.get("from_name") in ("choice", "drug"):
                    vals = (res.get("value") or {}).get("choices") or []
                    if vals:
                        label = vals[0]
        fname = Path((r.get("file_upload") or
                      (r.get("data") or {}).get("image", ""))).name
        m = re.match(r"^[0-9a-f]{8}-(.+)$", fname)
        core = m.group(1) if m else fname
        rows.append(dict(task=r.get("id"), file=core, label=label))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--export", required=True)
    ap.add_argument("--size", choices=["orig", "300", "112"], default="112")
    ap.add_argument("--out", default=None)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--test-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    src_dir = {"orig": RAW / "all", "300": RAW / "all300",
               "112": RAW / "all112"}[args.size]
    out = Path(args.out) if args.out else (
        Path(r"F:\Code\SCI\Nanjing_Children_Hospital\抑菌区域分析_返修\药片CNN")
        / f"dataset_{args.size}")
    if out.exists():
        shutil.rmtree(out)
    (out / "images").mkdir(parents=True, exist_ok=True)

    rows = load_export(Path(args.export))
    print(f"export rows: {len(rows)}")
    labelled = [r for r in rows if r["label"] in CLASSES]
    dropped = [r for r in rows if r["label"] not in CLASSES]
    print(f"labelled: {len(labelled)}   unlabelled/skipped: {len(dropped)}")
    if dropped:
        print("  dropped tasks:", [r["task"] for r in dropped][:10])

    # batch = the acquisition session encoded in the filename
    def batch_of(f):
        m = re.match(r"^(raw_crops_\d{8}_\d{6})__", f)
        return m.group(1) if m else "unknown"

    for r in labelled:
        r["batch"] = batch_of(r["file"])
        r["src"] = src_dir / r["file"]

    missing = [r for r in labelled if not r["src"].exists()]
    if missing:
        raise SystemExit(f"{len(missing)} source files missing, e.g. {missing[0]['src']}")
    print(f"all {len(labelled)} source files found in {src_dir.name}")

    # ---- split by batch, stratified so each split sees every class where possible
    by_batch = defaultdict(list)
    for r in labelled:
        by_batch[r["batch"]].append(r)
    batches = sorted(by_batch)
    print(f"\nbatches: {len(batches)}")

    # greedy: assign whole batches to whichever split most needs their classes
    need = {"train": Counter(), "val": Counter(), "test": Counter()}
    total = Counter(r["label"] for r in labelled)
    target = {s: {c: total[c] * f for c in CLASSES}
              for s, f in (("train", 1 - args.val_frac - args.test_frac),
                           ("val", args.val_frac), ("test", args.test_frac))}
    rng = random.Random(args.seed)
    rng.shuffle(batches)
    assign = {}
    for b in batches:
        cnt = Counter(r["label"] for r in by_batch[b])
        best, best_score = None, None
        for s in ("val", "test", "train"):
            deficit = sum(max(0.0, target[s][c] - need[s][c]) for c in cnt)
            score = deficit / max(1, sum(cnt.values()))
            if best_score is None or score > best_score:
                best, best_score = s, score
        assign[b] = best
        need[best].update(cnt)

    for r in labelled:
        r["split"] = assign[r["batch"]]
        r["size"] = args.size

    # ---- materialise
    for r in labelled:
        d = out / "images" / r["split"] / r["label"]
        d.mkdir(parents=True, exist_ok=True)
        shutil.copy2(r["src"], d / r["file"])
    # flat class folders too (useful for ImageFolder)
    for r in labelled:
        d = out / "by_class" / r["label"]
        d.mkdir(parents=True, exist_ok=True)
        shutil.copy2(r["src"], d / r["file"])

    with open(out / "manifest.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["file", "label", "batch", "split",
                                          "task", "size"])
        w.writeheader()
        for r in sorted(labelled, key=lambda x: (x["split"], x["label"], x["file"])):
            w.writerow({k: r[k] for k in
                        ("file", "label", "batch", "split", "task", "size")})

    meta = {"image_size": args.size, "classes": CLASSES,
            "class_full_names": FULL, "n_images": len(labelled),
            "n_batches": len(by_batch),
            "split_by": "acquisition batch (no batch spans two splits)"}
    (out / "data.yaml").write_text(
        json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")

    # ---- report
    print(f"\n=== split x class ===")
    print(f"{'class':6} {'total':>6} {'train':>6} {'val':>6} {'test':>6}")
    for c in CLASSES:
        t = Counter(r["split"] for r in labelled if r["label"] == c)
        print(f"{c:6} {total[c]:>6} {t.get('train',0):>6} {t.get('val',0):>6} "
              f"{t.get('test',0):>6}")
    print(f"{'ALL':6} {len(labelled):>6} "
          f"{sum(1 for r in labelled if r['split']=='train'):>6} "
          f"{sum(1 for r in labelled if r['split']=='val'):>6} "
          f"{sum(1 for r in labelled if r['split']=='test'):>6}")

    batches_per_split = Counter(assign.values())
    print(f"\nbatches per split: {dict(batches_per_split)}")

    # verify no batch straddles splits
    bs = defaultdict(set)
    for r in labelled:
        bs[r["batch"]].add(r["split"])
    bad = {k: v for k, v in bs.items() if len(v) > 1}
    print(f"batches spanning >1 split: {len(bad)}  (0 = clean)")

    print(f"\nwrote {out}")
    print(f"  images/<split>/<class>/   {len(labelled)} files")
    print(f"  by_class/<class>/         {len(labelled)} files")
    print(f"  manifest.csv, data.yaml")


if __name__ == "__main__":
    main()
