"""Create a batch-disjoint (plate-held-out) train/val/test split.

The shipped split cannot be used as a test set: all 60 source disks, and all 13
acquisition plates, appear on both sides, so every "validation" image is an
augmented twin of a *training* disk.

This script builds a split that respects the acquisition batch. Whole plates are
confined to one side, so no disk and no augmentation twin crosses a boundary.

Disks and augmented copies live together inside a split; for grouping-based
evaluation use `--group-field core` (leave-disks-out) or `--group-field plate`
(leave-batches-out), which ignore the physical split and only use it to define
which images are eligible.

Usage
-----
python make_split.py                     # report the available designs
python make_split.py --test-plate 20260505_062940 --val-plate 20260505_063155
python make_split.py --emit --test-plate ... --val-plate ...
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPORTS = HERE / "reports"
CLASSES = ["CRO", "DA", "E", "LEV", "LZD", "P", "VA"]
MANIFEST = HERE / "manifest_plate.csv"


def load():
    with open(MANIFEST, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["class_idx"] = int(r["class_idx"])
        r["clean"] = int(r["clean"])
    return rows


def describe(rows):
    plates = defaultdict(list)
    for r in rows:
        plates[r["plate"]].append(r)
    print(f"{len(rows)} images, {len(plates)} plates, "
          f"{len({r['core'] for r in rows})} disks\n")
    print(f"{'plate':22} {'imgs':>5} {'disks':>5}  classes covered")
    for p in sorted(plates):
        rs = plates[p]
        cov = sorted({r["class"] for r in rs})
        print(f"{p:22} {len(rs):>5} {len({r['core'] for r in rs}):>5}  "
              f"{len(cov)}/7 {cov}")


def propose(rows):
    """Rank plates by how well they can serve as a held-out test set."""
    plates = defaultdict(list)
    for r in rows:
        plates[r["plate"]].append(r)
    print("\nplates that cover all 7 classes (usable as a standalone test set):")
    cands = []
    for p, rs in plates.items():
        cov = {r["class"] for r in rs}
        if len(cov) == 7:
            cnt = Counter(r["class"] for r in rs)
            cands.append((p, len(rs), len({r["core"] for r in rs}),
                          min(cnt.get(c, 0) for c in CLASSES)))
    for p, n, nd, mn in sorted(cands, key=lambda t: t[1]):
        rest = [r for r in rows if r["plate"] != p]
        ctr = Counter(r["class"] for r in rest)
        print(f"  {p:22} test {n:5} imgs ({nd:2} disks)  "
              f"min disks/class in test {mn}  | remaining train "
              f"{len(rest):5} imgs, min class {min(ctr.get(c, 0) for c in CLASSES)}")


def build(rows, test_plates, val_plates):
    """Assign every image to train/val/test by plate; assert zero overlap."""
    if set(test_plates) & set(val_plates):
        raise SystemExit("test and val plates must be disjoint")
    def side(p):
        if p in test_plates:
            return "test"
        if p in val_plates:
            return "val"
        return "train"
    for r in rows:
        r["split_plate"] = side(r["plate"])

    # hard guarantees
    for field in ("plate", "core"):
        groups = defaultdict(set)
        for r in rows:
            groups[r[field]].add(r["split_plate"])
        bad = {k: v for k, v in groups.items() if len(v) > 1}
        if bad:
            raise SystemExit(f"{field} straddles splits: {list(bad)[:5]}")
    print("\nsplit built with no plate or disk straddling a boundary")
    for s in ("train", "val", "test"):
        rs = [r for r in rows if r["split_plate"] == s]
        cnt = Counter(r["class"] for r in rs)
        print(f"  {s:5} {len(rs):5} imgs  {len({r['plate'] for r in rs}):2} plates  "
              f"{len({r['core'] for r in rs}):2} disks  "
              f"per-class {[cnt.get(c, 0) for c in CLASSES]}")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-plate", nargs="*", default=[])
    ap.add_argument("--val-plate", nargs="*", default=[])
    ap.add_argument("--emit", action="store_true")
    args = ap.parse_args()

    rows = load()
    describe(rows)
    propose(rows)
    if not args.test_plate:
        print("\npass --test-plate/--val-plate to build and --emit to write the csv")
        return
    rows = build(rows, set(args.test_plate), set(args.val_plate))
    if args.emit:
        out = HERE / "manifest_plate_split.csv"
        with open(out, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        meta = dict(test_plates=sorted(args.test_plate),
                    val_plates=sorted(args.val_plate),
                    n_images=len(rows))
        (REPORTS / "plate_split.json").write_text(json.dumps(meta, indent=1),
                                                 encoding="utf-8")
        print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
