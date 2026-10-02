"""Apply the supplied enhancement pipeline to every image set.

Pipeline (unchanged from imagepross.py):
  grayscale -> CLAHE(clipLimit=3.0, tile=8x8) -> Laplacian sharpen -> gamma 1.5

Sets processed (each keeps its own folder structure):
  <CNN>/enhanced/raw_crops_<batch>/pill_XXX_enhanced.jpg
  <CNN>/enhanced/new_batch/<batch>/pill_XXX_enhanced.jpg
  <CNN>/enhanced/labelstudio/<plate>/<crop>_enhanced.jpg
  <CNN>/dataset_expanded/images/<split>/<class>/<file>_enhanced.jpg
  <CNN>/dataset_expanded/by_class/<class>/<file>_enhanced.jpg

Only the enhanced copies are written; originals are never modified.
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from enhance_lib import SUPPORTED_EXTS, enhance_pill_image, imwrite_unicode  # noqa: E402
import cv2  # noqa: E402
import numpy as np  # noqa: E402

CNN = Path(r"F:\Code\SCI\Nanjing_Children_Hospital\抑菌区域分析_返修\药片CNN")
RAW = Path(r"F:\Code\SCI\Nanjing_Children_Hospital\抑菌区域分析_返修\药片分类数据")
OUT = CNN / "enhanced"

SOURCES = {
    "raw_crops": RAW / "extracted",
    "new_batch": CNN / "new_batch_112",
    "labelstudio": CNN / "labelstudio" / "files",
    "dataset_split": CNN / "dataset_expanded" / "images",
    "dataset_class": CNN / "dataset_expanded" / "by_class",
}


def process(name: str, src: Path, out_root: Path, rows: list,
            comparison: bool, overwrite: bool) -> tuple[int, int]:
    files = [p for p in sorted(src.rglob("*"))
             if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS
             and not p.stem.endswith("_enhanced")]
    ok = fail = 0
    for i, p in enumerate(files, 1):
        rel = p.relative_to(src)
        dest_dir = out_root / rel.parent
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / (p.stem + "_enhanced.jpg")
        try:
            if dest.exists() and not overwrite:
                ok += 1
                continue
            enhanced = enhance_pill_image(p)
            imwrite_unicode(dest, enhanced, quality=95)
            rows.append(dict(set=name, source=str(p), output=str(dest),
                             w=enhanced.shape[1], h=enhanced.shape[0],
                             bytes=dest.stat().st_size))
            if comparison:
                from PIL import Image as PILImage
                with PILImage.open(p) as im:
                    og = np.asarray(im.convert("L"))
                ob = cv2.cvtColor(og, cv2.COLOR_GRAY2BGR)
                eb = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
                h2, w2 = og.shape
                pad = 28

                def labeled(im_bgr, text):
                    cv = np.zeros((h2 + pad, w2, 3), dtype=np.uint8)
                    cv[:pad] = (40, 40, 40)
                    cv[pad:] = im_bgr
                    cv2.putText(cv, text, (4, 20), cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, (200, 200, 200), 1)
                    return cv

                comp = np.hstack([labeled(ob, "Original"), labeled(eb, "Enhanced")])
                imwrite_unicode(dest_dir / (p.stem + "_comparison.jpg"), comp, 92)
            ok += 1
            if i % 100 == 0:
                print(f"    {name}: {i}/{len(files)}")
        except Exception as exc:
            fail += 1
            print(f"    [FAIL] {p.name}: {exc}")
    return ok, fail


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--comparison", action="store_true",
                    help="also write original|enhanced side-by-side sheets")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--only", nargs="*", default=None,
                    help="subset of set names to process")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    rows: list = []
    totals = Counter()
    for name, src in SOURCES.items():
        if args.only and name not in args.only:
            continue
        if not src.exists():
            print(f"skip {name}: {src} missing")
            continue
        n_src = sum(1 for p in src.rglob("*")
                    if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS
                    and not p.stem.endswith("_enhanced"))
        print(f"[{name}] {n_src} images from {src}")
        ok, fail = process(name, src, OUT / name, rows, args.comparison,
                           args.overwrite)
        totals[name] = (ok, fail)
        print(f"  -> {ok} ok, {fail} failed")

    with open(OUT / "enhance_manifest.csv", "w", newline="", encoding="utf-8") as f:
        if rows:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    print("\n=== summary ===")
    for name, (ok, fail) in totals.items():
        print(f"  {name:16} {ok:5} ok  {fail:3} failed")
    total = sum(ok for ok, _ in totals.values())
    print(f"  {'TOTAL':16} {total:5} images enhanced")
    print(f"\noutput -> {OUT}")

    # verify every output is a readable grayscale JPEG of the expected size
    outs = [p for p in OUT.rglob("*_enhanced.jpg")]
    bad = 0
    for p in outs[:200]:
        img = cv2.imdecode(np.fromfile(str(p), dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        if img is None or img.ndim != 2:
            bad += 1
    print(f"verified {min(200, len(outs))} outputs, problems: {bad}")
    print(f"manifest -> {OUT/'enhance_manifest.csv'}")


if __name__ == "__main__":
    main()
