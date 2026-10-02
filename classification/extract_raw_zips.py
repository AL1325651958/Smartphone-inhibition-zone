"""Extract the 药片分类数据 archives, grouped by archive (prefix) name.

Layout produced:
  <root>/extracted/<archive-prefix>/            files as stored in the archive
  <root>/extracted/<archive-prefix>/_meta.txt   source archive + file count
  <root>/all/                                   flat copy, names prefixed with
                                                the archive to avoid collisions

The archives all use the same internal names (pill_001.png ...), so a flat
directory needs the archive name as a prefix or files would overwrite each other.
"""
from __future__ import annotations

import csv
import re
import zipfile
from collections import Counter
from pathlib import Path

RAW = Path(r"F:\Code\SCI\Nanjing_Children_Hospital\抑菌区域分析_返修\药片分类数据")
OUT = RAW / "extracted"
FLAT = RAW / "all"
EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

zips = sorted(RAW.glob("*.zip"))
print(f"archives found: {len(zips)}")

if not zips:
    raise SystemExit("no .zip files found")

# ---------------------------------------------------------------- plan
plan = []
for z in zips:
    # prefix = archive name without the trailing timestamp, e.g. raw_crops
    stem = z.stem
    m = re.match(r"^(.*?)_(\d{8}_\d{6})$", stem)
    prefix = m.group(1) if m else stem
    stamp = m.group(2) if m else stem
    with zipfile.ZipFile(z) as zf:
        names = [n for n in zf.namelist()
                 if Path(n).suffix.lower() in EXTS and not n.endswith("/")]
    plan.append(dict(zip=z, prefix=prefix, stamp=stamp, names=names))

print(f"\n{'archive':34} {'prefix':12} {'files':>6}")
for p in plan:
    print(f"{p['zip'].name:34} {p['prefix']:12} {len(p['names']):>6}")
print(f"\ntotal files: {sum(len(p['names']) for p in plan)}")
print("distinct prefixes:", sorted({p['prefix'] for p in plan}))

# name collisions inside one archive?
for p in plan:
    c = Counter(Path(n).name for n in p["names"])
    dup = {k: v for k, v in c.items() if v > 1}
    if dup:
        print(f"  NOTE {p['zip'].name} has duplicate names: {dup}")

# cross-archive name overlap (this is why a flat dir needs prefixes)
allnames = Counter(Path(n).name for p in plan for n in p["names"])
shared = sum(1 for v in allnames.values() if v > 1)
print(f"\nfile names appearing in more than one archive: {shared} of {len(allnames)}")

# ---------------------------------------------------------------- extract
OUT.mkdir(parents=True, exist_ok=True)
FLAT.mkdir(parents=True, exist_ok=True)

manifest = []
for p in plan:
    target = OUT / f"{p['prefix']}_{p['stamp']}"
    target.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(p["zip"]) as zf:
        zf.extractall(target)
    for n in p["names"]:
        src = target / n
        flat_name = f"{p['prefix']}_{p['stamp']}__{Path(n).name}"
        dst = FLAT / flat_name
        dst.write_bytes(src.read_bytes())
        manifest.append(dict(archive=p["zip"].name, prefix=p["prefix"],
                             stamp=p["stamp"], member=n,
                             extracted=str(src.relative_to(RAW)),
                             flat=str(dst.relative_to(RAW)),
                             bytes=dst.stat().st_size))
    (target / "_meta.txt").write_text(
        f"source archive: {p['zip'].name}\nfiles: {len(p['names'])}\n",
        encoding="utf-8")
    print(f"  extracted {p['zip'].name} -> {target.name} ({len(p['names'])} files)")

with open(RAW / "extract_manifest.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(manifest[0].keys()))
    w.writeheader()
    w.writerows(manifest)

print(f"\nwrote {RAW/'extract_manifest.csv'}  ({len(manifest)} rows)")
print(f"grouped by archive -> {OUT}")
print(f"flat copy          -> {FLAT}  ({len(list(FLAT.glob('*')))} files)")
