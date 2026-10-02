"""Extract labels from the Label Studio export and locate the source images."""
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

SRC = Path(r"C:\Users\13256\.dsh\attachments\v1\files\3f\3f8f87c455d3c1f05ec15171b0fdfe26872c955abc8edb56c3167a40a94363c3\project-2-at-2026-09-20-12-27-05837aea.json")
RAW = Path(r"F:\Code\SCI\Nanjing_Children_Hospital\抑菌区域分析_返修\药片分类数据")
LS = Path(r"F:\Code\SCI\Nanjing_Children_Hospital\抑菌区域分析_返修\药片CNN\labelstudio")

d = json.loads(SRC.read_text(encoding="utf-8"))
print(f"records: {len(d)}  project: {d[0].get('project')}")

rows = []
for r in d:
    anns = r.get("annotations") or []
    label, quality = None, None
    for a in anns:
        if a.get("was_cancelled"):
            continue
        for res in (a.get("result") or []):
            fn = res.get("from_name")
            vals = (res.get("value") or {}).get("choices") or []
            if not vals:
                continue
            if fn in ("choice", "drug"):
                label = vals[0]
            elif fn == "quality":
                quality = vals[0]
    fname = r.get("file_upload") or (r.get("data") or {}).get("image", "")
    fname = Path(fname).name
    # strip the upload uuid prefix
    m = re.match(r"^[0-9a-f]{8}-(.+)$", fname)
    core = m.group(1) if m else fname
    rows.append(dict(task=r.get("id"), file=core, label=label, quality=quality,
                     lead_time=(anns[0].get("lead_time") if anns else None)))

print(f"annotated: {sum(1 for r in rows if r['label'])}/{len(rows)}")
print("\nlabel distribution:", dict(Counter(r["label"] for r in rows)))
print("quality distribution:", dict(Counter(r["quality"] for r in rows)))

print("\nfirst 12 rows:")
for r in rows[:12]:
    print(f"  {r['file']:52} {str(r['label']):6} {r['quality']}")

# where do these files live?
names = {r["file"] for r in rows}
print(f"\nsearching {len(names)} distinct filenames in the workspace...")
targets = {
    "药片分类数据/all": RAW / "all",
    "药片分类数据/all300": RAW / "all300",
    "药片分类数据/all112": RAW / "all112",
    "labelstudio/files": LS / "files",
}
for label, p in targets.items():
    if not p.exists():
        continue
    present = sum(1 for n in names if (p / n).exists())
    print(f"  {label:28} {present}/{len(names)} found")

# also check the grouped sets by basename
for label, p in [("extracted", RAW / "extracted"),
                 ("extracted300", RAW / "extracted300"),
                 ("extracted112", RAW / "extracted112")]:
    if not p.exists():
        continue
    have = {q.name for q in p.rglob("*.png")}
    # names there are pill_XXX.png only, so compare the pill part
    pills = {re.sub(r"^.*__", "", n) for n in names}
    print(f"  {label:28} basenames available: {len(have)}; "
          f"annotated basenames not in set: {len(pills - have)}")

# quick peek at one annotated file's dimensions where found
import PIL.Image as I
for label, p in targets.items():
    if not p.exists():
        continue
    for n in list(names)[:1]:
        f = p / n
        if f.exists():
            with I.open(f) as im:
                print(f"\nsample {label}/{n}: {im.size} {im.mode}")
            break
