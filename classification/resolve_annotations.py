"""Inspect task image paths and pull annotation results for every task."""
from __future__ import annotations

import json
import re
import urllib.request
from collections import Counter
from pathlib import Path

HERE = Path(r"F:\Code\SCI\Nanjing_Children_Hospital\抑菌区域分析_返修\药片CNN\labelstudio")
BASE = "http://127.0.0.1:8081"
TOKEN = (HERE / "_token.txt").read_text(encoding="utf-8").strip()
HDR = {"Authorization": "Bearer " + TOKEN}
CLASSES = ["CRO", "DA", "E", "LEV", "LZD", "P", "VA"]

tasks = json.loads((HERE / "annotations_full.json").read_text(encoding="utf-8"))
print(f"tasks: {len(tasks)}")

print("\nimage path patterns (first 3 and last 3):")
for t in tasks[:3] + tasks[-3:]:
    print(f"  id={t['id']:4}  {t['data'].get('image')}")

pat = Counter()
for t in tasks:
    u = t["data"].get("image", "")
    if "/data/upload/" in u:
        pat["upload"] += 1
    elif ":8082/" in u:
        pat["static-8082"] += 1
    elif "/data/local-files/" in u:
        pat["local-files"] += 1
    else:
        pat["other"] += 1
print("\nimage path kinds:", dict(pat))

print("\nfirst upload-style image values:")
n = 0
for t in tasks:
    u = t["data"].get("image", "")
    if "/data/upload/" in u:
        print("   ", t["id"], u)
        n += 1
        if n >= 5:
            break

# ---- fetch annotations per task ----
def api(path):
    req = urllib.request.Request(BASE + path, headers=HDR)
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read())

out = []
for i, t in enumerate(tasks, 1):
    try:
        d = api(f"/api/tasks/{t['id']}/")
    except Exception as exc:
        print(f"  task {t['id']} failed: {exc}")
        continue
    anns = [a for a in (d.get("annotations") or []) if not a.get("was_cancelled")]
    label = None
    quality = None
    for a in anns:
        for res in (a.get("result") or []):
            fn = res.get("from_name")
            vals = (res.get("value") or {}).get("choices") or []
            if not vals:
                continue
            if fn in ("choice", "drug"):
                label = vals[0]
            elif fn == "quality":
                quality = vals[0]
    u = (d.get("data") or {}).get("image", "")
    out.append(dict(task=t["id"], image=u, label=label, quality=quality,
                    n_annotations=len(anns)))
    if i % 40 == 0:
        print(f"  fetched {i}/{len(tasks)}")

(HERE / "annotations_resolved.json").write_text(
    json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")

lab = [o for o in out if o["label"] in CLASSES]
print(f"\nfetched {len(out)} tasks; labelled {len(lab)}")
print("labels:", dict(Counter(o["label"] for o in out)))
print("quality:", dict(Counter(o["quality"] for o in out)))
print(f"wrote {HERE/'annotations_resolved.json'}")
