"""Download all tasks+annotations for the disk project via the Label Studio API."""
from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from pathlib import Path

HERE = Path(r"F:\Code\SCI\Nanjing_Children_Hospital\抑菌区域分析_返修\药片CNN\labelstudio")
BASE = "http://127.0.0.1:8081"
PROJECT = 2
OUT = HERE / "annotations_full.json"

TOKEN = (HERE / "_token.txt").read_text(encoding="utf-8").strip()
HDR = {"Authorization": "Bearer " + TOKEN}


def api(path):
    req = urllib.request.Request(BASE + path, headers=HDR)
    with urllib.request.urlopen(req, timeout=90) as r:
        return json.loads(r.read())


tasks = []
page = 1
while True:
    d = api(f"/api/projects/{PROJECT}/tasks?page={page}&page_size=100")
    batch = d.get("tasks", []) if isinstance(d, dict) else d
    if not batch:
        break
    tasks.extend(batch)
    print(f"  page {page}: {len(batch)} tasks (total {len(tasks)})")
    if len(batch) < 100:
        break
    page += 1
    if page > 30:
        break

print(f"\ndownloaded {len(tasks)} tasks")
OUT.write_text(json.dumps(tasks, ensure_ascii=False), encoding="utf-8")
print(f"wrote {OUT}")

labelled = [t for t in tasks if t.get("annotations")]
print(f"tasks with annotations: {len(labelled)}")
if tasks:
    t0 = tasks[0]
    print("task keys:", [k for k in t0.keys()][:14])
    print("data keys:", list((t0.get("data") or {}).keys()))
