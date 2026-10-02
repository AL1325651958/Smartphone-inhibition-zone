"""Read annotations directly from the Label Studio SQLite DB.

More reliable than a partial export: the DB holds every annotation currently in
the project.
"""
from __future__ import annotations

import json
import re
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path

DB = Path(r"F:\Code\SCI\Nanjing_Children_Hospital\抑菌区域分析_返修\药片CNN"
          r"\labelstudio\data\label_studio.sqlite3")
PROJECT = 2
CLASSES = ["CRO", "DA", "E", "LEV", "LZD", "P", "VA"]

con = sqlite3.connect(DB)
cur = con.cursor()

print("=== tables with 'annot' ===")
for (n,) in cur.execute("SELECT name FROM sqlite_master WHERE type='table' "
                        "AND name LIKE '%annot%'"):
    print("  ", n)

cur.execute("PRAGMA table_info(tasks_annotation)")
print("\ntasks_annotation columns:",
      [r[1] for r in cur.fetchall()])

# join annotations to tasks
q = """
SELECT t.id, t.data, a.result, a.was_cancelled
FROM task t
JOIN tasks_annotation a ON a.task_id = t.id
WHERE t.project_id = ?
"""
rows = cur.execute(q, (PROJECT,)).fetchall()
print(f"\nannotations found for project {PROJECT}: {len(rows)}")

tasks = {}
for tid, data, result, cancelled in rows:
    if cancelled:
        continue
    d = json.loads(data) if data else {}
    fname = Path(d.get("crop") or d.get("image") or "").name
    if not fname and d.get("image"):
        fname = Path(d["image"]).name
    label = None
    quality = None
    for res in (json.loads(result) if result else []):
        fn = res.get("from_name")
        vals = (res.get("value") or {}).get("choices") or []
        if not vals:
            continue
        if fn in ("choice", "drug"):
            label = vals[0]
        elif fn == "quality":
            quality = vals[0]
    # prefer the most recent annotation per task
    tasks[tid] = dict(task=tid, file=fname, label=label, quality=quality, data=d)

print(f"labelled tasks (unique): {len(tasks)}")
print("label distribution:", dict(Counter(t['label'] for t in tasks.values())))
print("quality distribution:", dict(Counter(t['quality'] for t in tasks.values())))

unlabelled = [t for t in tasks.values() if t["label"] not in CLASSES]
print(f"without a valid class: {len(unlabelled)}")
for t in unlabelled[:6]:
    print("   ", t["task"], t["file"], repr(t["label"]))

# what do the extra tasks (156 vs 125) look like?
cur.execute("SELECT id, data FROM task WHERE project_id=? ORDER BY id", (PROJECT,))
allt = cur.fetchall()
print(f"\ntotal tasks in project {PROJECT}: {len(allt)}")
print("first task id", allt[0][0], "last", allt[-1][0])
sample = json.loads(allt[-1][1])
print("last task data keys:", list(sample.keys()))
print("last task crop/image:", sample.get("crop"), "|", sample.get("image"))
con.close()
