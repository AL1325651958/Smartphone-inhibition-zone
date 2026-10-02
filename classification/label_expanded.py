"""Label the extracted crops and build the expanded training set.

Labels come from two sources combined:
  1. the inhibition-zone spreadsheet, which gives the exact drug set of each
     plate (blank = not tested, 6 = disk present with no zone);
  2. the existing classifier's per-crop prediction.
An optimal one-to-one assignment between the 7 detected disks and the plate's
known drug set (Hungarian) resolves conflicts, so a plate can never yield two
disks of the same drug when that drug was tested once.  Assignments whose
prediction disagrees with the plate set are dropped rather than guessed.
"""
from __future__ import annotations

import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import openpyxl
import torch
from PIL import Image

HERE = Path(r"F:\Code\SCI\Nanjing_Children_Hospital\抑菌区域分析_返修\药片CNN")
sys.path.insert(0, str(HERE))
from train import CLASSES, PillCNN, MEAN, STD  # noqa: E402

XL = Path(r"F:\Code\SCI\Nanjing_Children_Hospital\抑菌区域分析_返修\链球菌检测\Sheet1.xlsx")
OUT = HERE / "expanded"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device {device}")


# ---------------------------------------------------------------- label map
def plate_drugs():
    wb = openpyxl.load_workbook(XL, read_only=True, data_only=True)
    grid = [list(r) for r in wb.worksheets[0].iter_rows(values_only=True)]
    header = grid[2]
    col_plate, cur = {}, None
    for c, v in enumerate(header):
        if v is not None and str(v).strip():
            cur = str(v).strip()
        if c >= 2:
            col_plate[c] = cur
    drugs = {}
    for r in grid[3:]:
        if not r or r[0] is None or not str(r[0]).strip():
            continue
        name = str(r[0]).strip()
        for c, p in col_plate.items():
            if p is None or c >= len(r):
                continue
            val = r[c]
            if val is not None and str(val).strip() != "":
                drugs.setdefault(p, set()).add(name)
    return {p: sorted(v) for p, v in drugs.items()}


plate_sets = plate_drugs()
print(f"plates with a known drug set in the spreadsheet: {sorted(plate_sets)}")
for p, d in sorted(plate_sets.items()):
    print(f"  {p}: {d}")

# ---------------------------------------------------------------- classify
rows = list(csv.DictReader(open(OUT / "detections.csv", encoding="utf-8")))
print(f"\ncrops to label: {len(rows)}")

ck = torch.load(HERE / "runs" / "orig" / "best.pt", map_location=device,
                weights_only=False)
model = PillCNN().to(device)
model.load_state_dict(ck["state"])
model.eval()


def probs_for(paths):
    xs = []
    for p in paths:
        g = Image.open(p).convert("L").resize((160, 160), Image.BICUBIC)
        x = (torch.from_numpy(np.asarray(g, dtype=np.float32) / 255.0)[None] - MEAN) / STD
        xs.append(x.repeat(3, 1, 1))
    with torch.no_grad():
        return torch.softmax(model(torch.stack(xs).to(device)).float(), 1).cpu().numpy()


by_photo = defaultdict(list)
for r in rows:
    by_photo[r["photo"]].append(r)

labelled, unlabelled, conflicts = [], [], 0
for photo, items in by_photo.items():
    paths = [OUT / "crops" / it["crop"] for it in items]
    P = probs_for(paths)
    plate = items[0]["plate"]                      # e.g. '001'
    plate_id = plate.split("_")[0]                 # '025_1' -> '025'
    # spreadsheet keys are like '002' -> match the numeric stem
    key = None
    for k in plate_sets:
        if plate_id.lstrip("0") == k.lstrip("0"):
            key = k
            break
    for it, pr in zip(items, P):
        it["pred"] = CLASSES[int(pr.argmax())]
        it["conf"] = float(pr.max())
        it["probs"] = pr

    if key is None:
        for it in items:
            unlabelled.append(it)
        continue

    allowed = plate_sets[key]
    # optimal assignment of detected disks to allowed drugs
    n, m = len(items), len(allowed)
    cost = np.full((n, m), -1e9)
    for i, it in enumerate(items):
        for j, d in enumerate(allowed):
            cost[i, j] = float(it["probs"][CLASSES.index(d)])
    try:
        from scipy.optimize import linear_sum_assignment
        ri, cj = linear_sum_assignment(-cost)
        pairs = list(zip(ri, cj))
    except Exception:
        pairs = []
        used = set()
        for i, j in sorted(((i, j) for i in range(n) for j in range(m)),
                           key=lambda t: -cost[t[0], t[1]]):
            if i in used or j in used:
                continue
            used.add(i)
            pairs.append((i, j))
    for i, j in pairs:
        it = items[i]
        d = allowed[j]
        if it["pred"] != d:
            conflicts += 1
        it["label"] = d
        it["agrees"] = (it["pred"] == d)
        labelled.append(it)

print(f"\nlabelled {len(labelled)} crops, unlabelled {len(unlabelled)} "
      f"(plates absent from the spreadsheet)")
print(f"assignment changed the prediction for {conflicts} crops "
      f"({conflicts/max(len(labelled),1):.1%})")
print(f"model agreed with the assignment for "
      f"{sum(1 for it in labelled if it['agrees'])}/{len(labelled)}")

print("\nlabel distribution (expanded set):")
c = Counter(it["label"] for it in labelled)
for k in CLASSES:
    print(f"  {k:4} {c.get(k,0):4d}")

print("\nindependent disks per class (1 disk per photo):")
seen = defaultdict(set)
for it in labelled:
    seen[it["label"]].add(it["photo"])
for k in CLASSES:
    print(f"  {k:4} {len(seen[k]):4d} disks from "
          f"{len({it['plate'] for it in labelled if it['label']==k})} plates")

with open(OUT / "labelled_crops.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["crop", "photo", "plate", "idx", "label", "model_pred",
                "model_conf", "agrees", "cx", "cy", "r", "side"])
    for it in labelled:
        w.writerow([it["crop"], it["photo"], it["plate"], it["idx"], it["label"],
                    it["pred"], f"{it['conf']:.4f}", int(it["agrees"]),
                    it["cx"], it["cy"], it["r"], it["side"]])
print(f"\nwrote {OUT/'labelled_crops.csv'}")
