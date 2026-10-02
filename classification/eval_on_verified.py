"""How accurate is the classifier on the 79 spreadsheet-verified new crops?

This is the gate for auto-labelling the remaining crops: if accuracy here is low,
auto-labels would just bake in errors.
"""
import csv
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from PIL import Image

HERE = Path(r"F:\Code\SCI\Nanjing_Children_Hospital\抑菌区域分析_返修\药片CNN")
CLASSES = ["CRO", "DA", "E", "LEV", "LZD", "P", "VA"]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import sys
sys.path.insert(0, str(HERE))
from train import PillCNN, MEAN, STD  # noqa: E402

lab = list(csv.DictReader(open(HERE / "expanded" / "labelled_crops.csv",
                                encoding="utf-8")))
print(f"verified crops: {len(lab)}")

for ck_name in ["orig", "plate_main"]:
    ck = torch.load(HERE / "runs" / ck_name / "best.pt", map_location=device,
                    weights_only=False)
    model = PillCNN().to(device)
    model.load_state_dict(ck["state"])
    model.eval()
    xs = []
    for r in lab:
        g = Image.open(HERE / "expanded" / "crops" / r["crop"]).convert("L") \
            .resize((160, 160), Image.BICUBIC)
        x = (torch.from_numpy(np.asarray(g, dtype=np.float32) / 255.0)[None] - MEAN) / STD
        xs.append(x.repeat(3, 1, 1))
    with torch.no_grad():
        P = torch.softmax(model(torch.stack(xs).to(device)).float(), 1).cpu().numpy()
    y = [CLASSES.index(r["label"]) for r in lab]
    pred = P.argmax(1)
    acc = (pred == np.array(y)).mean()
    print(f"\ncheckpoint {ck_name}: accuracy on verified crops {acc:.4f} "
          f"({(pred == np.array(y)).sum()}/{len(y)})")
    print(f"  mean confidence {P.max(1).mean():.3f}")
    c = Counter()
    for t, p in zip(y, pred):
        c[(CLASSES[t], CLASSES[p])] += 1
    print("  per-class recall:")
    for i, k in enumerate(CLASSES):
        n = sum(1 for t in y if t == i)
        if n:
            r = sum(1 for t, p in zip(y, pred) if t == i and p == i) / n
            print(f"    {k:4} n={n:3d} recall {r:.3f}")
    print("  confidence split:")
    for lo in (0.0, 0.7, 0.9, 0.99):
        m = P.max(1) >= lo
        if m.sum():
            print(f"    conf>={lo:.2f}: n={m.sum():3d} acc={(pred[m]==np.array(y)[m]).mean():.3f}")
