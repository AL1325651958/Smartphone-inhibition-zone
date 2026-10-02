"""Evaluate one checkpoint on every acquisition batch, to see the per-plate spread."""
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(r"F:\Code\SCI\Nanjing_Children_Hospital\抑菌区域分析_返修\药片CNN")
sys.path.insert(0, str(HERE))
from train import (CLASSES, PillCNN, build_loader, evaluate, load_manifest,  # noqa: E402
                   metrics_from_cm, prewarm_cache)

ck_path = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "runs" / "plate_main" / "best.pt"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
rows = load_manifest(HERE / "manifest_plate.csv")
plates = sorted({r["plate"] for r in rows})

cache = prewarm_cache(rows, 0.0, 192)
model = PillCNN().to(device)
sd = torch.load(ck_path, map_location=device, weights_only=False)
model.load_state_dict(sd["state"])
model.eval()

print(f"checkpoint {ck_path}")
print(f"\n{'plate':22} {'n':>6} {'disks':>6} {'acc':>7} {'macroF1':>8}  per-class recall")
accs = []
for p in plates:
    sel = [r for r in rows if r["plate"] == p]
    loader = build_loader(sel, False, 160, 0.0, 64, 0, tta=1, cache=cache)
    cm, _ = evaluate(model, loader, device)
    m = metrics_from_cm(cm)
    accs.append(m["accuracy"])
    rec = " ".join(f"{c}={m['per_class'][c]['recall']:.2f}"
                   for c in CLASSES if m["per_class"][c]["support"] > 0)
    print(f"{p:22} {m['n']:>6} {len({r['base_key'] for r in sel}):>6} "
          f"{m['accuracy']:>7.4f} {m['macro_f1']:>8.4f}  {rec}")

print(f"\nper-plate mean {np.mean(accs):.4f} +- {np.std(accs):.4f} "
      f"(range {min(accs):.4f}-{max(accs):.4f})")
