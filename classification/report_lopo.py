"""Report leave-one-plate-out per-plate results (works for any run dir)."""
import json
import sys
from pathlib import Path

HERE = Path(r"F:\Code\SCI\Nanjing_Children_Hospital\抑菌区域分析_返修\药片CNN")
run = HERE / "runs" / (sys.argv[1] if len(sys.argv) > 1 else "smoke_lopo")
CLASSES = ["CRO", "DA", "E", "LEV", "LZD", "P", "VA"]

files = sorted(run.glob("plate*_result.json"))
if not files:
    print(f"no per-plate results in {run}")
    raise SystemExit(1)

print(f"{'plate':22} {'n':>6} {'disks':>6} {'acc':>7} {'macroF1':>8} {'valAcc':>7}")
accs, f1s = [], []
for f in files:
    d = json.loads(f.read_text(encoding="utf-8"))
    accs.append(d["accuracy"])
    f1s.append(d["macro_f1"])
    print(f"{d['plate']:22} {d['n']:>6} {d.get('n_te_disks', 0):>6} "
          f"{d['accuracy']:>7.4f} {d['macro_f1']:>8.4f} "
          f"{d.get('val_accuracy', float('nan')):>7.4f}")

import numpy as np
print(f"\nfolds {len(accs)} | mean acc {np.mean(accs):.4f} +- {np.std(accs):.4f} "
      f"| mean macro-F1 {np.mean(f1s):.4f} +- {np.std(f1s):.4f}")
print(f"acc range {min(accs):.4f} - {max(accs):.4f}")

# pooled confusion over all folds
cm = np.sum([np.array(json.loads(f.read_text(encoding='utf-8'))["cm"])
             for f in files], axis=0)
n = cm.sum()
print(f"\npooled recall per class (n={n}):")
for i, c in enumerate(CLASSES):
    r = cm[i, i] / max(cm[i].sum(), 1)
    print(f"  {c:4} n={cm[i].sum():5d} recall {r:.3f}")
print(f"pooled accuracy {np.trace(cm)/n:.4f}")
