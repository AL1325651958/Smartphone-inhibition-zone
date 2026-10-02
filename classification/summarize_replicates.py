"""Summarise the replicate runs."""
import json
import statistics as st
from pathlib import Path

CNN = Path(r"F:\Code\SCI\Nanjing_Children_Hospital\抑菌区域分析_返修\药片CNN")
CLASSES = ["CRO", "DA", "E", "LEV", "LZD", "P", "VA"]

accs, f1s, bals, vals = [], [], [], []
for s in (1, 2, 3):
    p = CNN / "runs" / f"aug_rep_s{s}" / "results.json"
    if not p.exists():
        print(f"seed {s}: no results")
        continue
    r = json.loads(p.read_text(encoding="utf-8"))
    t, v = r["test"], r["val"]
    accs.append(t["accuracy"]); f1s.append(t["macro_f1"])
    bals.append(t["balanced_accuracy"]); vals.append(v["accuracy"])
    print(f"seed {s}: TEST acc {t['accuracy']:.4f}  macro-F1 {t['macro_f1']:.4f}  "
          f"balanced {t['balanced_accuracy']:.4f}  (n={t['n']})   "
          f"| val acc {v['accuracy']:.4f}")

if accs:
    print()
    print(f"TEST accuracy   mean {st.mean(accs):.4f} +- {st.pstdev(accs):.4f}  "
          f"range {min(accs):.4f}-{max(accs):.4f}")
    print(f"TEST macro-F1   mean {st.mean(f1s):.4f} +- {st.pstdev(f1s):.4f}  "
          f"range {min(f1s):.4f}-{max(f1s):.4f}")
    print(f"TEST balanced   mean {st.mean(bals):.4f}")
    print(f"VAL  accuracy   mean {st.mean(vals):.4f}")

# main 60-epoch run for reference
p = CNN / "runs" / "aug_run" / "results.json"
if p.exists():
    r = json.loads(p.read_text(encoding="utf-8"))
    t = r["test"]
    print(f"\nmain run (60 epochs, dataset_augmented): TEST acc {t['accuracy']:.4f} "
          f"macro-F1 {t['macro_f1']:.4f}")
    print("per-class (main run):")
    for c in CLASSES:
        v = t["per_class"][c]
        print(f"  {c:4} n={v['support']:>3} recall {v['recall']:.3f} "
              f"prec {v['precision']:.3f} F1 {v['f1']:.3f}")
