"""Evaluate a saved classifier checkpoint on any split of a manifest.

Useful for scoring the batch-held-out test set without retraining.

python eval_checkpoint.py --checkpoint runs/plate_main/best.pt \
    --manifest manifest_plate_split.csv --split test
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from train import (CLASSES, PillCNN, build_loader, evaluate, load_manifest,  # noqa: E402
                   metrics_from_cm, prewarm_cache)

FULL = {"CRO": "Ceftriaxone", "DA": "Clindamycin", "E": "Erythromycin",
        "LEV": "Levofloxacin", "LZD": "Linezolid", "P": "Penicillin",
        "VA": "Vancomycin"}


def wilson(k, n, z=1.96):
    if n == 0:
        return 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--manifest", default=str(HERE / "manifest_plate_split.csv"))
    ap.add_argument("--split", default="test",
                    help="value of split_plate (plate manifest) or orig_split")
    ap.add_argument("--plate", default=None,
                    help="restrict to a single acquisition batch (matches manifest_plate.csv)")
    ap.add_argument("--img-size", type=int, default=160)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--tta", type=int, default=1)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = load_manifest(Path(args.manifest))
    key = "split_plate" if "split_plate" in rows[0] else "orig_split"
    if args.plate:
        sel = [r for r in rows if r.get("plate") == args.plate]
    else:
        sel = [r for r in rows if r[key] == args.split]
    if not sel:
        raise SystemExit(f"no rows matched (plate={args.plate}, {key}={args.split!r})")
    print(f"evaluating {len(sel)} images "
          f"({'plate ' + args.plate if args.plate else key + '=' + args.split}) "
          f"({len({r['base_key'] for r in sel})} disks, "
          f"{len({r.get('plate', '-') for r in sel})} plates)")

    cache = prewarm_cache(sel, 0.0, max(192, args.img_size + 32))
    model = PillCNN().to(device)
    sd = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(sd["state"] if "state" in sd else sd)
    model.eval()

    loader = build_loader(sel, False, args.img_size, 0.0, args.batch, 0,
                          tta=args.tta, cache=cache)
    cm, loss = evaluate(model, loader, device)
    m = metrics_from_cm(cm)
    lo, hi = wilson(int(np.trace(cm)), int(cm.sum()))

    print(f"\naccuracy {m['accuracy']:.4f} (95% CI {lo:.4f}-{hi:.4f}, "
          f"image-level) | macro-F1 {m['macro_f1']:.4f} | "
          f"balanced {m['balanced_accuracy']:.4f} | macro spec {m['macro_specificity']:.4f}")
    print(f"macro precision {m['macro_precision']:.4f} | loss {loss:.4f}")
    print(f"\n{'class':5} {'n':>5} {'prec':>7} {'recall':>7} {'F1':>7} {'spec':>7}")
    for c in CLASSES:
        v = m["per_class"][c]
        print(f"{c:5} {v['support']:>5} {v['precision']:>7.3f} {v['recall']:>7.3f} "
              f"{v['f1']:>7.3f} {v['specificity']:>7.3f}")

    cm = np.asarray(cm)
    print("\nconfusion (rows true, cols predicted):")
    print("      " + "".join(f"{c:>7}" for c in CLASSES))
    for i, c in enumerate(CLASSES):
        print(f"{c:5} " + "".join(f"{v:>7d}" for v in cm[i]))

    print("\ntop confusions:")
    pairs = [(cm[i, j], CLASSES[i], CLASSES[j])
             for i in range(7) for j in range(7) if i != j and cm[i, j]]
    for n, a, b in sorted(pairs, reverse=True)[:8]:
        print(f"  {a:4} -> {b:4} {n:5d} ({n / cm[CLASSES.index(a)].sum():.1%} of {a})")

    if args.out:
        json.dump(dict(metrics=m, cm=cm.tolist(), split=args.split,
                       checkpoint=args.checkpoint), open(args.out, "w"), indent=1)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
