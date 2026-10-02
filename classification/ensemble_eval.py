"""Evaluate single models and their ensemble on the shared held-out test split.

Averages softmax probabilities across the given checkpoints. Reports accuracy,
macro-F1 and balanced accuracy, plus an optional 5-view test-time augmentation.
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

CNN = Path(r"F:\Code\SCI\Nanjing_Children_Hospital\抑菌区域分析_返修\药片CNN")
sys.path.insert(0, str(CNN))
from disknet_model import DiskNet  # noqa: E402
from train_augmented import CLASSES, FolderSet, metrics  # noqa: E402

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def predict(ckpt: Path, data: Path, img_size: int, tta: int = 1, variant=None):
    sd = torch.load(ckpt, map_location=DEVICE, weights_only=False)
    var = variant or sd.get("variant", "small")
    model = DiskNet(7, var).to(DEVICE)
    model.load_state_dict(sd["state"])
    model.eval()
    ds = FolderSet(data / "images" / "test", False, online_aug=False,
                   img_size=img_size)
    from PIL import Image
    from torchvision import transforms as T
    import torch as _t
    probs, labels = [], []
    for p, y in ds.items:
        im = Image.open(p).convert("L")
        views = []
        if tta <= 1:
            views = [im]
        else:
            views = [im, T.functional.hflip(im), T.functional.vflip(im)]
            if tta >= 5:
                views.append(im.rotate(-10, resample=Image.BICUBIC, fillcolor=128))
                views.append(im.rotate(10, resample=Image.BICUBIC, fillcolor=128))
        acc = []
        for v in views:
            v = v.resize((img_size, img_size), Image.BICUBIC)
            x = (T.functional.to_tensor(v) - 0.5) / 0.25
            acc.append(model(x.repeat(3, 1, 1)[None].to(DEVICE)))
        logit = torch.stack(acc).mean(0)
        probs.append(F.softmax(logit.float(), 1).cpu().numpy()[0])
        labels.append(y)
    return np.stack(probs), np.array(labels)


def score(probs, labels):
    cm = np.zeros((7, 7), dtype=np.int64)
    for t, p in zip(labels, probs.argmax(1)):
        cm[t, p] += 1
    return metrics(cm), cm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="*", default=None,
                    help="run dir names under runs/")
    ap.add_argument("--data", default=str(CNN / "dataset_augmented224"))
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--tta", type=int, default=1)
    args = ap.parse_args()

    runs = args.runs
    if runs is None:
        runs = [p.name for p in sorted((CNN / "runs").iterdir())
                if p.is_dir() and p.name.startswith("ens_")
                and (p / "best.pt").exists()]
    print(f"runs: {runs}   tta={args.tta}")
    data = Path(args.data)

    all_probs, labels = [], None
    for r in runs:
        p = CNN / "runs" / r / "best.pt"
        if not p.exists():
            print(f"  skip {r}: no best.pt")
            continue
        pr, lb = predict(p, data, args.img_size, args.tta)
        labels = lb
        m, _ = score(pr, lb)
        print(f"  {r:24} acc {m['accuracy']:.4f} mF1 {m['macro_f1']:.4f} "
              f"bal {m['balanced_accuracy']:.4f}")
        all_probs.append(pr)

    if not all_probs:
        raise SystemExit("no checkpoints")

    ens = np.mean(all_probs, axis=0)
    m, cm = score(ens, labels)
    print(f"\n=== ensemble of {len(all_probs)} models (tta={args.tta}) ===")
    print(f"TEST acc {m['accuracy']:.4f}  macro-F1 {m['macro_f1']:.4f}  "
          f"balanced {m['balanced_accuracy']:.4f}  n={m['n']}")
    print(f"\n{'class':6} {'n':>4} {'recall':>7} {'prec':>7} {'F1':>7}")
    for c in CLASSES:
        v = m["per_class"][c]
        print(f"{c:6} {v['support']:>4} {v['recall']:>7.3f} {v['precision']:>7.3f} "
              f"{v['f1']:>7.3f}")
    print("\nensemble confusion (rows true, cols pred):")
    print("      " + "".join(f"{c:>6}" for c in CLASSES))
    for i, c in enumerate(CLASSES):
        print(f"{c:5} " + "".join(f"{v:>6d}" for v in cm[i]))

    singles = [score(p, labels)[0]["accuracy"] for p in all_probs]
    print(f"\nsingle-model acc: mean {np.mean(singles):.4f} "
          f"(min {min(singles):.4f} max {max(singles):.4f})")
    print(f"ensemble gain over mean single: {m['accuracy'] - np.mean(singles):+.4f}")

    out = CNN / "runs" / f"ensemble_{len(all_probs)}.json"
    out.write_text(json.dumps({"runs": runs, "tta": args.tta, "metrics": m,
                               "cm": cm.tolist(),
                               "singles": singles}, indent=1), encoding="utf-8")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
