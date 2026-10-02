"""Replicate runs with different batch-level splits, to check reproducibility.

The augmented dataset has only 28 held-out test images, so a single number can be
luck of the split. This rebuilds the split with several seeds (batches kept whole,
sparse classes stratified) and retrains, reporting mean and spread.
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

CNN = Path(r"F:\Code\SCI\Nanjing_Children_Hospital\抑菌区域分析_返修\药片CNN")
SRC = CNN / "dataset_expanded"
CLASSES = ["CRO", "DA", "E", "LEV", "LZD", "P", "VA"]


def build_split(seed: int, n_train_aug: int, out: Path):
    """Copy originals into a fresh batch-stratified split, augmenting train only."""
    if out.exists():
        shutil.rmtree(out)
    rows = list(csv.DictReader(open(SRC / "manifest.csv", encoding="utf-8")))
    by_batch = defaultdict(list)
    for r in rows:
        by_batch[r["batch"]].append(r)
    batches = sorted(by_batch)
    rarity = {c: sum(1 for b in batches
                     if any(x["label"] == c for x in by_batch[b]))
              for c in CLASSES}

    def stratum(b):
        present = [c for c in CLASSES if any(x["label"] == c for x in by_batch[b])]
        return min(present, key=lambda c: rarity[c])

    X = batches
    y = [stratum(b) for b in batches]
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    folds = list(sgkf.split(X, y, groups=X))
    assign = {}
    for fi, (_, idx) in enumerate(folds):
        for i in idx:
            assign[X[i]] = {3: "test", 4: "val"}.get(fi, "train")

    # materialise originals, then augment train
    rng = np.random.default_rng(seed)
    made = 0
    for r in rows:
        split = assign[r["batch"]]
        src = SRC / "images" / r["split"] / r["label"] / r["file"]
        if not src.exists():
            src = SRC / "images" / split / r["label"] / r["file"]
        d = out / "images" / split / r["label"]
        d.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, d / r["file"])
        made += 1
    return out, assign, made


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="*", default=[1, 2, 3])
    ap.add_argument("--epochs", type=int, default=40)
    args = ap.parse_args()

    results = []
    for s in args.seeds:
        data = CNN / f"dataset_replicate_s{s}"
        build_split(s, 0, data)
        # augment train in place: replicate each train image 6x with mild transforms
        sys.path.insert(0, str(CNN))
        from augment_lib import mild_combo  # noqa: E402
        from PIL import Image  # noqa: E402
        import random  # noqa: E402
        rng = random.Random(1000 + s)
        tr = data / "images" / "train"
        n_add = 0
        for cls_dir in sorted(p for p in tr.iterdir() if p.is_dir()):
            for p in sorted(cls_dir.glob("*.png")):
                if "__aug" in p.stem:
                    continue
                with Image.open(p) as im:
                    base = im.convert("L")
                for k in range(6):
                    aug, _ = mild_combo(base, rng, strength="mild")
                    aug.save(cls_dir / f"{p.stem}__aug{k:02d}.png", format="PNG")
                    n_add += 1
        print(f"\n=== seed {s}: train augmented with {n_add} variants -> {data.name} ===")

        cmd = [sys.executable, str(CNN / "train_augmented.py"),
               "--data", str(data), "--run-name", f"aug_rep_s{s}",
               "--epochs", str(args.epochs), "--batch", "32"]
        log = CNN / "logs" / f"aug_rep_s{s}.log"
        log.parent.mkdir(exist_ok=True)
        with open(log, "w", encoding="utf-8") as f:
            p = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                 stderr=subprocess.STDOUT, text=True,
                                 encoding="utf-8", errors="replace", bufsize=1)
            last = ""
            for line in p.stdout:
                f.write(line)
                if "TEST (held-out" in line or "val :" in line:
                    last += line
            p.wait()
        res = json.loads((CNN / "runs" / f"aug_rep_s{s}" / "results.json")
                         .read_text(encoding="utf-8"))
        results.append(dict(seed=s, **{k: res["test"][k] for k in
                                       ("accuracy", "macro_f1",
                                        "balanced_accuracy")},
                            n=res["test"]["n"]))
        print(f"  seed {s}: test acc {res['test']['accuracy']:.4f} "
              f"macro-F1 {res['test']['macro_f1']:.4f}")

    accs = [r["accuracy"] for r in results]
    f1s = [r["macro_f1"] for r in results]
    print(f"\n=== over {len(results)} splits ===")
    print(f"  test accuracy  {np.mean(accs):.4f} +- {np.std(accs):.4f} "
          f"(range {min(accs):.4f}-{max(accs):.4f})")
    print(f"  test macro-F1  {np.mean(f1s):.4f} +- {np.std(f1s):.4f} "
          f"(range {min(f1s):.4f}-{max(f1s):.4f})")
    for r in results:
        print(f"    seed {r['seed']}: acc {r['accuracy']:.4f} mF1 {r['macro_f1']:.4f}")

    out = CNN / "runs" / "aug_replicates.json"
    out.write_text(json.dumps(results, indent=1), encoding="utf-8")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
