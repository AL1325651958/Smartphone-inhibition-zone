"""Per-image predictions and error analysis from the saved fold checkpoints.

Every image is scored by each fold model that did NOT train on its pill, so the
output is a genuine out-of-fold ensemble prediction.  Produces:
  reports/predictions.csv        per-image predicted class + confidence
  reports/reliability.png        confidence calibration
  reports/error_gallery.png      most confident mistakes with true/predicted labels
  reports/confidence_stats.csv   summary of confidence vs correctness
"""
from __future__ import annotations

import csv
import json
import sys
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from train import CLASSES, EvalDataset, PillCNN, load_manifest, prewarm_cache  # noqa: E402

REPORTS = HERE / "reports"
RUNS = HERE / "runs"


def group_folds(rows, k, seed=0):
    import random
    folds = {}
    for ci, c in enumerate(CLASSES):
        keys = sorted({r["base_key"] for r in rows if r["class"] == c})
        rng = random.Random(seed * 1000 + ci)
        rng.shuffle(keys)
        for i, key in enumerate(keys):
            folds[key] = i % k
    return folds


@torch.no_grad()
def predict_rows(model, rows, device, img_size=160, tta=5, batch=64):
    ds = EvalDataset(rows, img_size=img_size, tta=tta)
    out = []
    for s in range(0, len(ds), batch):
        chunk = [ds[i] for i in range(s, min(s + batch, len(ds)))]
        x = torch.stack([c[0] for c in chunk])
        b, v = x.shape[:2]
        x = x.view(b * v, *x.shape[2:]).to(device)
        logits = model(x).view(b, v, -1).mean(1)
        p = F.softmax(logits.float(), 1).cpu().numpy()
        out.append(p)
    return np.concatenate(out)


def fig_reliability(conf, correct, path, nbin=10):
    edges = np.linspace(1 / 7, 1.0, nbin + 1)
    xs, ys, ns, ece = [], [], [], 0.0
    for i in range(nbin):
        m = (conf >= edges[i]) & (conf < edges[i + 1] if i < nbin - 1 else conf <= 1.0)
        if m.sum() == 0:
            continue
        xs.append(conf[m].mean())
        ys.append(correct[m].mean())
        ns.append(m.sum())
        ece += m.sum() / len(conf) * abs(ys[-1] - xs[-1])
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(9.6, 3.9),
                                  gridspec_kw={"width_ratios": [1, 1.15]})
    ax.plot([1 / 7, 1], [1 / 7, 1], ls="--", c="grey", lw=1.1, label="perfect calibration")
    ax.plot(xs, ys, "o-", c="#2166ac", lw=1.6, ms=5, label="CNN (out-of-fold)")
    ax.set_xlabel("Predicted confidence")
    ax.set_ylabel("Observed accuracy")
    ax.set_title(f"Reliability diagram (ECE = {ece:.3f})")
    ax.legend(fontsize=8, frameon=False)
    ax.grid(alpha=0.25, lw=0.6)
    ax.set_xlim(1 / 7 - 0.02, 1.02)
    ax.set_ylim(0, 1.02)

    ax2.hist(conf[correct], bins=25, range=(0, 1), color="#4393c3", alpha=0.85,
             label=f"correct (n={correct.sum()})")
    ax2.hist(conf[~correct], bins=25, range=(0, 1), color="#d6604d", alpha=0.75,
             label=f"incorrect (n={(~correct).sum()})")
    ax2.set_xlabel("Predicted confidence")
    ax2.set_ylabel("Number of images")
    ax2.set_title("Confidence distribution")
    ax2.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return dict(ece=float(ece), mean_confidence=float(conf.mean()),
                accuracy=float(correct.mean()))


def fig_error_gallery(rows, probs, path, n=16):
    pred = probs.argmax(1)
    true = np.array([r["class_idx"] for r in rows])
    conf = probs.max(1)
    wrong = np.where(pred != true)[0]
    wrong = wrong[np.argsort(-conf[wrong])][:n]
    if len(wrong) == 0:
        return
    cols = 8
    rowsn = int(np.ceil(len(wrong) / cols))
    from PIL import Image
    fig, axes = plt.subplots(rowsn, cols, figsize=(cols * 1.35, rowsn * 1.72))
    axes = np.atleast_1d(axes).ravel()
    for ax in axes:
        ax.axis("off")
    for ax, i in zip(axes, wrong):
        im = Image.open(rows[i]["path"]).convert("L").resize((112, 112))
        ax.imshow(np.asarray(im), cmap="gray", vmin=0, vmax=255)
        ax.set_title(f"T:{CLASSES[true[i]]} P:{CLASSES[pred[i]]}\n{conf[i]:.2f}",
                     fontsize=7, color="#b2182b")
        ax.axis("off")
    fig.suptitle("Most confident misclassifications (out-of-fold ensemble)", y=1.0,
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", dpi=200)
    plt.close(fig)


def main():
    REPORTS.mkdir(exist_ok=True)
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, default=3, help="folds used by the group run")
    ap.add_argument("--split", default="group", help="run directory holding fold*.pt")
    ap.add_argument("--img-size", type=int, default=160)
    ap.add_argument("--tta", type=int, default=1)
    args = ap.parse_args()

    rows = load_manifest(HERE / "manifest.csv")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    prewarm_cache(rows, 0.0, 192)

    k = args.folds
    folds = group_folds(rows, k, seed=0)
    ckpts = []
    for f in range(k):
        p = RUNS / args.split / f"fold{f}.pt"
        if p.exists():
            ckpts.append((f, p))
    if not ckpts:
        print(f"no fold checkpoints under runs/{args.split} - run train.py first")
        return
    print(f"using {len(ckpts)} fold checkpoints as an out-of-fold ensemble")

    models = []
    for f, p in ckpts:
        m = PillCNN().to(device)
        sd = torch.load(p, map_location=device, weights_only=False)
        m.load_state_dict(sd["state"])
        m.eval()
        models.append((f, m))
        print(f"  fold{f}: val macro-F1 {sd['metrics'].get('macro_f1', float('nan')):.4f}")

    probs = np.zeros((len(rows), 7), dtype=np.float64)
    counts = np.zeros(len(rows))
    for f, m in models:
        idx = [i for i, r in enumerate(rows) if folds[r["base_key"]] == f]
        p = predict_rows(m, [rows[i] for i in idx], device, args.img_size, args.tta)
        probs[idx] += p
        counts[idx] += 1
    # normalise: every image is scored only by models that never saw its pill
    probs = probs / np.maximum(counts[:, None], 1)

    true = np.array([r["class_idx"] for r in rows])
    pred = probs.argmax(1)
    conf = probs.max(1)
    correct = pred == true
    # only keep images that were actually scored
    keep = counts > 0
    print(f"scored {keep.sum()}/{len(rows)} images | accuracy {correct[keep].mean():.4f} "
          f"| macro-F1 {(np.mean([_f1(true[keep] == i, pred[keep] == i) for i in range(7)])):.4f}")

    with open(REPORTS / "predictions.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["path", "base_key", "true", "pred", "confidence", "correct",
                    *[f"p_{c}" for c in CLASSES]])
        for i in range(len(rows)):
            if not keep[i]:
                continue
            w.writerow([rows[i]["path"], rows[i]["base_key"], CLASSES[true[i]],
                        CLASSES[pred[i]], f"{conf[i]:.4f}", int(correct[i]),
                        *[f"{v:.4f}" for v in probs[i]]])

    stats = fig_reliability(conf[keep], correct[keep], REPORTS / "reliability.png")
    stats.update(n_scored=int(keep.sum()), accuracy=float(correct[keep].mean()))
    fig_error_gallery([rows[i] for i in np.where(keep)[0]], probs[keep],
                      REPORTS / "error_gallery.png")

    # accuracy retained at increasing confidence thresholds: tells the operator
    # which calls the system can auto-accept and which need human review
    thr_rows = []
    for t in [0.0, 0.5, 0.7, 0.9, 0.95, 0.99, 0.999]:
        m = conf[keep] >= t
        if m.sum() == 0:
            continue
        thr_rows.append(dict(threshold=t, n_retained=int(m.sum()),
                             coverage=float(m.mean()),
                             accuracy=float(correct[keep][m].mean())))
    with open(REPORTS / "confidence_thresholds.csv", "w", newline="",
              encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(thr_rows[0]))
        w.writeheader()
        for r in thr_rows:
            w.writerow({k: (f"{v:.4f}" if isinstance(v, float) else v)
                        for k, v in r.items()})
    stats["auto_accept_ge_0.70"] = next(
        (r["accuracy"] for r in thr_rows if r["threshold"] == 0.7), float("nan"))
    stats["coverage_ge_0.70"] = next(
        (r["coverage"] for r in thr_rows if r["threshold"] == 0.7), float("nan"))

    with open(REPORTS / "confidence_stats.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        for kk, vv in stats.items():
            w.writerow([kk, vv])
    print(stats)
    print("\naccuracy at increasing confidence thresholds:")
    for r in thr_rows:
        print(f"  conf >= {r['threshold']:.3f}: retain {r['coverage']:6.1%} "
              f"({r['n_retained']:5d} crops), accuracy {r['accuracy']:.4f}")

    # where the errors go
    cm = np.zeros((7, 7), int)
    for t, p in zip(true[keep], pred[keep]):
        cm[t, p] += 1
    print("\ntop confusions (true -> predicted):")
    pairs = [(cm[i, j], CLASSES[i], CLASSES[j]) for i in range(7) for j in range(7) if i != j]
    for cnt, a, b in sorted(pairs, reverse=True)[:8]:
        print(f"  {a:4} -> {b:4}  {cnt:5d}  ({cnt/cm[CLASSES.index(a)].sum():.1%} of {a})")
    print(f"\nartifacts -> {REPORTS}")


def _f1(tp_mask, pred_mask):
    tp = (tp_mask & pred_mask).sum()
    p = tp / max(pred_mask.sum(), 1)
    r = tp / max(tp_mask.sum(), 1)
    return 2 * p * r / max(p + r, 1e-9)


if __name__ == "__main__":
    main()
