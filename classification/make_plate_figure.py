"""Batch-level (plate) evaluation figure and summary.

Built from a leave-one-plate-out run directory. Produces:
  figures/FigureS_PlateGeneralisation.png   per-plate accuracy/F1 + pooled matrix
  reports/plate_generalisation.csv          per-plate table

python make_plate_figure.py lopo_main
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

HERE = Path(__file__).resolve().parent
REPORTS = HERE / "reports"
FIGS = HERE / "figures"
MM = 1 / 25.4
DPI = 500
CLASSES = ["CRO", "DA", "E", "LEV", "LZD", "P", "VA"]
FULL = {"CRO": "Ceftriaxone", "DA": "Clindamycin", "E": "Erythromycin",
        "LEV": "Levofloxacin", "LZD": "Linezolid", "P": "Penicillin",
        "VA": "Vancomycin"}

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 7.5, "axes.labelsize": 7.5, "axes.titlesize": 8,
    "xtick.labelsize": 6.8, "ytick.labelsize": 6.8, "legend.fontsize": 6.5,
    "axes.linewidth": 0.6, "figure.dpi": DPI, "savefig.dpi": DPI,
    "axes.spines.top": False, "axes.spines.right": False,
})
BLUE, RED, GREY = "#2166ac", "#b2182b", "#8c8c8c"


def load(run):
    d = HERE / "runs" / run
    files = sorted(d.glob("plate*_result.json"),
                   key=lambda p: int(p.stem.replace("plate", "").split("_")[0]))
    return [json.loads(f.read_text(encoding="utf-8")) for f in files]


def main():
    run = sys.argv[1] if len(sys.argv) > 1 else "lopo_main"
    res = load(run)
    if not res:
        print(f"no per-plate results in runs/{run}")
        return
    FIGS.mkdir(exist_ok=True)
    REPORTS.mkdir(exist_ok=True)

    # ---- table ----
    with open(REPORTS / f"plate_generalisation_{run}.csv", "w", newline="",
              encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["plate", "n_images", "n_disks", "classes", "accuracy",
                    "macro_f1", "balanced_accuracy", "val_plate", "val_accuracy"])
        for m in res:
            cov = sum(1 for c in CLASSES if m["per_class"][c]["support"] > 0)
            w.writerow([m["plate"], m["n"], m.get("n_te_disks", ""), cov,
                        f"{m['accuracy']:.4f}", f"{m['macro_f1']:.4f}",
                        f"{m['balanced_accuracy']:.4f}", m.get("val_plate", ""),
                        f"{m.get('val_accuracy', float('nan')):.4f}"])

    order = sorted(res, key=lambda m: -m["accuracy"])
    labels = [f"{m['plate'][-4:]}\n{sum(1 for c in CLASSES if m['per_class'][c]['support'] > 0)}/7"
              for m in order]
    accs = [m["accuracy"] for m in order]
    f1s = [m["macro_f1"] for m in order]
    nimg = [m["n"] for m in order]

    fig = plt.figure(figsize=(6.25, 4.2))
    gs = GridSpec(1, 2, figure=fig, width_ratios=[1.45, 1], wspace=0.3,
                  left=0.085, right=0.985, top=0.86, bottom=0.20)

    ax = fig.add_subplot(gs[0, 0])
    x = np.arange(len(order))
    ax.bar(x - 0.2, accs, 0.4, color=BLUE, label="Accuracy")
    ax.bar(x + 0.2, f1s, 0.4, color="#92c5de", label="Macro-F1")
    ax.axhline(1 / 7, color="k", ls=":", lw=0.8)
    ax.text(len(order) - 0.6, 1 / 7 + 0.02, "chance", fontsize=6, ha="right")
    ax.set_xticks(x, labels, fontsize=5.6)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_xlabel("Held-out acquisition batch (last 4 digits) / classes present", labelpad=2)
    ax.set_title("Leave-one-batch-out: performance on unseen plates", pad=4)
    ax.legend(frameon=False, ncol=2, loc="upper center",
              bbox_to_anchor=(0.5, -0.22), handlelength=1.4)
    for i, (a, n) in enumerate(zip(accs, nimg)):
        ax.text(i - 0.2, a + 0.015, f"{a:.2f}", ha="center", fontsize=5.2)
    ax.grid(axis="y", alpha=0.25, lw=0.5)
    ax.set_axisbelow(True)

    ax2 = fig.add_subplot(gs[0, 1])
    cm = np.sum([np.array(m["cm"], dtype=float) for m in res], axis=0)
    cmn = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    im = ax2.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    ax2.set_xticks(range(7), CLASSES, rotation=90)
    ax2.set_yticks(range(7), CLASSES)
    ax2.set_xlabel("Predicted")
    ax2.set_ylabel("True")
    ax2.set_title("Pooled over all held-out batches", pad=4)
    for i in range(7):
        for j in range(7):
            v = cmn[i, j]
            if v >= 0.005:
                ax2.text(j, i, f"{v:.2f}".lstrip("0") if v < 1 else "1.0",
                         ha="center", va="center", fontsize=5.6,
                         color="white" if v > 0.55 else "#1a1a1a")
    ax2.set_xticks(np.arange(-.5, 7, 1), minor=True)
    ax2.set_yticks(np.arange(-.5, 7, 1), minor=True)
    ax2.grid(which="minor", color="white", lw=0.6)
    ax2.tick_params(which="minor", length=0)

    for ext, kw in [("png", {}), ("tiff", dict(pil_kwargs={"compression": "tiff_lzw"}))]:
        fig.savefig(FIGS / f"FigureS_PlateGeneralisation.{ext}",
                    bbox_inches="tight", **kw)
    plt.close(fig)
    from PIL import Image
    with Image.open(FIGS / "FigureS_PlateGeneralisation.png") as im2:
        w, h = im2.size
    print(f"wrote FigureS_PlateGeneralisation.png {w}x{h} px = "
          f"{w/DPI*25.4:.1f} x {h/DPI*25.4:.1f} mm")

    print(f"\n{len(res)} held-out batches")
    print(f"  accuracy   mean {np.mean(accs):.4f} +- {np.std(accs):.4f} "
          f"(range {min(accs):.4f}-{max(accs):.4f})")
    print(f"  macro-F1   mean {np.mean(f1s):.4f} +- {np.std(f1s):.4f}")
    print(f"  pooled accuracy {np.trace(cm)/cm.sum():.4f}")
    print("\n  pooled recall per class:")
    for i, c in enumerate(CLASSES):
        print(f"    {c:4} n={int(cm[i].sum()):5d} recall {cm[i,i]/max(cm[i].sum(),1):.3f}")


if __name__ == "__main__":
    main()
