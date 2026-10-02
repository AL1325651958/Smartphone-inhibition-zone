"""Publication figure for the antibiotic-abbreviation classifier (Figure 6 style).

Sized to the conventions used by the rest of this manuscript
(写作/数据处理/*/plot_*.py): explicit inch figsize, 500 dpi, tight bbox,
LZW-compressed TIFF, sans-serif 7-10 pt text.

Figure 1  Figure6_Classifier.tiff/.png   180 mm double-column composite
Figure 2  FigureS_ClassifierProtocols    grouped vs shipped-split comparison

Run after train.py + predict.py.
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
sys.path.insert(0, str(HERE))

# ---------------------------------------------------------------- figure style
MM = 1 / 25.4
COL_1, COL_1_5, COL_2 = 90 * MM, 140 * MM, 180 * MM   # BMC column widths
DPI = 500
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 7.5,
    "axes.labelsize": 7.5,
    "axes.titlesize": 8,
    "xtick.labelsize": 6.8,
    "ytick.labelsize": 6.8,
    "legend.fontsize": 6.5,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 2.4,
    "ytick.major.size": 2.4,
    "lines.linewidth": 1.0,
    "figure.dpi": DPI,
    "savefig.dpi": DPI,
    "savefig.pad_inches": 0.0,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

CLASSES = ["CRO", "DA", "E", "LEV", "LZD", "P", "VA"]
FULL = {"CRO": "Ceftriaxone", "DA": "Clindamycin", "E": "Erythromycin",
        "LEV": "Levofloxacin", "LZD": "Linezolid", "P": "Penicillin",
        "VA": "Vancomycin"}
BLUE, RED, GREY = "#2166ac", "#b2182b", "#8c8c8c"
OUT = HERE / "figures"
OUT.mkdir(exist_ok=True)


def save_fig(fig, stem, target_mm=None):
    """Save PNG + LZW TIFF and report the physical size at 500 dpi."""
    for ext, kw in [("png", {}), ("tiff", dict(pil_kwargs={"compression": "tiff_lzw"}))]:
        fig.savefig(OUT / f"{stem}.{ext}", bbox_inches="tight", **kw)
    from PIL import Image
    with Image.open(OUT / f"{stem}.png") as im:
        w, h = im.size
    wmm, hmm = w / DPI * 25.4, h / DPI * 25.4
    flag = ""
    if target_mm and wmm > target_mm + 0.5:
        flag = f"  ** exceeds {target_mm:.0f} mm by {wmm-target_mm:.1f} mm **"
    print(f"wrote {OUT/(stem + '.png')}  {w}x{h} px = {wmm:.1f} x {hmm:.1f} mm "
          f"at {DPI} dpi{flag}")


def panel_label(ax, letter, x=-0.16, y=1.10, size=9):
    ax.text(x, y, f"({letter})", transform=ax.transAxes, fontsize=size,
            fontweight="bold", va="bottom", ha="left")


def load(proto):
    p = HERE / "runs" / proto / "results.json"
    h = HERE / "runs" / proto / "history.json"
    if not p.exists():
        return None
    r = json.loads(p.read_text(encoding="utf-8"))
    r["_hist"] = json.loads(h.read_text(encoding="utf-8")) if h.exists() else None
    return r


def wilson(k, n, z=1.96):
    if n == 0:
        return 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def pooled(r):
    return r.get("pooled") or r.get("orig_split")


def fold_curves(hist, key):
    """Average metric across folds for cross-validation runs."""
    if hist and isinstance(hist[0], list):
        L = min(len(f) for f in hist)
        return np.mean([[f[e][key] for e in range(L)] for f in hist], axis=0)
    if hist and isinstance(hist[0], dict):
        return np.array([e[key] for e in hist])
    return None


# --------------------------------------------------------------------- panels
def panel_training(ax, orig, group, ctrl):
    ax.set_title("Training dynamics", pad=4)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    maxlen = 0
    for r, lab, c, ls in [(group, "Grouped CV (leave-pills-out)", BLUE, "-"),
                          (ctrl, "Grouped CV, label blurred", RED, "-"),
                          (orig, "Shipped split (pills shared)", GREY, "--")]:
        if not r or not r.get("_hist"):
            continue
        acc = fold_curves(r["_hist"], "accuracy")
        if acc is None or len(acc) == 0:
            continue
        maxlen = max(maxlen, len(acc))
        ax.plot(np.arange(1, len(acc) + 1), acc, color=c, ls=ls, lw=1.1, label=lab)
    # xlim must be set AFTER plotting: get_xlim() on an empty axes returns (0, 1)
    # and would clip every curve off-screen
    ax.set_xlim(1, max(maxlen, 2))
    ax.set_ylim(0, 1.03)
    ax.axhline(1 / 7, color="k", ls=":", lw=0.7)
    ax.text(ax.get_xlim()[1], 1 / 7 + 0.03, "chance", ha="right", fontsize=6, color="k")
    ax.legend(loc="center right", frameon=False, handlelength=1.6,
              bbox_to_anchor=(1.0, 0.42))


def panel_confusion(ax, cm, title, note=None):
    cm = np.asarray(cm, dtype=float)
    cm = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(7), CLASSES, rotation=90)
    ax.set_yticks(range(7), CLASSES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title, pad=4)
    for i in range(7):
        for j in range(7):
            v = cm[i, j]
            if v >= 0.005:
                ax.text(j, i, f"{v:.2f}".lstrip("0") if v < 1 else "1.0",
                        ha="center", va="center", fontsize=5.6,
                        color="white" if v > 0.55 else "#1a1a1a")
    ax.set_xticks(np.arange(-.5, 7, 1), minor=True)
    ax.set_yticks(np.arange(-.5, 7, 1), minor=True)
    ax.grid(which="minor", color="white", lw=0.6)
    ax.tick_params(which="minor", length=0)
    if note:
        ax.text(0.5, -0.36, note, transform=ax.transAxes, ha="center", fontsize=6,
                color="#444444")
    return im


def panel_perclass(ax, pm, title):
    per = pm["per_class"]
    y = np.arange(7)
    vals = np.array([per[c]["f1"] for c in CLASSES])
    # Wilson interval on recall, rescaled through the harmonic mean so the
    # whiskers stay on the F1 scale; clamped so they can never invert.
    lo_a, hi_a = [], []
    for c in CLASSES:
        rec, sup = per[c]["recall"], per[c]["support"]
        l, h = wilson(round(rec * sup), sup)
        f1 = per[c]["f1"]
        lo_a.append(f1 * l / rec if rec > 0 else f1)
        hi_a.append(f1 * h / rec if rec > 0 else f1)
    lo = np.minimum(np.array(lo_a), vals)
    hi = np.maximum(np.array(hi_a), vals)
    err = np.vstack([np.clip(vals - lo, 0.0, None), np.clip(hi - vals, 0.0, None)])
    err = np.where(np.isfinite(err), err, 0.0)
    err = np.abs(err)                       # guard against -0.0 / NaN
    ax.barh(y, vals, height=0.62, color=BLUE, xerr=err, capsize=1.6,
            error_kw=dict(lw=0.7, ecolor="#555555"))
    ax.set_yticks(y, [f"{c} ({FULL[c]})" for c in CLASSES], fontsize=6.4)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.06)
    ax.set_xlabel("F1 score")
    ax.set_title(title, pad=4)
    for i, v in enumerate(vals):
        ax.text(min(v + 0.03, 0.99), i, f"{v:.2f}", va="center", fontsize=6)
    ax.grid(axis="x", alpha=0.25, lw=0.5)
    ax.set_axisbelow(True)


def panel_reliability(ax, pred_csv):
    if not pred_csv.exists():
        ax.axis("off")
        return
    rows = list(csv.DictReader(pred_csv.open(encoding="utf-8")))
    conf = np.array([float(r["confidence"]) for r in rows])
    corr = np.array([int(r["correct"]) for r in rows]).astype(bool)
    nbin = 8
    edges = np.linspace(1 / 7, 1.0, nbin + 1)
    xs, ys, ns, ece = [], [], [], 0.0
    for i in range(nbin):
        m = (conf >= edges[i]) & (conf < edges[i + 1] if i < nbin - 1 else conf <= 1)
        if m.sum() < 5:
            continue
        xs.append(conf[m].mean()); ys.append(corr[m].mean()); ns.append(m.sum())
        ece += m.sum() / len(conf) * abs(ys[-1] - xs[-1])
    ax.plot([0, 1], [0, 1], ls="--", color=GREY, lw=0.8, label="Perfect calibration")
    ax.plot(xs, ys, "o-", color=BLUE, ms=3.2, lw=1.1, label="CNN")
    ax.set_xlim(0.1, 1.02); ax.set_ylim(0, 1.03)
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Observed accuracy")
    ax.set_title(f"Calibration (ECE = {ece:.3f})", pad=4)
    ax.legend(loc="upper left", frameon=False, handlelength=1.5)
    ax.grid(alpha=0.25, lw=0.5)


def panel_confidence_hist(ax, pred_csv):
    if not pred_csv.exists():
        ax.axis("off")
        return
    rows = list(csv.DictReader(pred_csv.open(encoding="utf-8")))
    conf = np.array([float(r["confidence"]) for r in rows])
    corr = np.array([int(r["correct"]) for r in rows]).astype(bool)
    ax.hist(conf[corr], bins=20, range=(0.1, 1), color=BLUE, alpha=0.85,
            label=f"Correct (n={corr.sum()})")
    ax.hist(conf[~corr], bins=20, range=(0.1, 1), color=RED, alpha=0.8,
            label=f"Incorrect (n={(~corr).sum()})")
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Disk crops")
    ax.set_title("Confidence distribution", pad=4)
    ax.legend(frameon=False, loc="upper left")
    ax.set_yscale("log")


# ------------------------------------------------------------------- figure 1
def figure6(orig, group, ctrl):
    # content width 6.25 in -> ~180 mm after the tight bbox trim
    fig = plt.figure(figsize=(6.25, 5.35))
    gs = GridSpec(2, 3, figure=fig, height_ratios=[1, 1.02], hspace=0.62, wspace=0.42,
                  left=0.075, right=0.985, top=0.93, bottom=0.085)

    ax_a = fig.add_subplot(gs[0, 0])
    panel_training(ax_a, orig, group, ctrl)
    panel_label(ax_a, "a")

    ax_b = fig.add_subplot(gs[0, 1])
    panel_confusion(ax_b, pooled(group)["cm"], "Confusion matrix - grouped CV")
    panel_label(ax_b, "b")

    ax_c = fig.add_subplot(gs[0, 2])
    if orig and pooled(orig).get("cm"):
        panel_confusion(ax_c, pooled(orig)["cm"], "Confusion matrix - shipped split")
    panel_label(ax_c, "c")

    ax_d = fig.add_subplot(gs[1, 0])
    panel_perclass(ax_d, pooled(group), "Per-class performance (grouped CV)")
    panel_label(ax_d, "d", x=-0.62)

    ax_e = fig.add_subplot(gs[1, 1])
    panel_reliability(ax_e, HERE / "reports" / "predictions.csv")
    panel_label(ax_e, "e", x=-0.20)

    ax_f = fig.add_subplot(gs[1, 2])
    panel_confidence_hist(ax_f, HERE / "reports" / "predictions.csv")
    panel_label(ax_f, "f", x=-0.20)

    save_fig(fig, "Figure6_Classifier", target_mm=COL_2 / MM)
    plt.close(fig)


# ------------------------------------------------------------------- figure 2
def figure_protocols(orig, group, ctrl):
    fig, axes = plt.subplots(1, 2, figsize=(COL_2, 2.75),
                             gridspec_kw=dict(width_ratios=[1.15, 1], wspace=0.3,
                                              left=0.085, right=0.985, top=0.86,
                                              bottom=0.19))
    ax = axes[0]
    keys, names = [], []
    for r, lab, c in [(orig, "Shipped\nsplit", GREY), (group, "Grouped CV\n(leave-pills-out)", BLUE),
                      (ctrl, "Grouped CV\nlabel blurred", RED)]:
        if r:
            keys.append(r); names.append(lab)
    metrics = [("accuracy", "Accuracy"), ("macro_f1", "Macro-F1"),
               ("balanced_accuracy", "Balanced acc."), ("macro_specificity", "Macro spec.")]
    x = np.arange(len(keys))
    w = 0.2
    colors = ["#2166ac", "#67a9cf", "#92c5de", "#d1e5f0"]
    for i, ((m, lab), col) in enumerate(zip(metrics, colors)):
        vals = [pooled(r)[m] for r in keys]
        err = None
        if m == "accuracy":
            e = []
            for r in keys:
                pm = pooled(r)
                if "cm" in pm:
                    lo, hi = wilson(np.trace(np.array(pm["cm"])), np.array(pm["cm"]).sum())
                    e.append([pm[m] - lo, hi - pm[m]])
                else:
                    e.append([0, 0])
            if any(v[1] > 0 for v in e):
                err = np.abs(np.clip(np.array(e).T, 0.0, None))
        ax.bar(x + (i - 1.5) * w, vals, w, label=lab, color=col, yerr=err,
               capsize=1.5, error_kw=dict(lw=0.7, ecolor="#444444"))
    ax.axhline(1 / 7, color="k", ls=":", lw=0.8)
    ax.text(len(keys) - 0.55, 1 / 7 + 0.018, "chance (1/7)", fontsize=6, ha="right")
    ax.set_xticks(x, names, fontsize=6.6)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Effect of evaluation protocol on reported performance", pad=4)
    ax.legend(frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.16),
              handlelength=1.4, columnspacing=1.2)
    panel_label(ax, "a", x=-0.10)

    ax2 = axes[1]
    ax2.set_title("Generalisation gap by antibiotic", pad=4)
    per_o = pooled(orig)["per_class"] if orig else None
    per_g = pooled(group)["per_class"]
    y = np.arange(7)
    if per_o:
        ax2.barh(y - 0.19, [per_o[c]["f1"] for c in CLASSES], 0.36, color=GREY,
                 label="Shipped split")
    ax2.barh(y + 0.19, [per_g[c]["f1"] for c in CLASSES], 0.36, color=BLUE,
             label="Grouped CV")
    ax2.set_yticks(y, CLASSES)
    ax2.invert_yaxis()
    ax2.set_xlim(0, 1.06)
    ax2.set_xlabel("F1 score")
    ax2.legend(frameon=False, loc="lower right")
    ax2.grid(axis="x", alpha=0.25, lw=0.5)
    ax2.set_axisbelow(True)
    panel_label(ax2, "b", x=-0.12)

    save_fig(fig, "FigureS_ClassifierProtocols", target_mm=COL_2 / MM)
    plt.close(fig)


def main():
    orig, group, ctrl = load("orig"), load("group"), load("control")
    if not group:
        print("grouped run missing - cannot build the figure yet")
        return
    figure6(orig, group, ctrl)
    if orig:
        figure_protocols(orig, group, ctrl)


if __name__ == "__main__":
    main()
