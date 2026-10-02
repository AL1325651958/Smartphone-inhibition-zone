"""Build evaluation figures, tables and a summary from the saved run artifacts.

Reads  runs/<protocol>/results.json + history.json  (written by train.py) and
writes PNG figures, CSV tables and a markdown summary into reports/.
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

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
CLASSES = ["CRO", "DA", "E", "LEV", "LZD", "P", "VA"]
LABELS = {
    "CRO": "CRO\n(Ceftriaxone)", "DA": "DA\n(Clindamycin)", "E": "E\n(Erythromycin)",
    "LEV": "LEV\n(Levofloxacin)", "LZD": "LZD\n(Linezolid)", "P": "P\n(Penicillin)",
    "VA": "VA\n(Vancomycin)",
}
PROTO_TITLE = {
    "orig": "Dataset split (per-image, pills shared)",
    "group": "Grouped 3-fold CV (leave-pills-out)",
    "control": "Grouped CV, label blurred (texture control)",
    "group1": "Grouped hold-out (single fold)",
}
REPORTS = HERE / "reports"
# protocols that make up the paper's story; other run directories in runs/ are
# ablations and are deliberately excluded from the summary
PROTOCOLS = ["orig", "group", "control"]
plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 300, "font.size": 10,
    "axes.spines.top": False, "axes.spines.right": False,
    "font.family": "DejaVu Sans",
})


def load(proto):
    d = HERE / "runs" / proto
    res, hist = d / "results.json", d / "history.json"
    if not res.exists():
        return None
    r = json.loads(res.read_text(encoding="utf-8"))
    r["_history"] = json.loads(hist.read_text(encoding="utf-8")) if hist.exists() else []
    return r


def wilson(k, n, z=1.96):
    if n == 0:
        return 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def pooled_metrics(proto_res):
    """Prefer the pooled cross-validation estimate when available."""
    if "pooled" in proto_res:
        return proto_res["pooled"]
    return proto_res.get("orig_split")


def fig_confusion(cm, title, path, normalize=False):
    cm = np.asarray(cm, dtype=float)
    if normalize:
        cm = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(6.2, 5.4))
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=cm.max())
    ax.set_xticks(range(7), CLASSES)
    ax.set_yticks(range(7), CLASSES)
    ax.set_xlabel("Predicted antibiotic disk")
    ax.set_ylabel("True antibiotic disk")
    ax.set_title(title)
    fmt = "{:.2f}" if normalize else "{:d}"
    thresh = cm.max() * 0.55
    for i in range(7):
        for j in range(7):
            v = cm[i, j]
            ax.text(j, i, f"{v:.2f}" if normalize else f"{int(round(v)):d}",
                    ha="center", va="center", fontsize=8.5,
                    color="white" if v > thresh else "#1a1a1a")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03,
                 label="Fraction of true class" if normalize else "Number of images")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def fig_summary_panel(protos, path):
    have = {k: v for k, v in protos.items() if v}
    if not have:
        return
    keys = list(have)
    metrics = ["accuracy", "macro_f1", "balanced_accuracy", "macro_specificity"]
    nice = ["Accuracy", "Macro-F1", "Balanced accuracy", "Macro specificity"]
    x = np.arange(len(keys))
    w = 0.2
    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    colors = ["#2166ac", "#4393c3", "#92c5de", "#d1e5f0"]
    for i, (m, n, c) in enumerate(zip(metrics, nice, colors)):
        vals, errs = [], []
        for k in keys:
            pm = pooled_metrics(have[k])
            v = pm[m]
            vals.append(v)
            if m == "accuracy" and "pooled" in have[k]:
                cm = np.asarray(have[k]["pooled"]["cm"], dtype=float)
                lo, hi = wilson(np.trace(cm), cm.sum())
                errs.append([v - lo, hi - v])
            else:
                errs.append([0, 0])
        if m == "accuracy" and any(e[1] > 0 for e in errs):
            errs = np.array(errs).T
        else:
            errs = None
        ax.bar(x + (i - 1.5) * w, vals, w, label=n, color=c,
               yerr=errs, capsize=3, error_kw=dict(lw=1))
    ax.axhline(1 / 7, ls=":", c="crimson", lw=1.2)
    ax.text(len(keys) - 0.5 - 1.5 * w, 1 / 7 + 0.02, "chance 1/7", color="crimson",
            fontsize=8, ha="right")
    ax.set_xticks(x, [PROTO_TITLE.get(k, k).replace(" (", "\n(") for k in keys],
                  fontsize=8.5)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Disk-classification performance by evaluation protocol")
    ax.legend(fontsize=8, ncol=2, frameon=False)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def fig_training_curves(protos, path):
    have = {k: v for k, v in protos.items() if v and v["_history"]}
    if not have:
        return
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for k, v in have.items():
        h = v["_history"]
        if k in ("group", "control"):
            # list of per-fold histories -> average across folds
            L = min(len(f) for f in h)
            if L == 0:
                continue
            acc = np.mean([[f[e]["accuracy"] for e in range(L)] for f in h], 0)
            f1 = np.mean([[f[e]["macro_f1"] for e in range(L)] for f in h], 0)
            nf = len(h)
            lab = f"{k} (mean of {nf} folds)"
        else:
            if isinstance(h, list) and h and isinstance(h[0], list):
                h = h[0]
            acc = [e["accuracy"] for e in h]
            f1 = [e["macro_f1"] for e in h]
            lab = k
        ep = np.arange(1, len(acc) + 1)
        axes[0].plot(ep, acc, label=lab, lw=1.6)
        axes[1].plot(ep, f1, label=lab, lw=1.6)
    for ax, t in zip(axes, ["Validation accuracy", "Validation macro-F1"]):
        ax.axhline(1 / 7, ls=":", c="crimson", lw=1.1)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(t)
        ax.set_ylim(0, 1.02)
        ax.grid(alpha=0.25, lw=0.6)
    axes[0].legend(fontsize=8, frameon=False)
    fig.suptitle("Training dynamics (dotted line = chance)", y=1.03)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def fig_per_class(pm, proto_key, path):
    per = pm["per_class"]
    y = np.arange(7)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6),
                             gridspec_kw={"width_ratios": [1.25, 1]})
    for ax, (metric, ttl) in zip(axes, [("f1", "Per-class F1"), ("recall", "Per-class sensitivity (recall)")]):
        vals = np.array([per[c][metric] for c in CLASSES])
        ci = [wilson(round(per[c]["recall"] * per[c]["support"]), per[c]["support"])
              if metric == "recall" else (0, 0) for c in CLASSES]
        if metric == "recall":
            lo = np.minimum(np.array([ci[i][0] for i in range(7)]), vals)
            hi = np.maximum(np.array([ci[i][1] for i in range(7)]), vals)
            # clip to >= 0 and take abs to also kill -0.0, which matplotlib rejects
            err = np.abs(np.clip(np.vstack([vals - lo, hi - vals]), 0.0, None))
        else:
            err = None
        ax.barh(y, vals, color="#4393c3", height=0.62,
                xerr=err, capsize=3, error_kw=dict(lw=1))
        ax.set_yticks(y, [f"{c}  (n={per[c]['support']})" for c in CLASSES], fontsize=9)
        ax.invert_yaxis()
        ax.set_xlim(0, 1.05)
        ax.set_xlabel(ttl)
        for i, v in enumerate(vals):
            ax.text(min(v + 0.02, 0.97), i, f"{v:.3f}", va="center", fontsize=8)
    fig.suptitle(PROTO_TITLE.get(proto_key, proto_key), y=1.04, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def write_tables(protos, outdir):
    rows = []
    for k, v in protos.items():
        if not v:
            continue
        pm = pooled_metrics(v)
        cm = np.asarray(pm.get("cm", v.get("fold0", {}).get("cm", [])), dtype=float)
        n = pm["n"]
        if cm.size:
            lo, hi = wilson(np.trace(cm), cm.sum())
            ci = f"[{lo:.3f}, {hi:.3f}]"
        else:
            ci = ""
        rows.append(dict(protocol=k, evaluation=PROTO_TITLE.get(k, k), n=n,
                         accuracy=f"{pm['accuracy']:.4f}", acc_95CI=ci,
                         macro_f1=f"{pm['macro_f1']:.4f}",
                         balanced_accuracy=f"{pm['balanced_accuracy']:.4f}",
                         macro_specificity=f"{pm['macro_specificity']:.4f}"))
    p = outdir / "summary_metrics.csv"
    with open(p, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    for k, v in protos.items():
        if not v:
            continue
        pm = pooled_metrics(v)
        per = outdir / f"per_class_{k}.csv"
        with open(per, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["class", "antibiotic", "support", "precision", "recall",
                        "f1", "specificity"])
            full = {"CRO": "Ceftriaxone", "DA": "Clindamycin", "E": "Erythromycin",
                    "LEV": "Levofloxacin", "LZD": "Linezolid", "P": "Penicillin",
                    "VA": "Vancomycin"}
            for c in CLASSES:
                d = pm["per_class"][c]
                w.writerow([c, full[c], d["support"], f"{d['precision']:.4f}",
                            f"{d['recall']:.4f}", f"{d['f1']:.4f}",
                            f"{d['specificity']:.4f}"])
    return rows


def markdown_summary(protos, rows, outdir):
    L = ["# Antibiotic disk classification - CNN results", "",
         "Seven-class classification of single antibiotic disks cropped from "
         "disk-diffusion plates (CRO/DA/E/LEV/LZD/P/VA). Model: compact "
         "ResNet-style CNN trained from scratch on 160x160 grayscale crops "
         "(2.85 M parameters).", "",
         "## Metrics by evaluation protocol", "",
         "| Protocol | Evaluation | n | Accuracy | 95% CI | Macro-F1 | Balanced acc. | Macro spec. |",
         "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        L.append("| {protocol} | {evaluation} | {n} | {accuracy} | {acc_95CI} | "
                 "{macro_f1} | {balanced_accuracy} | {macro_specificity} |".format(**r))
    L += ["", "## Protocol definitions", "",
          "- **orig** - the dataset's own `train`/`val` folders. No augmented copy of a "
          "validation image appears in training (verified by MD5 and by augmentation "
          "index), but all 60 source pills are shared, so this estimates accuracy on "
          "new images of familiar disks and reproduces the 100% previously reported.",
          "- **group** - 3-fold cross-validation in which every source pill is confined "
          "to a single fold (leave-pills-out). This is the strict generalisation bound "
          "and is the number to quote for *unseen* disks.",
          "- **control** - the grouped folds re-run with a 9 px Gaussian blur that "
          "erases the printed drug code, isolating how much of the accuracy comes from "
          "reading the label.", ""]
    for k, v in protos.items():
        if not v:
            continue
        pm = pooled_metrics(v)
        L.append(f"![{k} confusion](confusion_{k}_norm.png)")
        L.append("")
    (outdir / "summary.md").write_text("\n".join(L), encoding="utf-8")


def main():
    REPORTS.mkdir(exist_ok=True)
    protos = {k: load(k) for k in PROTOCOLS}
    present = {k: v for k, v in protos.items() if v}
    print("loaded protocols:", list(present))

    for k, v in present.items():
        pm = pooled_metrics(v)
        if not pm:
            continue
        cm = pm.get("cm") or v.get("fold0", {}).get("cm")
        if cm:
            fig_confusion(cm, PROTO_TITLE.get(k, k), REPORTS / f"confusion_{k}.png")
            fig_confusion(cm, PROTO_TITLE.get(k, k), REPORTS / f"confusion_{k}_norm.png",
                          normalize=True)
        fig_per_class(pm, k, REPORTS / f"per_class_{k}.png")

    fig_summary_panel(present, REPORTS / "summary_panel.png")
    fig_training_curves(present, REPORTS / "training_curves.png")
    rows = write_tables(present, REPORTS)
    markdown_summary(present, rows, REPORTS)
    # drop figures left over from earlier protocol sets / ablations
    keep = set()
    for k in present:
        keep |= {f"confusion_{k}.png", f"confusion_{k}_norm.png", f"per_class_{k}.png"}
    keep |= {"summary_panel.png", "training_curves.png", "reliability.png",
             "error_gallery.png"}
    for p in REPORTS.glob("*.png"):
        if p.name not in keep:
            p.unlink()
    for r in rows:
        print(r)
    print(f"\nfigures + tables -> {REPORTS}")


if __name__ == "__main__":
    main()
