"""Report accuracy and dump the misclassified images.

Each image is scored by the fold model that never saw its acquisition batch, so
every prediction here is out-of-batch (no leakage).
"""
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

HERE = Path(r"F:\Code\SCI\Nanjing_Children_Hospital\抑菌区域分析_返修\药片CNN")
sys.path.insert(0, str(HERE))
from train import CLASSES, PillCNN, build_loader, load_manifest  # noqa: E402

OUT = HERE / "error_report"
OUT.mkdir(exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

rows = load_manifest(HERE / "manifest_plate.csv")
plates = sorted({r["plate"] for r in rows})
plate_to_fold = {p: i for i, p in enumerate(plates)}

# ---------------------------------------------------------------- accuracy
all_true, all_pred, all_conf, all_plate = [], [], [], []
per_plate = defaultdict(lambda: [0, 0])

for pi, p in enumerate(plates):
    ck = HERE / "runs" / "lopo_main" / f"plate{pi}.pt"
    if not ck.exists():
        print(f"missing {ck}")
        continue
    model = PillCNN().to(device)
    sd = torch.load(ck, map_location=device, weights_only=False)
    model.load_state_dict(sd["state"])
    model.eval()
    sel = [r for r in rows if r["plate"] == p]
    loader = build_loader(sel, False, 160, 0.0, 64, 0, tta=1)
    P, Y = [], []
    with torch.no_grad():
        for x, y in loader:
            if x.dim() == 5:
                b, v = x.shape[:2]
                x = x.reshape(b * v, *x.shape[2:])
            P.append(torch.softmax(model(x.to(device)).float(), 1).cpu().numpy())
            Y.append(y.numpy())
    P, Y = np.concatenate(P), np.concatenate(Y)
    pred = P.argmax(1)
    conf = P.max(1)
    n_ok = int((pred == Y).sum())
    per_plate[p] = [n_ok, len(Y)]
    all_true.append(Y)
    all_pred.append(pred)
    all_conf.append(conf)
    all_plate += [p] * len(Y)

Y = np.concatenate(all_true)
PR = np.concatenate(all_pred)
CF = np.concatenate(all_conf)
PL = np.array(all_plate)
ok = PR == Y

print("=" * 62)
print(f"OVERALL (leave-one-batch-out, {len(Y)} images)")
print(f"  accuracy = {ok.mean():.4f}   ({ok.sum()}/{len(Y)})")
print("=" * 62)
print(f"\n{'batch':22} {'n':>5} {'correct':>8} {'accuracy':>9}")
for p in sorted(plates, key=lambda p: per_plate[p][0] / max(per_plate[p][1], 1)):
    c, n = per_plate[p]
    print(f"{p:22} {n:>5} {c:>8} {c/n:>9.4f}")

print("\nper-class recall / precision:")
print(f"{'class':6} {'n':>5} {'recall':>7} {'prec':>7}")
for i, c in enumerate(CLASSES):
    tp = ((PR == i) & (Y == i)).sum()
    rec = tp / max((Y == i).sum(), 1)
    pre = tp / max((PR == i).sum(), 1)
    print(f"{c:6} {(Y==i).sum():>5} {rec:>7.3f} {pre:>7.3f}")

cm = np.zeros((7, 7), int)
for t, p_ in zip(Y, PR):
    cm[t, p_] += 1
print("\nconfusion (rows true, cols predicted):")
print("      " + "".join(f"{c:>7}" for c in CLASSES))
for i, c in enumerate(CLASSES):
    print(f"{c:5} " + "".join(f"{v:>7d}" for v in cm[i]))

wrong = np.where(~ok)[0]
print(f"\nmisclassified: {len(wrong)} images "
      f"(from {len(set(PL[wrong]))} batches)")
print("worst confusions:")
pairs = Counter((CLASSES[Y[i]], CLASSES[PR[i]]) for i in wrong)
for (a, b), n in pairs.most_common(8):
    print(f"  {a:4} -> {b:4} {n:5d}")

# ---------------------------------------------------------------- CSV
with open(OUT / "predictions_lopo.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["path", "plate", "class", "predicted", "confidence", "correct"])
    for i in wrong:
        w.writerow([rows[i]["path"], PL[i], CLASSES[Y[i]], CLASSES[PR[i]],
                    f"{CF[i]:.4f}", 0])

# ---------------------------------------------------------------- galleries
def gallery(idx, path, title, cols=8, cell=150):
    if len(idx) == 0:
        return
    idx = idx[:64]
    rowsn = int(np.ceil(len(idx) / cols))
    canvas = Image.new("RGB", (cols * cell, rowsn * (cell + 26)), "white")
    d = ImageDraw.Draw(canvas)
    for k, i in enumerate(idx):
        im = Image.open(rows[i]["path"]).convert("L").resize((cell, cell))
        r, c = divmod(k, cols)
        canvas.paste(im.convert("RGB"), (c * cell, r * (cell + 26)))
        d.text((c * cell + 3, r * (cell + 26) + cell + 2),
               f"T:{CLASSES[Y[i]]} P:{CLASSES[PR[i]]} {CF[i]:.2f}", fill=(180, 0, 0))
        d.text((c * cell + 3, r * (cell + 26) + cell + 13), PL[i][-4:], fill=(80, 80, 80))
    canvas.save(OUT / path)
    print(f"wrote {OUT / path}  ({len(idx)} images, {title})")


# most confident mistakes overall
order = wrong[np.argsort(-CF[wrong])]
gallery(order, "errors_all_confident.png", "most confident errors (all batches)")

# worst batches
for p in ["20260505_063258", "20260505_064101", "20260505_063229"]:
    sub = wrong[PL[wrong] == p]
    sub = sub[np.argsort(-CF[sub])]
    gallery(sub, f"errors_{p[-4:]}.png", f"errors on batch {p}")

# one sheet per confused pair on the worst batch
worst = "20260505_063258"
for a, b in [("VA", "LZD"), ("P", "LEV"), ("DA", "CRO")]:
    ia, ib = CLASSES.index(a), CLASSES.index(b)
    sel = [i for i in np.where(PL == worst)[0] if Y[i] == ia and PR[i] == ib]
    if sel:
        sel = np.array(sel)[np.argsort(-CF[sel])]
        gallery(sel, f"pair_{worst[-4:]}_{a}_to_{b}.png", f"{a} -> {b} on {worst}")

print(f"\nall error sheets -> {OUT}")
