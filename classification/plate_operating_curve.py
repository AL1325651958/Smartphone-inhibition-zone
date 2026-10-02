"""Accuracy vs coverage (confidence gating) on the held-out batch.

The model is over-confident but its confidence is still informative, so this
produces the practical operating curve for a deployed classifier: accept the
high-confidence calls automatically, refer the rest to a human.
"""
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(r"F:\Code\SCI\Nanjing_Children_Hospital\抑菌区域分析_返修\药片CNN")
sys.path.insert(0, str(HERE))
from train import CLASSES, PillCNN, build_loader, load_manifest, prewarm_cache  # noqa: E402

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
rows = load_manifest(HERE / "manifest_plate_split.csv")
ck = torch.load(HERE / "runs" / "plate_main" / "best.pt", map_location=device,
                weights_only=False)
model = PillCNN().to(device)
model.load_state_dict(ck["state"])
model.eval()


def infer(sel):
    cache = prewarm_cache(sel, 0.0, 192)
    loader = build_loader(sel, False, 160, 0.0, 64, 0, tta=1, cache=cache)
    P, Y = [], []
    with torch.no_grad():
        for x, y in loader:
            if x.dim() == 5:
                b, v = x.shape[:2]
                x = x.reshape(b * v, *x.shape[2:])
            P.append(torch.softmax(model(x.to(device)).float(), 1).cpu().numpy())
            Y.append(y.numpy())
    return np.concatenate(P), np.concatenate(Y)


def ece(conf, correct, nbin=10):
    edges = np.linspace(1 / 7, 1.0, nbin + 1)
    e = 0.0
    for i in range(nbin):
        m = (conf >= edges[i]) & (conf < edges[i + 1] if i < nbin - 1 else conf <= 1)
        if m.sum():
            e += m.sum() / len(conf) * abs(correct[m].mean() - conf[m].mean())
    return e


for split in ("val", "test"):
    sel = [r for r in rows if r["split_plate"] == split]
    if not sel:
        continue
    P, Y = infer(sel)
    pred = P.argmax(1)
    conf = P.max(1)
    corr = pred == Y
    print(f"\n===== {split} plate ({len(sel)} images) =====")
    print(f"accuracy {corr.mean():.4f}  ECE {ece(conf, corr):.4f}  "
          f"mean confidence {conf.mean():.4f}")

    print(f"\n{'threshold':>9} {'coverage':>9} {'accepted':>9} {'accuracy':>9}")
    for t in [0.0, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]:
        m = conf >= t
        if m.sum() == 0:
            print(f"{t:>9.2f} {0.0:>9.1%} {0:>9d} {'-':>9}")
            continue
        print(f"{t:>9.2f} {m.mean():>9.1%} {int(m.sum()):>9d} {corr[m].mean():>9.4f}")

    # precision of the confident subset per class
    print(f"\nper-class recall at confidence >= 0.9 (accepted subset):")
    m = conf >= 0.9
    for i, c in enumerate(CLASSES):
        tot = (Y == i).sum()
        got = ((Y == i) & m & (pred == i)).sum()
        acc_n = ((Y == i) & m).sum()
        print(f"  {c:4} accepted {acc_n:5d}/{tot:5d}  correct {got:5d}  "
              f"recall {got / max(tot, 1):.3f}")
