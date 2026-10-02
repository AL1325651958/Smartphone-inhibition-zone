"""Train the disk classifier on the augmented dataset (folder layout).

Reads:
  dataset_augmented/images/{train,val,test}/<class>/*.png   (112x112)
  dataset_augmented/manifest.csv                            (for provenance)

The split in that folder is already batch-disjoint (no plate spans two splits),
so the test set stays a genuine held-out-batch estimate.

Online augmentation is applied on top of the offline variants, and EMA weights
are used for evaluation.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms as T

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

CNN = HERE
CLASSES = ["CRO", "DA", "E", "LEV", "LZD", "P", "VA"]
FULL = {"CRO": "Ceftriaxone", "DA": "Clindamycin", "E": "Erythromycin",
        "LEV": "Levofloxacin", "LZD": "Linezolid", "P": "Penicillin",
        "VA": "Vancomycin"}
MEAN, STD = 0.5, 0.25


# ---------------------------------------------------------------- model
class SE(nn.Module):
    def __init__(self, c, r=8):
        super().__init__()
        self.fc1 = nn.Conv2d(c, max(4, c // r), 1)
        self.fc2 = nn.Conv2d(max(4, c // r), c, 1)

    def forward(self, x):
        s = F.adaptive_avg_pool2d(x, 1)
        return x * torch.sigmoid(self.fc2(F.silu(self.fc1(s))))


class BasicBlock(nn.Module):
    def __init__(self, cin, cout, stride=1, drop=0.0):
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False)
        self.n1 = nn.GroupNorm(min(8, cout), cout)
        self.conv2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False)
        self.n2 = nn.GroupNorm(min(8, cout), cout)
        self.se = SE(cout)
        self.drop = nn.Dropout2d(drop) if drop > 0 else nn.Identity()
        self.short = nn.Identity()
        if stride != 1 or cin != cout:
            self.short = nn.Sequential(
                nn.Conv2d(cin, cout, 1, stride, bias=False),
                nn.GroupNorm(min(8, cout), cout))

    def forward(self, x):
        y = F.silu(self.n1(self.conv1(x)))
        y = self.se(self.n2(self.conv2(y)))
        return F.silu(self.drop(y) + self.short(x))


class PillCNN(nn.Module):
    """Same architecture as the earlier runs, for comparability."""

    def __init__(self, ncls=7, widths=(32, 64, 128, 256), blocks=(2, 2, 2, 2),
                 stem=32, drop=0.15, head_drop=0.3):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, stem, 5, 2, 2, bias=False),
            nn.GroupNorm(min(8, stem), stem), nn.SiLU(inplace=True),
            nn.Conv2d(stem, stem, 3, 1, 1, bias=False),
            nn.GroupNorm(min(8, stem), stem), nn.SiLU(inplace=True),
            nn.MaxPool2d(3, 2, 1))
        layers, cin = [], stem
        for i, (w, b) in enumerate(zip(widths, blocks)):
            for j in range(b):
                stride = 2 if (j == 0 and i > 0) else 1
                layers.append(BasicBlock(cin, w, stride, drop if i >= 2 else 0.0))
                cin = w
        self.body = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                                  nn.Dropout(head_drop), nn.Linear(cin, ncls))
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out",
                                        nonlinearity="relu")
            elif isinstance(m, nn.Linear):
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.head(self.body(self.stem(x)))


# ---------------------------------------------------------------- data
class FolderSet(Dataset):
    def __init__(self, root: Path, train: bool, online_aug: bool = True,
                 img_size: int = 112):
        self.items = []
        for ci, c in enumerate(CLASSES):
            for p in sorted((root / c).glob("*.png")):
                self.items.append((p, ci))
        self.train = train
        self.img_size = img_size
        self.aug = None
        if train and online_aug:
            self.aug = T.Compose([
                T.RandomAffine(degrees=12, translate=(0.04, 0.04),
                               scale=(0.94, 1.06), fill=128),
                T.ColorJitter(brightness=0.12, contrast=0.12),
        ])

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        p, y = self.items[i]
        img = Image.open(p).convert("L")
        if img.size != (self.img_size, self.img_size):
            img = img.resize((self.img_size, self.img_size), Image.BICUBIC)
        if self.aug is not None:
            img = self.aug(img)
        x = (T.functional.to_tensor(img) - MEAN) / STD
        return x.repeat(3, 1, 1), y

    def labels(self):
        return [y for _, y in self.items]


class EMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {k: v.detach().clone().float()
                       for k, v in model.state_dict().items()
                       if v.dtype.is_floating_point}

    @torch.no_grad()
    def update(self, model):
        for k, v in model.state_dict().items():
            if k in self.shadow:
                self.shadow[k].mul_(self.decay).add_(v.detach().float(),
                                                     alpha=1 - self.decay)

    def copy_to(self, model):
        sd = model.state_dict()
        for k, v in self.shadow.items():
            sd[k].copy_(v.to(sd[k].dtype))


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    cm = np.zeros((7, 7), dtype=np.int64)
    loss_sum, n = 0.0, 0
    for x, y in loader:
        out = model(x.to(device))
        loss_sum += F.cross_entropy(out, y.to(device), reduction="sum").item()
        n += y.numel()
        for t, p in zip(y.numpy(), out.argmax(1).cpu().numpy()):
            cm[t, p] += 1
    return cm, loss_sum / max(n, 1)


def metrics(cm):
    n = cm.sum()
    per, f1s, recs = {}, [], []
    for i, c in enumerate(CLASSES):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        tn = n - tp - fp - fn
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        spec = tn / (tn + fp) if tn + fp else 0.0
        per[c] = dict(precision=prec, recall=rec, f1=f1, specificity=spec,
                      support=int(cm[i, :].sum()))
        f1s.append(f1); recs.append(rec)
    return dict(accuracy=float(np.trace(cm) / n), macro_f1=float(np.mean(f1s)),
                balanced_accuracy=float(np.mean(recs)), n=int(n), per_class=per,
                macro_specificity=float(np.mean([v["specificity"]
                                                 for v in per.values()])))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(CNN / "dataset_augmented"))
    ap.add_argument("--run-name", default="aug_run")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--wd", type=float, default=5e-4)
    ap.add_argument("--warmup", type=int, default=4)
    ap.add_argument("--drop", type=float, default=0.15)
    ap.add_argument("--head-drop", type=float, default=0.4)
    ap.add_argument("--ema", type=float, default=0.999)
    ap.add_argument("--img-size", type=int, default=112,
                    help="images are resized to this; also set --data accordingly")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--balance", type=int, default=1, help="class-balanced sampler")
    ap.add_argument("--online-aug", type=int, default=1)
    ap.add_argument("--workers", type=int, default=0)
    args = ap.parse_args()

    random.seed(args.seed); np.random.seed(args.seed)
    torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    root = Path(args.data) / "images"
    outdir = CNN / "runs" / args.run_name
    outdir.mkdir(parents=True, exist_ok=True)
    print(f"device {device} | data {root} | run {args.run_name}")

    tr = FolderSet(root / "train", True, online_aug=bool(args.online_aug),
                   img_size=args.img_size)
    va = FolderSet(root / "val", False, online_aug=False, img_size=args.img_size)
    te = FolderSet(root / "test", False, online_aug=False, img_size=args.img_size)
    print(f"train {len(tr)}  val {len(va)}  test {len(te)}")
    print(f"train class counts: "
          f"{dict(Counter(CLASSES[y] for _, y in tr.items))}")
    print(f"val   class counts: "
          f"{dict(Counter(CLASSES[y] for _, y in va.items))}")
    print(f"test  class counts: "
          f"{dict(Counter(CLASSES[y] for _, y in te.items))}")

    if args.balance:
        cnt = Counter(tr.labels())
        w = np.array([1.0 / cnt[y] for _, y in tr.items], dtype=np.float64)
        sampler = WeightedRandomSampler(torch.as_tensor(w), len(tr), True)
        tl = DataLoader(tr, batch_size=args.batch, sampler=sampler,
                        num_workers=args.workers, pin_memory=True, drop_last=True)
    else:
        tl = DataLoader(tr, batch_size=args.batch, shuffle=True,
                        num_workers=args.workers, pin_memory=True, drop_last=True)
    vl = DataLoader(va, batch_size=64, shuffle=False, num_workers=args.workers)
    tel = DataLoader(te, batch_size=64, shuffle=False, num_workers=args.workers)

    model = PillCNN(drop=args.drop, head_drop=args.head_drop).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    steps = len(tl) * args.epochs
    warm = len(tl) * args.warmup
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / max(warm, 1) if s < warm else
        0.5 * (1 + math.cos(math.pi * (s - warm) / max(steps - warm, 1))))
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    ema = EMA(model, args.ema)
    crit = nn.CrossEntropyLoss(label_smoothing=0.05)

    hist, best = [], {"macro_f1": -1}
    for ep in range(1, args.epochs + 1):
        model.train()
        t0, run, seen, correct = time.time(), 0.0, 0, 0
        for x, y in tl:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                out = model(x)
                loss = crit(out, y)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(opt); scaler.update(); sched.step()
            ema.update(model)
            run += loss.item() * y.numel(); seen += y.numel()
            correct += (out.argmax(1) == y).sum().item()
        tr_loss, tr_acc = run / seen, correct / seen

        backup = {k: v.detach().clone() for k, v in model.state_dict().items()}
        ema.copy_to(model)
        cm, vloss = evaluate(model, vl, device)
        m = metrics(cm)
        model.load_state_dict(backup)
        m.update(epoch=ep, train_loss=tr_loss, train_acc=tr_acc, val_loss=vloss,
                 lr=opt.param_groups[0]["lr"], secs=time.time() - t0)
        hist.append({k: v for k, v in m.items() if k != "per_class"})
        if m["macro_f1"] >= best["macro_f1"]:
            best = {k: v for k, v in m.items() if k != "per_class"}
            best["cm"] = cm.tolist()
            best["state"] = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        print(f"  ep {ep:3d}/{args.epochs} lr {m['lr']:.2e} trloss {tr_loss:.3f} "
              f"tracc {tr_acc:.3f} | vloss {vloss:.3f} vacc {m['accuracy']:.3f} "
              f"mF1 {m['macro_f1']:.3f} | {m['secs']:.0f}s", flush=True)

    # final: EMA weights, evaluate val and the held-out test batches
    ema.copy_to(model)
    cm_va, _ = evaluate(model, vl, device)
    cm_te, _ = evaluate(model, tel, device)
    mva, mte = metrics(cm_va), metrics(cm_te)

    print("\n===== final (EMA weights) =====")
    print(f"val : acc {mva['accuracy']:.4f} macro-F1 {mva['macro_f1']:.4f} "
          f"balanced {mva['balanced_accuracy']:.4f} n={mva['n']}")
    print(f"TEST (held-out batches): acc {mte['accuracy']:.4f} "
          f"macro-F1 {mte['macro_f1']:.4f} balanced {mte['balanced_accuracy']:.4f} "
          f"n={mte['n']}")
    print(f"\n{'class':6} {'n':>4} {'recall':>7} {'prec':>7} {'F1':>7} {'spec':>7}")
    for c in CLASSES:
        v = mte["per_class"][c]
        print(f"{c:6} {v['support']:>4} {v['recall']:>7.3f} {v['precision']:>7.3f} "
              f"{v['f1']:>7.3f} {v['specificity']:>7.3f}")
    print("\ntest confusion (rows true, cols pred):")
    print("      " + "".join(f"{c:>6}" for c in CLASSES))
    for i, c in enumerate(CLASSES):
        print(f"{c:5} " + "".join(f"{v:>6d}" for v in cm_te[i]))

    torch.save({"state": {k: v.cpu() for k, v in model.state_dict().items()},
                "metrics_test": mte, "metrics_val": mva,
                "cm_test": cm_te.tolist()}, outdir / "best.pt")
    json.dump(hist, open(outdir / "history.json", "w"), indent=1)
    json.dump({"test": mte, "val": mva, "args": vars(args)},
              open(outdir / "results.json", "w"), indent=1)
    print(f"\nwrote {outdir}")


if __name__ == "__main__":
    main()
