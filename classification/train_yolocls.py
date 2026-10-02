"""Train the YOLO-style classifier with YOLO's recipe.

Recipe follows ultralytics' classification defaults:
  * SGD, momentum 0.9, nesterov, lr0 0.01 scaled by batch/64
  * cosine LR decay to lrf*lr0, 3-epoch warmup (bias lr, momentum ramp)
  * weight decay applied only to conv/linear weights (not BN or bias)
  * label smoothing 0.1, EMA of weights, gradient clipping
  * strong-but-mild augmentation (mild is required: the glyphs are only ~10 px)

Evaluates on batch-disjoint val and test folders.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from collections import Counter
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
from yolocls_model import YOLOCls, count_params  # noqa: E402

CNN = HERE
CLASSES = ["CRO", "DA", "E", "LEV", "LZD", "P", "VA"]
MEAN, STD = 0.5, 0.25


class FolderSet(Dataset):
    def __init__(self, root: Path, train: bool, img_size: int, online_aug: bool):
        self.items = []
        for ci, c in enumerate(CLASSES):
            for p in sorted((root / c).glob("*.png")):
                self.items.append((p, ci))
        self.train = train
        self.img_size = img_size
        self.tf = None
        if train and online_aug:
            self.tf = T.Compose([
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
        if self.tf is not None:
            img = self.tf(img)
        x = (T.functional.to_tensor(img) - MEAN) / STD
        return x.repeat(3, 1, 1), y

    def labels(self):
        return [y for _, y in self.items]


class EMA:
    def __init__(self, model, decay=0.9999):
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


def build_optimizer(model, lr, momentum, decay, warmup_bias_lr=0.1):
    """YOLO's parameter grouping: no weight decay on BN and bias."""
    g_decay, g_nodecay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim == 1:                      # BN weight/bias and conv bias
            g_nodecay.append(p)
        else:
            g_decay.append(p)
    return torch.optim.SGD(
        [{"params": g_decay, "weight_decay": decay},
         {"params": g_nodecay, "weight_decay": 0.0}],
        lr=lr, momentum=momentum, nesterov=True), g_nodecay


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(CNN / "dataset_augmented"))
    ap.add_argument("--run-name", default="yolocls_n")
    ap.add_argument("--scale", default="n", choices=["n", "s", "m"])
    ap.add_argument("--img-size", type=int, default=112)
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr0", type=float, default=0.01)
    ap.add_argument("--lrf", type=float, default=0.01)
    ap.add_argument("--momentum", type=float, default=0.9)
    ap.add_argument("--decay", type=float, default=5e-4)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--ls", type=float, default=0.1)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--ema", type=float, default=0.9999)
    ap.add_argument("--balance", type=int, default=1)
    ap.add_argument("--online-aug", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--no-resume", action="store_true")
    args = ap.parse_args()

    random.seed(args.seed); np.random.seed(args.seed)
    torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    root = Path(args.data) / "images"
    outdir = CNN / "runs" / args.run_name
    outdir.mkdir(parents=True, exist_ok=True)
    print(f"device {device} | scale {args.scale} | img {args.img_size} | "
          f"epochs {args.epochs} | batch {args.batch}")

    tr = FolderSet(root / "train", True, args.img_size, bool(args.online_aug))
    va = FolderSet(root / "val", False, args.img_size, False)
    te = FolderSet(root / "test", False, args.img_size, False)
    print(f"train {len(tr)}  val {len(va)}  test {len(te)}")
    print("train:", dict(Counter(CLASSES[y] for _, y in tr.items)))

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

    model = YOLOCls(7, args.scale, dropout=args.dropout).to(device)
    print(f"model params: {count_params(model)/1e6:.2f}M")

    lr = args.lr0 * args.batch / 64.0
    opt, nodecay = build_optimizer(model, lr, args.momentum, args.decay)
    nbs = len(tl)
    total = nbs * args.epochs
    warm = nbs * args.warmup
    # YOLO ramps the BN/bias group from a small initial value to the base lr
    bias_lr0 = args.lr0 * args.batch / 64.0 * 0.1

    def lr_at(it):
        if it < warm:
            return (it + 1) / max(warm, 1)
        p = (it - warm) / max(total - warm, 1)
        return args.lrf + (1 - args.lrf) * 0.5 * (1 + math.cos(math.pi * p))

    def bias_lr_at(it):
        if it < warm:
            return bias_lr0 + (lr - bias_lr0) * (it + 1) / max(warm, 1)
        return lr * lr_at(it)

    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    ema = EMA(model, args.ema)
    crit = nn.CrossEntropyLoss(label_smoothing=args.ls)

    hist, best = [], {"macro_f1": -1}
    it = 0
    for ep in range(1, args.epochs + 1):
        model.train()
        t0, run, seen, correct = time.time(), 0.0, 0, 0
        for x, y in tl:
            frac = lr_at(it)
            for g in opt.param_groups:
                g["lr"] = lr * frac if g["weight_decay"] > 0 else bias_lr_at(it)
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                out = model(x)
                loss = crit(out, y)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            scaler.step(opt); scaler.update()
            ema.update(model)
            run += loss.item() * y.numel(); seen += y.numel()
            correct += (out.argmax(1) == y).sum().item()
            it += 1
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
        print(f"  ep {ep:3d}/{args.epochs} lr {m['lr']:.2e} trloss {tr_loss:.3f} "
              f"tracc {tr_acc:.3f} | vloss {vloss:.3f} vacc {m['accuracy']:.3f} "
              f"mF1 {m['macro_f1']:.3f} | {m['secs']:.0f}s", flush=True)
        if ep % 25 == 0:
            json.dump(hist, open(outdir / "history.json", "w"), indent=1)

    ema.copy_to(model)
    cm_va, _ = evaluate(model, vl, device)
    cm_te, _ = evaluate(model, tel, device)
    mva, mte = metrics(cm_va), metrics(cm_te)
    print("\n===== final (EMA) =====")
    print(f"val : acc {mva['accuracy']:.4f} mF1 {mva['macro_f1']:.4f} "
          f"bal {mva['balanced_accuracy']:.4f} n={mva['n']}")
    print(f"TEST: acc {mte['accuracy']:.4f} mF1 {mte['macro_f1']:.4f} "
          f"bal {mte['balanced_accuracy']:.4f} n={mte['n']}")
    print(f"\n{'class':6} {'n':>4} {'recall':>7} {'prec':>7} {'F1':>7}")
    for c in CLASSES:
        v = mte["per_class"][c]
        print(f"{c:6} {v['support']:>4} {v['recall']:>7.3f} {v['precision']:>7.3f} "
              f"{v['f1']:>7.3f}")
    print(f"best val macro-F1 during training: {best['macro_f1']:.4f} "
          f"(epoch {best['epoch']})")

    torch.save({"state": {k: v.cpu() for k, v in model.state_dict().items()},
                "metrics_test": mte, "metrics_val": mva, "scale": args.scale},
               outdir / "best.pt")
    json.dump(hist, open(outdir / "history.json", "w"), indent=1)
    json.dump({"test": mte, "val": mva, "args": vars(args),
               "params": count_params(model)},
              open(outdir / "results.json", "w"), indent=1)
    print(f"\nwrote {outdir}")


if __name__ == "__main__":
    main()
