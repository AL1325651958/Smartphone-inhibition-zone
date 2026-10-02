"""Train DiskNet (the new multi-scale model) on the augmented dataset.

Optimiser defaults to AdamW, which is what worked for the earlier CNN here;
use --opt sgd for a YOLO-style run.
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
from torch.utils.data import DataLoader, WeightedRandomSampler

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from disknet_model import DiskNet, count_params  # noqa: E402
from train_augmented import CLASSES, EMA, FolderSet, evaluate, metrics  # noqa: E402

CNN = HERE


@torch.no_grad()
def recalibrate_bn(model, loader, device, reset: bool = False, batches: int = 30):
    """Recompute BatchNorm running statistics on `loader` in training mode.

    With only ~25 batches per epoch, BN running stats lag badly (train accuracy
    100% while val loss rises). Refreshing them before each validation removes
    that train/eval mismatch.
    """
    momenta = []
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            momenta.append(m.momentum)
            if reset:
                m.reset_running_stats()
            m.momentum = None          # cumulative moving average
    model.train()
    for i, (x, _) in enumerate(loader):
        model(x.to(device))
        if i + 1 >= batches:
            break
    for m, mo in zip([m for m in model.modules() if isinstance(m, nn.BatchNorm2d)],
                     momenta):
        m.momentum = mo
    model.eval()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(CNN / "dataset_augmented"))
    ap.add_argument("--run-name", default="disknet_small")
    ap.add_argument("--variant", default="small", choices=["tiny", "small", "base"])
    ap.add_argument("--img-size", type=int, default=112)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--opt", default="adamw", choices=["adamw", "sgd"])
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--wd", type=float, default=5e-4)
    ap.add_argument("--momentum", type=float, default=0.9)
    ap.add_argument("--warmup", type=int, default=4)
    ap.add_argument("--ls", type=float, default=0.05)
    ap.add_argument("--drop", type=float, default=0.05)
    ap.add_argument("--head-drop", type=float, default=0.3)
    ap.add_argument("--ema", type=float, default=0.999)
    ap.add_argument("--balance", type=int, default=1)
    ap.add_argument("--online-aug", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--bn-recal", type=int, default=1,
                    help="refresh BatchNorm running stats before each validation")
    ap.add_argument("--bn-reset", type=int, default=0,
                    help="also reset BN stats before recalibrating")
    args = ap.parse_args()

    random.seed(args.seed); np.random.seed(args.seed)
    torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    root = Path(args.data) / "images"
    outdir = CNN / "runs" / args.run_name
    outdir.mkdir(parents=True, exist_ok=True)
    print(f"device {device} | DiskNet-{args.variant} | img {args.img_size} | "
          f"epochs {args.epochs} | opt {args.opt}")

    tr = FolderSet(root / "train", True, online_aug=bool(args.online_aug),
                   img_size=args.img_size)
    va = FolderSet(root / "val", False, online_aug=False, img_size=args.img_size)
    te = FolderSet(root / "test", False, online_aug=False, img_size=args.img_size)
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

    model = DiskNet(7, args.variant, drop=args.drop,
                    head_drop=args.head_drop).to(device)
    print(f"params: {count_params(model)/1e6:.2f}M")

    if args.opt == "adamw":
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                weight_decay=args.wd)
    else:
        decay, nodecay = [], []
        for n_, p_ in model.named_parameters():
            (nodecay if p_.ndim == 1 else decay).append(p_)
        opt = torch.optim.SGD(
            [{"params": decay, "weight_decay": args.wd},
             {"params": nodecay, "weight_decay": 0.0}],
            lr=args.lr, momentum=args.momentum, nesterov=True)

    nbs = len(tl)
    steps = nbs * args.epochs
    warm = nbs * args.warmup
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / max(warm, 1) if s < warm else
        0.5 * (1 + math.cos(math.pi * (s - warm) / max(steps - warm, 1))))
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    ema = EMA(model, args.ema)
    crit = nn.CrossEntropyLoss(label_smoothing=args.ls)

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
        if args.bn_recal:
            recalibrate_bn(model, tl, device, reset=args.bn_reset)
        cm, vloss = evaluate(model, vl, device)
        m = metrics(cm)
        model.load_state_dict(backup)
        m.update(epoch=ep, train_loss=tr_loss, train_acc=tr_acc, val_loss=vloss,
                 lr=opt.param_groups[0]["lr"], secs=time.time() - t0)
        hist.append({k: v for k, v in m.items() if k != "per_class"})
        if m["macro_f1"] >= best["macro_f1"]:
            best = {k: v for k, v in m.items() if k != "per_class"}
            best["cm"] = cm.tolist()
            # keep the selected weights so ensembling can use the best epoch
            best["state"] = {k: v.detach().cpu().clone()
                             for k, v in model.state_dict().items()}
        print(f"  ep {ep:3d}/{args.epochs} lr {m['lr']:.2e} trloss {tr_loss:.3f} "
              f"tracc {tr_acc:.3f} | vloss {vloss:.3f} vacc {m['accuracy']:.3f} "
              f"mF1 {m['macro_f1']:.3f} | {m['secs']:.0f}s", flush=True)
        if ep % 20 == 0:
            json.dump(hist, open(outdir / "history.json", "w"), indent=1)

    ema.copy_to(model)
    if args.bn_recal:
        recalibrate_bn(model, tl, device, reset=args.bn_reset)
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
    print(f"best val macro-F1: {best['macro_f1']:.4f} (ep {best['epoch']})")

    # the checkpoint is the epoch selected on validation macro-F1, not the last
    if "state" in best:
        model.load_state_dict(best["state"])
        if args.bn_recal:
            recalibrate_bn(model, tl, device, reset=args.bn_reset)
        cm_va, _ = evaluate(model, vl, device)
        cm_te, _ = evaluate(model, tel, device)
        mva, mte = metrics(cm_va), metrics(cm_te)
        print(f"selected-epoch model: val acc {mva['accuracy']:.4f} "
              f"mF1 {mva['macro_f1']:.4f} | TEST acc {mte['accuracy']:.4f} "
              f"mF1 {mte['macro_f1']:.4f}")
        best.pop("state", None)

    torch.save({"state": {k: v.cpu() for k, v in model.state_dict().items()},
                "metrics_test": mte, "variant": args.variant},
               outdir / "best.pt")
    json.dump(hist, open(outdir / "history.json", "w"), indent=1)
    json.dump({"test": mte, "val": mva, "args": vars(args),
               "params": count_params(model)},
              open(outdir / "results.json", "w"), indent=1)
    print(f"\nwrote {outdir}")


if __name__ == "__main__":
    main()
