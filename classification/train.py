"""Train a from-scratch ResNet-style CNN to classify antibiotic disks (7 classes).

Dataset: `Antibacterial zone mask_Class/dataset` - grayscale crops of single
antibiotic disks from disk-diffusion (Kirby-Bauer) plates; the drug abbreviation
(CRO/DA/E/LEV/LZD/P/VA) is printed on the disk.

Three evaluation protocols are provided, because the shipped train/val folders
are a *within-pill augmentation split* (all 60 source pills appear on both
sides), which inflates accuracy:

  --protocol orig    : reproduce the shipped split (optimistic, reported for
                       comparability only)
  --protocol group   : 5-fold cross-validation grouped by source pill -> honest
  --protocol group1  : single grouped hold-out split (fast)
  --protocol control : grouped CV but images are heavily blurred, removing the
                       printed text; quantifies how much of the accuracy comes
                       from reading the label rather than from the agar texture

Example
-------
python train.py --protocol pilot
python train.py --protocol group --folds 5 --epochs 40 --img-size 160
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageFilter
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms as T

HERE = Path(__file__).resolve().parent
CLASSES = ["CRO", "DA", "E", "LEV", "LZD", "P", "VA"]
NCLS = len(CLASSES)
MEAN, STD = 0.5, 0.25


# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #
def erase_random_box(x: torch.Tensor, area=(0.02, 0.18), p: float = 1.0):
    """In-place cutout on a CHW tensor (simulates glare / agar artefacts)."""
    if random.random() > p:
        return x
    _, h, w = x.shape
    for _ in range(10):
        ta = random.uniform(*area) * h * w
        ar = math.exp(random.uniform(math.log(0.4), math.log(2.5)))
        bh = int(round(math.sqrt(ta * ar)))
        bw = int(round(math.sqrt(ta / ar)))
        if bh < h and bw < w:
            y0 = random.randint(0, h - bh)
            x0 = random.randint(0, w - bw)
            x[:, y0:y0 + bh, x0:x0 + bw] = float(torch.empty(1).normal_().clamp(-2.5, 2.5))
            break
    return x


class PillDataset(Dataset):
    """Loads manifest rows from an in-RAM cache of decoded grayscale images.

    `blur` destroys the printed label (texture control).  Images are cached once
    at `cache_px` and resized per epoch, which is far cheaper than re-decoding
    JPEGs (DataLoader workers are unavailable in this sandbox).
    """

    def __init__(self, rows, train: bool, img_size: int = 160, blur: float = 0.0,
                 use_aug: bool = True, cache_px: int = 192, cache: dict | None = None,
                 rot: float = 30.0):
        self.rows = rows
        self.img_size = img_size
        self.blur = blur
        self.train = train and use_aug
        self.cache_px = cache_px
        self.cache = cache if cache is not None else {}
        if self.train:
            self.aug = T.Compose([
                T.RandomResizedCrop(img_size, scale=(0.72, 1.0), ratio=(0.85, 1.18),
                                    interpolation=T.InterpolationMode.BICUBIC),
                T.RandomAffine(degrees=rot, translate=(0.06, 0.06),
                               scale=(0.92, 1.08), fill=0.5),
                T.RandomHorizontalFlip(0.5),
                T.RandomVerticalFlip(0.5),
                T.ColorJitter(brightness=0.30, contrast=0.30),
                T.RandomApply([T.GaussianBlur(3, sigma=(0.1, 1.2))], p=0.25),
            ])
        else:
            self.aug = None

    def _base(self, i) -> Image.Image:
        key = self.rows[i]["path"]
        arr = self.cache.get(key)
        if arr is None:
            im = Image.open(key).convert("L")
            if self.blur > 0:
                im = im.filter(ImageFilter.GaussianBlur(self.blur))
            # pad to square so affine/rotation never invents empty corners
            w, h = im.size
            n = max(w, h)
            if (w, h) != (n, n):
                sq = Image.new("L", (n, n), int(np.asarray(im).mean()))
                sq.paste(im, ((n - w) // 2, (n - h) // 2))
                im = sq
            im = im.resize((self.cache_px, self.cache_px), Image.BICUBIC)
            arr = np.asarray(im, dtype=np.uint8)
            self.cache[key] = arr
        return Image.fromarray(arr)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        img = self._base(i)
        if self.aug is not None:
            img = self.aug(img)
        if img.size != (self.img_size, self.img_size):
            img = img.resize((self.img_size, self.img_size), Image.BICUBIC)
        x = (T.functional.to_tensor(img) - MEAN) / STD
        x = x.repeat(3, 1, 1)
        if self.train:
            erase_random_box(x)
        return x, self.rows[i]["class_idx"]


class EvalDataset(Dataset):
    """Deterministic resize; `tta>1` averages centre/offset/flip views."""

    def __init__(self, rows, img_size=160, blur=0.0, tta: int = 1, cache_px: int = 192,
                 cache: dict | None = None):
        self.rows = rows
        self.img_size = img_size
        self.blur = blur
        self.tta = tta
        self.cache_px = cache_px
        self.cache = cache if cache is not None else {}

    def _base(self, i) -> Image.Image:
        key = self.rows[i]["path"]
        arr = self.cache.get(key)
        if arr is None:
            im = Image.open(key).convert("L")
            if self.blur > 0:
                im = im.filter(ImageFilter.GaussianBlur(self.blur))
            w, h = im.size
            n = max(w, h)
            if (w, h) != (n, n):
                sq = Image.new("L", (n, n), int(np.asarray(im).mean()))
                sq.paste(im, ((n - w) // 2, (n - h) // 2))
                im = sq
            im = im.resize((self.cache_px, self.cache_px), Image.BICUBIC)
            arr = np.asarray(im, dtype=np.uint8)
            self.cache[key] = arr
        return Image.fromarray(arr)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        img = self._base(i)
        s = self.img_size
        if self.tta <= 1:
            views = [img.resize((s, s), Image.BICUBIC)]
        else:
            # Rotation/scale jitter only: flips are out of distribution here
            # (the printed drug code is always upright), so flip-TTA averages
            # confident nonsense and drags accuracy towards chance.
            views = [img.resize((s, s), Image.BICUBIC)]
            big = img.resize((int(s * 1.10), int(s * 1.10)), Image.BICUBIC)
            views.append(big.crop((0, 0, s, s)))
            views.append(big.crop((big.width - s, big.height - s, big.width, big.height)))
            if self.tta >= 5:
                for ang in (-12, 12):
                    views.append(img.rotate(ang, resample=Image.BICUBIC, fillcolor=128)
                                 .resize((s, s), Image.BICUBIC))
            views = views[: max(self.tta, 1)]
        xs = [((T.functional.to_tensor(v) - MEAN) / STD).repeat(3, 1, 1) for v in views]
        return torch.stack(xs), self.rows[i]["class_idx"]


# --------------------------------------------------------------------------- #
# model
# --------------------------------------------------------------------------- #
class SE(nn.Module):
    def __init__(self, c, r=8):
        super().__init__()
        self.fc1 = nn.Conv2d(c, max(4, c // r), 1)
        self.fc2 = nn.Conv2d(max(4, c // r), c, 1)

    def forward(self, x):
        s = F.adaptive_avg_pool2d(x, 1)
        s = torch.sigmoid(self.fc2(F.silu(self.fc1(s))))
        return x * s


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
                nn.GroupNorm(min(8, cout), cout),
            )

    def forward(self, x):
        y = F.silu(self.n1(self.conv1(x)))
        y = self.n2(self.conv2(y))
        y = self.se(y)
        return F.silu(self.drop(y) + self.short(x))


class PillCNN(nn.Module):
    """Compact ResNet for 7-way disk classification on 128-160 px grayscale crops."""

    def __init__(self, ncls=NCLS, widths=(32, 64, 128, 256), blocks=(2, 2, 2, 2),
                 stem=32, drop=0.15, head_drop=0.3):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, stem, 5, 2, 2, bias=False),   # 160 -> 80
            nn.GroupNorm(min(8, stem), stem),
            nn.SiLU(inplace=True),
            nn.Conv2d(stem, stem, 3, 1, 1, bias=False),
            nn.GroupNorm(min(8, stem), stem),
            nn.SiLU(inplace=True),
            nn.MaxPool2d(3, 2, 1),                     # 80 -> 40
        )
        layers = []
        cin = stem
        for i, (w, b) in enumerate(zip(widths, blocks)):
            for j in range(b):
                stride = 2 if (j == 0 and i > 0) else 1
                layers.append(BasicBlock(cin, w, stride, drop if i >= 2 else 0.0))
                cin = w
        self.body = nn.Sequential(*layers)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Dropout(head_drop), nn.Linear(cin, ncls),
        )
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.Linear):
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.head(self.body(self.stem(x)))


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def set_seed(s: int):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def load_manifest(path: Path):
    with open(path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["class_idx"] = int(r["class_idx"])
        # manifests differ in naming: `build_manifest.py` writes base_key,
        # `make_split.py` writes core.  Normalise so either can be used.
        if "base_key" not in r and "core" in r:
            r["base_key"] = r["core"]
        if "orig_split" not in r:
            r["orig_split"] = r.get("split_plate", "")
    return rows


def group_folds(rows, k: int, seed: int = 0):
    """Assign folds so that every source pill sits in exactly one fold.

    Pills are distributed per class (round robin over a seeded shuffle) so the
    class balance is approximately equal across folds.
    """
    folds = {}
    for ci, c in enumerate(CLASSES):
        keys = sorted({r["base_key"] for r in rows if r["class"] == c})
        rng = random.Random(seed * 1000 + ci)
        rng.shuffle(keys)
        for i, key in enumerate(keys):
            folds[key] = i % k
    return folds


@torch.no_grad()
def evaluate(model, loader, device, ncls=NCLS):
    model.eval()
    cm = np.zeros((ncls, ncls), dtype=np.int64)
    loss_sum, n = 0.0, 0
    for x, y in loader:
        if x.dim() == 5:                      # TTA: (B, V, C, H, W)
            b, v = x.shape[:2]
            x = x.view(b * v, *x.shape[2:])
            logits = model(x.to(device, non_blocking=True))
            logits = logits.view(b, v, ncls).mean(1)
        else:
            logits = model(x.to(device, non_blocking=True))
        y = y.to(device)
        loss_sum += F.cross_entropy(logits, y, reduction="sum").item()
        n += y.numel()
        for t, p in zip(y.cpu().numpy(), logits.argmax(1).cpu().numpy()):
            cm[t, p] += 1
    return cm, loss_sum / max(n, 1)


def metrics_from_cm(cm):
    n = cm.sum()
    acc = np.trace(cm) / n
    per_cls = {}
    f1s, recs, precs = [], [], []
    for i, c in enumerate(CLASSES):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        tn = n - tp - fp - fn
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        spec = tn / (tn + fp) if tn + fp else 0.0
        per_cls[c] = dict(precision=prec, recall=rec, f1=f1, specificity=spec,
                          support=int(cm[i, :].sum()))
        f1s.append(f1); recs.append(rec); precs.append(prec)
    return dict(
        accuracy=float(acc),
        balanced_accuracy=float(np.mean(recs)),
        macro_precision=float(np.mean(precs)),
        macro_f1=float(np.mean(f1s)),
        macro_specificity=float(np.mean([v["specificity"] for v in per_cls.values()])),
        n=int(n),
        per_class=per_cls,
    )


def mixup_cutmix(x, y, ncls, alpha_mix=0.4, alpha_cut=1.0, prob=0.5, cut_prob=0.5):
    """Mixup / CutMix on a batch. Returns (x, soft target)."""
    if random.random() > prob:
        return x, F.one_hot(y, ncls).float()
    perm = torch.randperm(x.size(0), device=x.device)
    y2 = y[perm]
    if random.random() < cut_prob:                      # CutMix
        lam = float(np.random.beta(alpha_cut, alpha_cut))
        _, _, h, w = x.shape
        rh, rw = int(h * math.sqrt(1 - lam)), int(w * math.sqrt(1 - lam))
        if rh > 0 and rw > 0:
            cy, cx = random.randint(0, h - rh), random.randint(0, w - rw)
            x = x.clone()
            x[:, :, cy:cy + rh, cx:cx + rw] = x[perm][:, :, cy:cy + rh, cx:cx + rw]
        lam = 1 - (rh * rw) / (h * w)
    else:                                               # Mixup
        lam = float(np.random.beta(alpha_mix, alpha_mix))
        x = lam * x + (1 - lam) * x[perm]
    t = lam * F.one_hot(y, ncls).float() + (1 - lam) * F.one_hot(y2, ncls).float()
    return x, t


class EMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {k: v.detach().clone().float()
                       for k, v in model.state_dict().items() if v.dtype.is_floating_point}

    @torch.no_grad()
    def update(self, model):
        for k, v in model.state_dict().items():
            if k in self.shadow:
                self.shadow[k].mul_(self.decay).add_(v.detach().float(), alpha=1 - self.decay)

    def copy_to(self, model):
        sd = model.state_dict()
        for k, v in self.shadow.items():
            sd[k].copy_(v.to(sd[k].dtype))


def build_loader(rows, train, img_size, blur, batch, workers, aug=True, tta=1,
                 cache=None, rot=30.0):
    if train:
        ds = PillDataset(rows, train=True, img_size=img_size, blur=blur, use_aug=aug,
                         cache=cache, rot=rot)
        counts = collections.Counter(r["class_idx"] for r in rows)
        w = np.array([1.0 / counts[r["class_idx"]] for r in rows], dtype=np.float64)
        sampler = WeightedRandomSampler(torch.as_tensor(w), num_samples=len(rows),
                                        replacement=True)
        return DataLoader(ds, batch_size=batch, sampler=sampler, num_workers=workers,
                          pin_memory=True, drop_last=True)
    ds = EvalDataset(rows, img_size=img_size, blur=blur, tta=tta, cache=cache)
    return DataLoader(ds, batch_size=max(32, batch), shuffle=False, num_workers=workers,
                      pin_memory=True)


def prewarm_cache(rows, blur=0.0, cache_px=192):
    """Decode every image once into a shared RAM cache (also inherited by workers)."""
    cache = {}
    t0 = time.time()
    ds = EvalDataset(rows, img_size=8, blur=blur, tta=1, cache_px=cache_px, cache=cache)
    for i in range(len(ds)):
        ds._base(i)
    mb = sum(a.nbytes for a in cache.values()) / 1e6
    print(f"prewarmed cache: {len(cache)} images, {mb:.0f} MB, {time.time()-t0:.1f}s",
          flush=True)
    return cache


def train_one(tr_rows, va_rows, args, device, tag="", cache=None, ckpt_path=None):
    set_seed(args.seed)
    model = PillCNN(drop=args.drop, head_drop=args.head_drop).to(device)
    if args.channels_last:
        model = model.to(memory_format=torch.channels_last)
    nparam = sum(p.numel() for p in model.parameters())
    if cache is None:
        cache = prewarm_cache(tr_rows + va_rows, args.blur, args.cache_px)
    tl = build_loader(tr_rows, True, args.img_size, args.blur, args.batch, args.workers,
                      cache=cache, rot=args.rot)
    # validation during training uses 1 view to keep epochs cheap; TTA is applied
    # to the selected checkpoint at the end of the run
    vl = build_loader(va_rows, False, args.img_size, args.blur, max(32, args.batch),
                      args.workers, tta=1, cache=cache)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    steps = len(tl) * args.epochs
    warm = len(tl) * args.warmup
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / max(warm, 1) if s < warm else
        0.5 * (1 + math.cos(math.pi * (s - warm) / max(steps - warm, 1))))
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device.type == "cuda")
    ema = EMA(model, args.ema) if args.ema < 1 else None
    crit = nn.CrossEntropyLoss(label_smoothing=args.ls)

    hist = []
    best = {"macro_f1": -1.0}
    start_ep = 1
    state_path = ckpt_path.with_suffix(".state.pt") if ckpt_path else None

    # ---- resume an interrupted run (long jobs may be killed by the harness) --
    if state_path and state_path.exists() and not args.no_resume:
        try:
            st = torch.load(state_path, map_location=device, weights_only=False)
            if st.get("epochs") == args.epochs and st.get("tag") == tag:
                model.load_state_dict(st["model"])
                opt.load_state_dict(st["opt"])
                sched.load_state_dict(st["sched"])
                scaler.load_state_dict(st["scaler"])
                if ema and st.get("ema"):
                    ema.shadow = {k: v.to(device) for k, v in st["ema"].items()}
                hist = st["hist"]
                best = st["best"]
                start_ep = st["epoch"] + 1
                print(f"  [{tag}] resumed from epoch {st['epoch']} "
                      f"(best macro-F1 {best.get('macro_f1', -1):.4f})", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"  [{tag}] resume failed ({exc}); starting fresh", flush=True)

    for ep in range(start_ep, args.epochs + 1):
        model.train()
        t0 = time.time()
        run, seen, correct = 0.0, 0, 0
        for x, y in tl:
            x = x.to(device, non_blocking=True, memory_format=torch.channels_last) \
                if args.channels_last else x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            if args.mixup > 0:
                x, t = mixup_cutmix(x, y, NCLS, args.mixup, args.mixup, args.mix_prob)
            with torch.amp.autocast("cuda", enabled=args.amp and device.type == "cuda"):
                out = model(x)
                if args.mixup > 0:
                    loss = -(t * F.log_softmax(out, 1)).sum(1).mean()
                else:
                    loss = crit(out, y)
            train_mix = args.mixup > 0
            tt = t if train_mix else y
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            if ema:
                ema.update(model)
            run += loss.item() * y.numel()
            seen += y.numel()
            if train_mix:
                correct += (out.argmax(1) == tt.argmax(1)).sum().item()
            else:
                correct += (out.argmax(1) == tt).sum().item()
        tr_loss, tr_acc = run / seen, correct / seen

        if ema:
            backup = {k: v.detach().clone() for k, v in model.state_dict().items()}
            ema.copy_to(model)
        cm, va_loss = evaluate(model, vl, device)
        m = metrics_from_cm(cm)
        if ema:
            model.load_state_dict(backup)
        m.update(epoch=ep, train_loss=tr_loss, train_acc=tr_acc, val_loss=va_loss,
                 lr=opt.param_groups[0]["lr"], secs=time.time() - t0)
        hist.append({k: v for k, v in m.items() if k != "per_class"})
        if m["macro_f1"] >= best["macro_f1"]:
            best = {k: v for k, v in m.items() if k != "state"}
            best["cm"] = cm.tolist()
        print(f"  [{tag}] ep {ep:3d}/{args.epochs} lr {m['lr']:.2e} "
              f"trloss {tr_loss:.3f} tracc {tr_acc:.3f} | vloss {va_loss:.3f} "
              f"vacc {m['accuracy']:.3f} mF1 {m['macro_f1']:.3f} "
              f"bAcc {m['balanced_accuracy']:.3f} | {m['secs']:.0f}s", flush=True)

        if state_path:
            torch.save({"epoch": ep, "epochs": args.epochs, "tag": tag,
                        "model": model.state_dict(), "opt": opt.state_dict(),
                        "sched": sched.state_dict(), "scaler": scaler.state_dict(),
                        "ema": (ema.shadow if ema else None), "hist": hist,
                        "best": best}, state_path)

    # ---- final: the smoothed weights from the last epoch are the deliverable;
    # re-evaluate them with multi-view TTA ------------------------------------
    if args.tta > 1:
        vl_tta = build_loader(va_rows, False, args.img_size, args.blur,
                              max(32, args.batch), args.workers, tta=args.tta,
                              cache=cache)
        cm_tta, loss_tta = evaluate(model, vl_tta, device)
        m_tta = metrics_from_cm(cm_tta)
        print(f"  [{tag}] selected ep {best['epoch']}: val macro-F1 "
              f"{best['macro_f1']:.4f} (1 view) -> {m_tta['macro_f1']:.4f} "
              f"(TTA x{args.tta}), acc {m_tta['accuracy']:.4f}", flush=True)
        best = {k: v for k, v in m_tta.items() if k != "per_class"}
        best.update(epoch=int(hist[-1]["epoch"]) if hist else 0,
                    per_class=m_tta["per_class"], cm=cm_tta.tolist(), tta=args.tta)
    state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    if ckpt_path is not None:
        torch.save({"state": state, "metrics": {k: v for k, v in best.items()
                                                if k != "cm"}, "cm": best["cm"]},
                   ckpt_path)
    return dict(best=best, hist=hist, nparam=nparam, state=state)


# --------------------------------------------------------------------------- #
# protocols
# --------------------------------------------------------------------------- #
def run_protocol(args, rows, device):
    outdir = HERE / "runs" / (args.run_name or args.protocol)
    outdir.mkdir(parents=True, exist_ok=True)
    results = {}
    cache = prewarm_cache(rows, args.blur, args.cache_px)

    if args.protocol == "orig":
        tr = [r for r in rows if r["orig_split"] == "train"]
        va = [r for r in rows if r["orig_split"] == "val"]
        res = train_one(tr, va, args, device, tag="orig", cache=cache,
                        ckpt_path=outdir / "best.pt")
        results["orig_split"] = {k2: v for k2, v in res["best"].items()}
        results["nparam"] = res["nparam"]
        json.dump(res["hist"], open(outdir / "history.json", "w"), indent=1)

    elif args.protocol == "plate":
        # batch-disjoint split: whole acquisition plates are confined to one side,
        # so no disk (and no augmented twin) crosses a boundary
        if "split_plate" not in rows[0]:
            raise SystemExit(
                "manifest has no split_plate column; build it with\n"
                "  python make_split.py --test-plate <P> --val-plate <P> --emit\n"
                "then pass --manifest manifest_plate_split.csv")
        tr = [r for r in rows if r["split_plate"] == "train"]
        va = [r for r in rows if r["split_plate"] == "val"]
        te = [r for r in rows if r["split_plate"] == "test"]
        if not te:
            raise SystemExit("no test rows in the plate split")
        print(f"plate protocol: train {len(tr)} val {len(va)} test {len(te)}", flush=True)
        res = train_one(tr, va, args, device, tag="plate", cache=cache,
                        ckpt_path=outdir / "best.pt")
        # final held-out batch evaluation with the selected weights
        model = PillCNN(drop=args.drop, head_drop=args.head_drop).to(device)
        model.load_state_dict(res["state"])
        model.eval()
        tl = build_loader(te, False, args.img_size, args.blur, max(32, args.batch),
                          args.workers, tta=1, cache=cache)
        cm, te_loss = evaluate(model, tl, device)
        te_m = metrics_from_cm(cm)
        te_m["cm"] = cm.tolist()
        results["test"] = te_m
        results["val"] = {k2: v for k2, v in res["best"].items()}
        results["nparam"] = res["nparam"]
        json.dump(res["hist"], open(outdir / "history.json", "w"), indent=1)
        print("\n--- held-out acquisition batches ---")
        for c in CLASSES:
            v = te_m["per_class"][c]
            print(f"  {c:4} n={v['support']:5d} P={v['precision']:.3f} "
                  f"R={v['recall']:.3f} F1={v['f1']:.3f} Spec={v['specificity']:.3f}")
        print(f"  TEST accuracy {te_m['accuracy']:.4f} | macro-F1 {te_m['macro_f1']:.4f} "
              f"| balanced {te_m['balanced_accuracy']:.4f} "
              f"| macro spec {te_m['macro_specificity']:.4f}")

    elif args.protocol == "lopocv":
        # leave-one-plate-out cross-validation: every acquisition batch is held
        # out exactly once, so every image is tested by a model that has never
        # seen its plate.  This is the robust honest estimate; a single
        # hand-picked test plate is enough to be misled by plate difficulty.
        if "plate" not in rows[0]:
            raise SystemExit("manifest has no plate column; run audit_plate_split.py")
        plates = sorted({r["plate"] for r in rows})
        print(f"leave-one-plate-out over {len(plates)} plates", flush=True)
        results["plate_order"] = plates
        per_plate, cms, all_hist = [], [], []
        for pi, ph in enumerate(plates):
            te = [r for r in rows if r["plate"] == ph]
            rest = [r for r in rows if r["plate"] != ph]
            # inner validation: hold out a second plate so model selection is
            # also plate-disjoint (never validate on the test plate)
            val_plate = plates[(pi + 1) % len(plates)]
            va = [r for r in rest if r["plate"] == val_plate]
            tr = [r for r in rest if r["plate"] != val_plate]
            tag = f"plate{pi+1}/{len(plates)}"
            cache_f = outdir / f"plate{pi}_result.json"
            if cache_f.exists() and not args.no_resume:
                d = json.loads(cache_f.read_text(encoding="utf-8"))
                print(f"\n=== {tag} {ph}: cached (test acc {d['accuracy']:.4f}) "
                      f"===", flush=True)
                per_plate.append(d)
                cms.append(np.array(d["cm"]))
                continue
            print(f"\n=== {tag}: test plate {ph} ({len(te)} imgs), "
                  f"val plate {val_plate} ({len(va)}), train {len(tr)} ===", flush=True)
            res = train_one(tr, va, args, device, tag=tag, cache=cache,
                            ckpt_path=outdir / f"plate{pi}.pt")
            model = PillCNN(drop=args.drop, head_drop=args.head_drop).to(device)
            model.load_state_dict(res["state"])
            model.eval()
            tl = build_loader(te, False, args.img_size, args.blur, max(32, args.batch),
                              args.workers, tta=1, cache=cache)
            cm, _ = evaluate(model, tl, device)
            m = metrics_from_cm(cm)
            m["cm"] = cm.tolist()
            m["plate"] = ph
            m["n_te_disks"] = len({r["base_key"] for r in te})
            m["val_plate"] = val_plate
            m["val_accuracy"] = res["best"]["accuracy"]
            per_plate.append(m)
            cms.append(cm)
            cache_f.write_text(json.dumps(m, default=float), encoding="utf-8")
            (outdir / f"plate{pi}_history.json").write_text(
                json.dumps(res["hist"], default=float), encoding="utf-8")
            st = outdir / f"plate{pi}.state.pt"
            if st.exists():
                st.unlink()
            print(f"  [{tag}] TEST plate {ph}: acc {m['accuracy']:.4f} "
                  f"macro-F1 {m['macro_f1']:.4f}", flush=True)

        results["per_plate"] = per_plate
        cm_sum = np.sum(cms, axis=0)
        results["pooled"] = metrics_from_cm(cm_sum)
        results["pooled"]["cm"] = cm_sum.tolist()
        accs = [m["accuracy"] for m in per_plate]
        f1s = [m["macro_f1"] for m in per_plate]
        results["per_plate_mean_accuracy"] = float(np.mean(accs))
        results["per_plate_std_accuracy"] = float(np.std(accs))
        results["per_plate_mean_macro_f1"] = float(np.mean(f1s))
        results["per_plate_std_macro_f1"] = float(np.std(f1s))
        json.dump(all_hist if all_hist else [], open(outdir / "history.json", "w"),
                  indent=1)

    elif args.protocol in ("group", "group1", "control"):
        k = args.folds if args.protocol != "group1" else 1
        folds = group_folds(rows, max(k, 1), seed=args.seed) if k > 1 else \
            {r["base_key"]: (0 if r["split_group"] == "val" else 1) for r in rows}
        all_hist, cms = [], []
        for f in range(max(k, 1)):
            va = [r for r in rows if folds[r["base_key"]] == f]
            tr = [r for r in rows if folds[r["base_key"]] != f]
            tag = f"fold{f+1}/{max(k,1)}"
            # folds are cached individually so an interrupted cross-validation
            # resumes where it stopped instead of starting over
            cache_f = outdir / f"fold{f}_result.json"
            hist_f = outdir / f"fold{f}_history.json"
            if cache_f.exists() and not args.no_resume:
                d = json.loads(cache_f.read_text(encoding="utf-8"))
                print(f"\n=== {args.protocol} {tag}: cached "
                      f"(macro-F1 {d['macro_f1']:.4f}), skipping ===", flush=True)
                results[f"fold{f}"] = d
                cms.append(np.array(d["cm"]))
                all_hist.append(json.loads(hist_f.read_text(encoding="utf-8"))
                                if hist_f.exists() else [])
                continue
            print(f"\n=== {args.protocol} {tag}: train {len(tr)} val {len(va)} "
                  f"pills {len({r['base_key'] for r in va})} ===", flush=True)
            res = train_one(tr, va, args, device, tag=tag, cache=cache,
                            ckpt_path=outdir / f"fold{f}.pt")
            all_hist.append(res["hist"])
            cms.append(np.array(res["best"]["cm"]))
            fold_res = {k2: v for k2, v in res["best"].items() if k2 != "state"}
            results[f"fold{f}"] = fold_res
            cache_f.write_text(json.dumps(fold_res, default=float), encoding="utf-8")
            hist_f.write_text(json.dumps(res["hist"], default=float), encoding="utf-8")
            # the per-epoch state file is only needed while a fold is in flight
            st = (outdir / f"fold{f}.state.pt")
            if st.exists():
                st.unlink()
        cm_sum = np.sum(cms, axis=0)
        results["pooled"] = metrics_from_cm(cm_sum)
        results["pooled"]["cm"] = cm_sum.tolist()
        results["per_fold_mean_macro_f1"] = float(np.mean(
            [r["macro_f1"] for k2, r in results.items() if k2.startswith("fold")]))
        results["per_fold_std_macro_f1"] = float(np.std(
            [r["macro_f1"] for k2, r in results.items() if k2.startswith("fold")]))
        json.dump(all_hist, open(outdir / "history.json", "w"), indent=1)

    json.dump(results, open(outdir / "results.json", "w"), indent=1)
    print(f"\n===== {args.protocol} summary =====")
    if "pooled" in results:
        p = results["pooled"]
        print(f"pooled accuracy {p['accuracy']:.4f} | macro-F1 {p['macro_f1']:.4f} | "
              f"balanced acc {p['balanced_accuracy']:.4f} | macro spec {p['macro_specificity']:.4f}")
        print(f"per-fold macro-F1 {results['per_fold_mean_macro_f1']:.4f} "
              f"+- {results['per_fold_std_macro_f1']:.4f}")
        for c in CLASSES:
            v = p["per_class"][c]
            print(f"  {c:4} n={v['support']:4d} P={v['precision']:.3f} R={v['recall']:.3f} "
                  f"F1={v['f1']:.3f} Spec={v['specificity']:.3f}")
    elif "test" in results:
        t = results["test"]
        print(f"HELD-OUT BATCHES: accuracy {t['accuracy']:.4f} | "
              f"macro-F1 {t['macro_f1']:.4f} | balanced {t['balanced_accuracy']:.4f} | "
              f"macro spec {t['macro_specificity']:.4f} (n={t['n']})")
        for c in CLASSES:
            v = t["per_class"][c]
            print(f"  {c:4} n={v['support']:5d} P={v['precision']:.3f} R={v['recall']:.3f} "
                  f"F1={v['f1']:.3f} Spec={v['specificity']:.3f}")
        v = results["val"]
        print(f"validation split: accuracy {v['accuracy']:.4f} macro-F1 {v['macro_f1']:.4f}")
    elif "per_plate" in results:
        pp = results["per_plate"]
        print(f"\n{'plate':22} {'test_n':>7} {'disks':>6} {'acc':>7} {'macroF1':>8}")
        for m in pp:
            print(f"{m['plate']:22} {m['n']:>7} {m['n_te_disks']:>6} "
                  f"{m['accuracy']:>7.4f} {m['macro_f1']:>8.4f}")
        p = results["pooled"]
        print(f"\nPOOLED over all held-out plates: accuracy {p['accuracy']:.4f} | "
              f"macro-F1 {p['macro_f1']:.4f} | balanced {p['balanced_accuracy']:.4f} | "
              f"macro spec {p['macro_specificity']:.4f} (n={p['n']})")
        print(f"per-plate accuracy {results['per_plate_mean_accuracy']:.4f} "
              f"+- {results['per_plate_std_accuracy']:.4f} | "
              f"per-plate macro-F1 {results['per_plate_mean_macro_f1']:.4f} "
              f"+- {results['per_plate_std_macro_f1']:.4f}")
        for c in CLASSES:
            v = p["per_class"][c]
            print(f"  {c:4} n={v['support']:5d} P={v['precision']:.3f} R={v['recall']:.3f} "
                  f"F1={v['f1']:.3f} Spec={v['specificity']:.3f}")
    else:
        b = results["orig_split"]
        print(f"accuracy {b['accuracy']:.4f} macro-F1 {b['macro_f1']:.4f} "
              f"balanced acc {b['balanced_accuracy']:.4f}")
    print(f"artifacts -> {outdir}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--protocol", default="pilot",
                   choices=["pilot", "orig", "group", "group1", "control", "plate",
                            "lopocv"])
    p.add_argument("--manifest", default=str(HERE / "manifest.csv"))
    p.add_argument("--img-size", type=int, default=160)
    p.add_argument("--cache-px", type=int, default=0,
                   help="resolution held in RAM; 0 = auto (img-size + 32, min 192)")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--wd", type=float, default=1e-4)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--ls", type=float, default=0.05)
    p.add_argument("--drop", type=float, default=0.10)
    p.add_argument("--head-drop", type=float, default=0.30)
    p.add_argument("--ema", type=float, default=0.999)
    p.add_argument("--mixup", type=float, default=0.4,
                   help="alpha for mixup/cutmix; 0 disables")
    p.add_argument("--mix-prob", type=float, default=0.5)
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--workers", type=int, default=0,
                   help="0 loads batches in-process, which is ~3.5x faster here: with "
                        "workers>0 every PIL image is pickled through a pipe per epoch")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--blur", type=float, default=0.0)
    p.add_argument("--tta", type=int, default=5)
    p.add_argument("--amp", type=int, default=1)
    p.add_argument("--channels-last", type=int, default=1)
    p.add_argument("--limit", type=int, default=0, help="subsample rows for a smoke test")
    p.add_argument("--no-resume", action="store_true",
                   help="ignore an existing per-epoch state file and start fresh")
    p.add_argument("--run-name", default=None,
                   help="directory under runs/ to write to (defaults to --protocol); "
                        "use distinct names for comparison arms")
    p.add_argument("--rot", type=float, default=30.0,
                   help="max rotation in degrees for random affine augmentation")
    args = p.parse_args()

    if args.protocol == "control" and args.blur <= 0:
        args.blur = 9.0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device {device} | protocol {args.protocol} | img {args.img_size} | "
          f"epochs {args.epochs} | blur {args.blur}")

    rows = load_manifest(Path(args.manifest))
    if args.limit:
        keep = collections.Counter()
        rows = [r for r in rows if (keep.update([r["base_key"]]) or True)][: args.limit]
    if args.cache_px <= 0:
        args.cache_px = max(192, args.img_size + 32)
    print(f"manifest rows {len(rows)} | pills {len({r['base_key'] for r in rows})}")

    if args.protocol == "pilot":
        tr = [r for r in rows if r["split_group"] == "train"]
        va = [r for r in rows if r["split_group"] == "val"]
        print(f"pilot: train {len(tr)} val {len(va)}")
        outdir = HERE / "runs" / "pilot"
        outdir.mkdir(parents=True, exist_ok=True)
        res = train_one(tr, va, args, device, tag="pilot", ckpt_path=outdir / "best.pt")
        json.dump(res["hist"], open(outdir / "history.json", "w"), indent=1)
        b = res["best"]
        print(f"\npilot best: epoch {b['epoch']} acc {b['accuracy']:.4f} "
              f"macro-F1 {b['macro_f1']:.4f} balanced {b['balanced_accuracy']:.4f} "
              f"params {res['nparam']/1e6:.2f}M")
        return
    run_protocol(args, rows, device)


if __name__ == "__main__":
    main()
