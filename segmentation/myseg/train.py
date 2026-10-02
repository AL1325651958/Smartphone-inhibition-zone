"""
myseg/train.py — 训练 ResNet + U-Net

    python -m myseg.train                        # 默认 resnet34
    python -m myseg.train --encoder resnet18     # 换 ResNet18
    python -m myseg.train --epochs 200 --batch 4
    python -m myseg.train --no-amp               # 关混合精度（排查 NaN 时用）

产出（runs_unet/<时间戳>_<encoder>/）
    best.pt  last.pt        权重（含模型结构参数，便于直接加载）
    history.json            每轮 loss / 指标
    curves.png              loss 与指标曲线
    test_report.json        测试集汇总
    test_report.txt         人可读的测试集报告
    val_vis/                验证集可视化
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from myseg import config as C
    from myseg import metrics as M
    from myseg.dataset import SegDataset
    from myseg.model import build_model, count_parameters
else:
    from myseg import config as C
    from myseg import metrics as M
    from myseg.dataset import SegDataset
    from myseg.model import build_model, count_parameters


# ────────────────────────────────────────────────
#  可复现性
# ────────────────────────────────────────────────
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ────────────────────────────────────────────────
#  损失
# ────────────────────────────────────────────────
class BCEDiceLoss(nn.Module):
    """
    BCEWithLogits（数值稳定）+ Soft Dice 的组合。
    小数据 + 前景占比低时，单独用 BCE 容易偏向背景，Dice 项能把前景拉回来。
    """

    def __init__(self, bce_weight: float = 0.5):
        super().__init__()
        self.bce_weight = bce_weight

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(logits, target)
        prob = torch.sigmoid(logits)
        dims = (0, 2, 3)
        inter = (prob * target).sum(dims)
        denom = prob.sum(dims) + target.sum(dims)
        dice = 1.0 - (2.0 * inter + 1e-6) / (denom + 1e-6)
        return self.bce_weight * bce + (1.0 - self.bce_weight) * dice.mean()


# ────────────────────────────────────────────────
#  每轮评估
# ────────────────────────────────────────────────
@torch.no_grad()
def evaluate(model, loader, device, max_vis: int = 0, vis_dir: Path | None = None,
             save_pred: bool = False) -> dict:
    model.eval()
    per_sample_px: list[dict] = []
    zone_errs: list[dict] = []
    n_pred_zones = n_gt_zones = n_unmatched_pred = n_unmatched_gt = 0
    saved = 0

    for batch in loader:
        x = batch["image"].to(device, non_blocking=True)
        y = batch["mask"].to(device, non_blocking=True)
        logits = model(x)
        prob = torch.sigmoid(logits).float().cpu().numpy()
        gt = y.float().cpu().numpy()

        for i in range(prob.shape[0]):
            r = M.evaluate_sample(prob[i], gt[i], C.IMG_SIZE)
            per_sample_px.append(r["pixel"])
            if "zone_error" in r and r["zone_error"].get("n", 0) > 0:
                zone_errs.append(r["zone_error"])
            n_pred_zones += r["n_pred_zones"]
            n_gt_zones += r["n_gt_zones"]
            n_unmatched_pred += r.get("n_unmatched_pred", 0)
            n_unmatched_gt += r.get("n_unmatched_gt", 0)

            if max_vis and vis_dir is not None and saved < max_vis:
                from myseg.visualize import save_overlay
                save_overlay(batch["image"][i], prob[i], gt[i],
                             vis_dir / f"{batch['name'][i]}.jpg")
                saved += 1

            if save_pred and vis_dir is not None:
                from myseg.visualize import save_overlay
                vis_dir.mkdir(parents=True, exist_ok=True)
                save_overlay(batch["image"][i], prob[i], gt[i],
                             vis_dir / f"{batch['name'][i]}.jpg")

    agg = M.aggregate_pixel_metrics(per_sample_px)

    # 汇总测量误差：把各样本的配对数展开后整体计算，避免小样本平均的偏倚
    if zone_errs:
        tot = int(sum(e["n"] for e in zone_errs))
        w = lambda k: float(sum(e[k] * e["n"] for e in zone_errs) / max(tot, 1))
        zone = {
            "n_total": tot,
            "MAE": w("MAE"), "RMSE": w("RMSE"), "bias": w("bias"),
            "LoA_low": w("LoA_low"), "LoA_high": w("LoA_high"),
            "pct_within_1mm": w("pct_within_1mm"),
            "pct_within_2mm": w("pct_within_2mm"),
            "n_plates_with_zones": len(zone_errs),
        }
    else:
        zone = {"n_total": 0}

    return {
        "pixel": agg,
        "zone": zone,
        "n_images": len(loader.dataset),
        "n_pred_zones": n_pred_zones,
        "n_gt_zones": n_gt_zones,
        "n_unmatched_pred": n_unmatched_pred,
        "n_unmatched_gt": n_unmatched_gt,
    }


def mean_dice(res: dict) -> float:
    return float(np.mean([v["Dice"] for v in res["pixel"].values()]))


def mean_iou(res: dict) -> float:
    return float(np.mean([v["IoU"] for v in res["pixel"].values()]))


# ────────────────────────────────────────────────
#  训练
# ────────────────────────────────────────────────
def train(args) -> Path:
    set_seed(args.seed)
    device = torch.device("cuda" if (torch.cuda.is_available() and not args.cpu) else "cpu")

    stamp = time.strftime("%Y%m%d_%H%M%S")
    run_dir = C.RUNS / f"{stamp}_{args.encoder}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "val_vis").mkdir(exist_ok=True)

    print("=" * 74)
    print(f"编码器      : {args.encoder}")
    print(f"设备        : {device}"
          + (f"  ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))
    print(f"输出目录    : {run_dir}")
    print("=" * 74)

    # ── 数据 ──
    train_ds = SegDataset("train", use_aug=True, img_size=args.img_size)
    val_ds = SegDataset("val", use_aug=True, img_size=args.img_size)
    test_ds = SegDataset("test", use_aug=False, img_size=args.img_size)   # 测试集不增强
    print(f"train : {len(train_ds):4d} 张  （来源 {train_ds.source}）")
    print(f"val   : {len(val_ds):4d} 张  （来源 {val_ds.source}）")
    print(f"test  : {len(test_ds):4d} 张  （来源 {test_ds.source}，原图）")

    dl_kw = dict(num_workers=args.workers, pin_memory=(device.type == "cuda"))
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                              drop_last=False, **dl_kw)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False, **dl_kw)
    test_loader = DataLoader(test_ds, batch_size=args.batch, shuffle=False, **dl_kw)
    print(f"\nbatch={args.batch}  train 每轮 {len(train_loader)} 步")

    # ── 模型 ──
    model = build_model(args.encoder, pretrained=True).to(device)
    tot, tr = count_parameters(model)
    print(f"参数量      : {tot/1e6:.2f} M（可训练 {tr/1e6:.2f} M）\n")

    criterion = BCEDiceLoss().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs,
                                                       eta_min=args.lr * 0.01)
    scaler = torch.amp.GradScaler("cuda", enabled=(args.amp and device.type == "cuda"))

    history: list[dict] = []
    best_dice = -1.0
    best_epoch = -1
    bad = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        run_loss = 0.0
        # 进度条：实时刷新，避免"看起来卡住"。
        # 输出重定向到文件时自动关闭 —— 否则 tqdm 的 \r 刷新会把日志撑到几十 MB。
        pbar = tqdm(train_loader, desc=f"epoch {epoch}/{args.epochs}",
                    leave=False, dynamic_ncols=True, mininterval=2.0,
                    disable=(not sys.stdout.isatty()))
        for batch in pbar:
            x = batch["image"].to(device, non_blocking=True)
            y = batch["mask"].to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=(args.amp and device.type == "cuda")):
                logits = model(x)
                loss = criterion(logits, y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            run_loss += float(loss.detach()) * x.shape[0]
            pbar.set_postfix(loss=f"{float(loss.detach()):.4f}")
        sched.step()
        train_loss = run_loss / max(len(train_ds), 1)

        val = evaluate(model, val_loader, device,
                       max_vis=(4 if epoch % 20 == 0 or epoch == 1 else 0),
                       vis_dir=run_dir / "val_vis")
        vd, vi = mean_dice(val), mean_iou(val)
        vmae = val["zone"].get("MAE", float("nan"))

        history.append({
            "epoch": epoch, "train_loss": train_loss,
            "val_dice": vd, "val_iou": vi,
            "val_zone_MAE": vmae,
            "val_pixel": val["pixel"],
            "lr": float(opt.param_groups[0]["lr"]),
            "sec": time.time() - t0,
        })

        flag = ""
        if vd > best_dice:
            best_dice, best_epoch, bad = vd, epoch, 0
            torch.save({
                "model_state": model.state_dict(),
                "encoder": args.encoder,
                "img_size": args.img_size,
                "epoch": epoch,
                "val_dice": vd,
                "val_iou": vi,
                "num_classes": C.NUM_CLASSES,
                "class_names": C.CLASS_NAMES,
            }, run_dir / "best.pt")
            flag = "  ← best"
        else:
            bad += 1

        print(f"epoch {epoch:3d}/{args.epochs}  loss={train_loss:.4f}  "
              f"val_dice={vd:.4f}  val_iou={vi:.4f}  "
              f"zone_MAE={vmae:.3f}mm  lr={opt.param_groups[0]['lr']:.2e}  "
              f"{time.time()-t0:.0f}s{flag}")

        if bad >= args.patience:
            print(f"\n早停：验证 Dice 连续 {args.patience} 轮未提升（最佳 epoch {best_epoch}）")
            break

    torch.save({
        "model_state": model.state_dict(),
        "encoder": args.encoder, "img_size": args.img_size,
        "epoch": epoch, "num_classes": C.NUM_CLASSES, "class_names": C.CLASS_NAMES,
    }, run_dir / "last.pt")
    (run_dir / "history.json").write_text(
        json.dumps(history, indent=2, ensure_ascii=False), encoding="utf-8")

    # ── 训练曲线 ──
    try:
        from myseg.visualize import plot_curves
        plot_curves(history, run_dir / "curves.png")
    except Exception as e:
        print(f"曲线绘制失败（不影响训练）：{e}")

    # ── 用最佳权重在测试集上评估 ──
    print("\n" + "=" * 74)
    print("用最佳权重评估测试集（原图，未增强，按平板划分，无泄露）")
    print("=" * 74)
    ckpt = torch.load(run_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    test = evaluate(model, test_loader, device,
                    vis_dir=run_dir / "test_vis", save_pred=True)

    report = {
        "encoder": args.encoder,
        "best_epoch": best_epoch,
        "best_val_dice": best_dice,
        "img_size": args.img_size,
        "n_train": len(train_ds), "n_val": len(val_ds), "n_test": len(test_ds),
        "test": test,
        "history_last": history[-1] if history else None,
    }
    (run_dir / "test_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = [
        f"编码器: {args.encoder}   最佳 epoch: {best_epoch}   最佳验证 Dice: {best_dice:.4f}",
        f"训练/验证/测试: {len(train_ds)} / {len(val_ds)} / {len(test_ds)}",
        "",
        "── 测试集 像素层指标（各类别）──",
    ]
    for name, v in test["pixel"].items():
        lines.append(f"  {name:10s} IoU={v['IoU']:.4f}  Dice={v['Dice']:.4f}  "
                     f"P={v['Precision']:.4f}  R={v['Recall']:.4f}")
    z = test["zone"]
    lines += [
        "",
        "── 测试集 抑菌圈直径误差（以 6 mm 药片内标标定）──",
        f"  配对数 n = {z.get('n_total', 0)}（涉及 {z.get('n_plates_with_zones', 0)} 块平板）",
    ]
    if z.get("n_total", 0):
        lines += [
            f"  MAE  = {z['MAE']:.3f} mm",
            f"  RMSE = {z['RMSE']:.3f} mm",
            f"  偏倚 = {z['bias']:+.3f} mm",
            f"  95% LoA = [{z['LoA_low']:.3f}, {z['LoA_high']:.3f}] mm",
            f"  ±1 mm 内 = {z['pct_within_1mm']:.1f}%",
            f"  ±2 mm 内 = {z['pct_within_2mm']:.1f}%",
        ]
    lines += [
        "",
        f"  预测圈数 {test['n_pred_zones']} / 人工圈数 {test['n_gt_zones']}"
        f"（未配上：预测 {test['n_unmatched_pred']}，人工 {test['n_unmatched_gt']}）",
        "",
        f"权重: {run_dir / 'best.pt'}",
        f"可视化: {run_dir / 'test_vis'}",
    ]
    txt = "\n".join(lines)
    (run_dir / "test_report.txt").write_text(txt, encoding="utf-8")
    print("\n" + txt)
    return run_dir


def main() -> None:
    ap = argparse.ArgumentParser(description="训练 ResNet + U-Net 分割模型")
    ap.add_argument("--encoder", default=C.ENCODER, choices=["resnet18", "resnet34"])
    ap.add_argument("--epochs", type=int, default=C.EPOCHS)
    ap.add_argument("--batch", type=int, default=C.BATCH_SIZE)
    ap.add_argument("--lr", type=float, default=C.LR)
    ap.add_argument("--weight-decay", type=float, default=C.WEIGHT_DECAY)
    ap.add_argument("--img-size", type=int, default=C.IMG_SIZE)
    ap.add_argument("--patience", type=int, default=C.PATIENCE)
    ap.add_argument("--workers", type=int, default=C.NUM_WORKERS)
    ap.add_argument("--seed", type=int, default=C.SPLIT_SEED)
    ap.add_argument("--no-amp", dest="amp", action="store_false", default=C.AMP)
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()
    train(args)


if __name__ == "__main__":
    main()
