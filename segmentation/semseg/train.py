"""
semseg.train — 训练双通道语义分割网络

    python -m semseg.train
    python -m semseg.train --encoder resnet18 --epochs 60 --batch 4
    python -m semseg.train --cpu

选模依据：验证集 **宏平均 Dice**（两类 Dice 的平均），因为两类都要准。
产出（runs_semseg/<时间戳>_<encoder>/）：
    best.pt / last.pt     权重
    history.json          每轮损失与指标
    curves.png            训练曲线
    test_report.md        测试集逐类别报告
    test_vis/             测试集可视化
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
from torch.utils.data import DataLoader

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from semseg import config as C
    from semseg import metrics as M
    from semseg.dataset import SegDataset, _worker_init
    from semseg.model import build_model, count_parameters
    from semseg.losses import TotalLoss
else:
    from semseg import config as C
    from semseg import metrics as M
    from semseg.dataset import SegDataset, _worker_init
    from semseg.model import build_model, count_parameters
    from semseg.losses import TotalLoss


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@torch.no_grad()
def _infer(model, x, tta: bool):
    """tta=True 时做 5 视图测试时增强（旋转 0/90/180/270 + 水平翻转）取平均。"""
    def one(t):
        with torch.amp.autocast("cuda", enabled=(t.is_cuda)):
            return torch.sigmoid(model(t))
    outs = [one(x)]
    if tta:
        for k in (1, 2, 3):
            xr = torch.rot90(x, k, dims=(2, 3))
            outs.append(torch.rot90(one(xr.contiguous()), -k, dims=(2, 3)))
        xf = torch.flip(x, dims=(3,))
        outs.append(torch.flip(one(xf.contiguous()), dims=(3,)))
    return torch.stack(outs).mean(0)


@torch.no_grad()
def evaluate(model, loader, device, collect_vis: int = 0,
             vis_dir: Path | None = None, tta: bool = False) -> dict:
    model.eval()
    counts = {c: [0, 0, 0, 0] for c in range(C.NUM_CLASSES)}
    absent = {c: 0 for c in range(C.NUM_CLASSES)}
    n_vis = 0

    for batch in loader:
        x = batch["image"].to(device, non_blocking=True)
        prob = _infer(model, x, tta).float().cpu().numpy()
        gt = batch["semantic"].numpy()

        for i in range(prob.shape[0]):
            for c, (tp, fp, fn, tn) in enumerate(M.confusion_counts(prob[i], gt[i])):
                counts[c][0] += tp
                counts[c][1] += fp
                counts[c][2] += fn
                counts[c][3] += tn
                if (gt[i][c] > 0.5).sum() == 0:
                    absent[c] += 1

            if vis_dir is not None and n_vis < collect_vis:
                from semseg.viz import save_panel
                save_panel(batch["image"][i], prob[i], gt[i],
                           vis_dir / f"{batch['name'][i]}.jpg")
                n_vis += 1

    m = M.metrics_from_counts([tuple(counts[c]) for c in range(C.NUM_CLASSES)], absent)
    m["_macro_dice"] = M.macro_dice({k: v for k, v in m.items() if not k.startswith("_")})
    return m


def train(args) -> Path:
    set_seed(args.seed)
    device = torch.device("cuda" if (torch.cuda.is_available() and not args.cpu) else "cpu")
    run_dir = C.RUNS / f"{time.strftime('%Y%m%d_%H%M%S')}_{args.encoder}"
    (run_dir / "val_vis").mkdir(parents=True, exist_ok=True)

    print("=" * 74)
    print(f"网络      : SegNet 双通道语义分割 ({args.encoder})")
    print(f"设备      : {device}"
          + (f"  ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))
    print(f"输出      : {run_dir}")
    print("=" * 74)

    tr = SegDataset("train", use_aug=True, img_size=args.img_size)
    va = SegDataset("val", use_aug=False, img_size=args.img_size)
    te = SegDataset("test", use_aug=False, img_size=args.img_size)
    print(f"train {len(tr):4d}（{tr.source}）  val {len(va):4d}（{va.source}）  "
          f"test {len(te):4d}（{te.source}）")

    dl = dict(num_workers=args.workers, pin_memory=(device.type == "cuda"),
              worker_init_fn=_worker_init)
    tl = DataLoader(tr, batch_size=args.batch, shuffle=True, **dl)
    vl = DataLoader(va, batch_size=args.batch, shuffle=False, **dl)
    el = DataLoader(te, batch_size=args.batch, shuffle=False, **dl)

    model = build_model(args.encoder, pretrained=True).to(device)
    tot, trn = count_parameters(model)
    print(f"参数量    : {tot/1e6:.2f} M\n")

    crit = TotalLoss().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs,
                                                       eta_min=args.lr * 0.01)
    scaler = torch.amp.GradScaler("cuda", enabled=(args.amp and device.type == "cuda"))

    history: list[dict] = []
    best_dice, best_epoch, bad = -1.0, -1, 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        run_loss = 0.0
        n_step = len(tl)
        for step, batch in enumerate(tl, start=1):
            x = batch["image"].to(device, non_blocking=True)
            y = batch["semantic"].to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=(args.amp and device.type == "cuda")):
                logits = model(x)
                loss, _ = crit(logits, y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            run_loss += float(loss.detach()) * x.shape[0]
            if step % args.log_every == 0 or step == n_step:
                el_s = time.time() - t0
                print(f"  epoch {epoch} step {step:4d}/{n_step}  "
                      f"loss={run_loss/(step*args.batch):.4f}  "
                      f"{el_s:.0f}s ({el_s/step*1000:.0f} ms/step)", flush=True)
        sched.step()
        n = max(len(tr), 1)

        vis_n = 4 if (epoch % 25 == 0 or epoch == 1) else 0
        val = evaluate(model, vl, device, collect_vis=vis_n,
                       vis_dir=run_dir / "val_vis")
        md = val["_macro_dice"]

        history.append({"epoch": epoch, "loss": run_loss / n, "val": val,
                        "lr": float(opt.param_groups[0]["lr"]),
                        "sec": time.time() - t0})

        flag = ""
        if md > best_dice:
            best_dice, best_epoch, bad = md, epoch, 0
            torch.save({"model_state": model.state_dict(), "encoder": args.encoder,
                        "img_size": args.img_size, "epoch": epoch,
                        "val_macro_dice": md,
                        "class_names": C.CLASS_NAMES}, run_dir / "best.pt")
            flag = "  <- best"
        else:
            bad += 1

        print(f"epoch {epoch:3d}/{args.epochs}  loss={run_loss/n:.4f}  "
              f"Area Dice={val['Area']['Dice']:.4f}  Yaoping Dice={val['Yaoping']['Dice']:.4f}  "
              f"macro={md:.4f}  {time.time()-t0:.0f}s{flag}")

        if bad >= args.patience:
            print(f"\n早停：宏平均 Dice 连续 {args.patience} 轮未提升（最佳 epoch {best_epoch}）")
            break

    torch.save({"model_state": model.state_dict(), "encoder": args.encoder,
                "img_size": args.img_size, "epoch": epoch,
                "class_names": C.CLASS_NAMES}, run_dir / "last.pt")
    (run_dir / "history.json").write_text(
        json.dumps(history, indent=2, ensure_ascii=False), encoding="utf-8")

    try:
        from semseg.viz import plot_curves
        plot_curves(history, run_dir / "curves.png")
    except Exception as e:
        print(f"曲线绘制失败（不影响训练）：{e}")

    print("\n" + "=" * 74)
    print("最佳权重 / 测试集（原图，按物理平板划分）")
    print("=" * 74)
    ck = torch.load(run_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(ck["model_state"])
    test = evaluate(model, el, device, collect_vis=len(te),
                    vis_dir=run_dir / "test_vis")
    # 交付配置默认开 TTA，所以报告里也给出含 TTA 的数字
    test_tta = evaluate(model, el, device, tta=True) if C.USE_TTA else None

    (run_dir / "test_report.json").write_text(json.dumps(
        {"encoder": args.encoder, "img_size": args.img_size,
         "best_epoch": best_epoch, "best_val_macro_dice": best_dice,
         "n_train": len(tr), "n_val": len(va), "n_test": len(te),
         "thresh": C.THRESH, "tta": C.USE_TTA,
         "test": test, "test_with_tta": test_tta},
        indent=2, ensure_ascii=False), encoding="utf-8")

    L = ["# 双通道语义分割 — 测试集报告", "",
         f"- 网络：SegNet（{args.encoder}）@ {args.img_size}×{args.img_size}",
         f"- 最佳 epoch：{best_epoch}（验证宏平均 Dice = {best_dice:.4f}）",
         f"- 数据：train {len(tr)} / val {len(va)} / test {len(te)} 张（按物理平板划分）",
         f"- 分割阈值：{C.THRESH}；测试时增强（TTA）：{'开' if C.USE_TTA else '关'}",
         ""]

    def table(res, title):
        out = [f"## {title}", "",
               "| 类别 | Dice | IoU | Precision | Recall | 正像素数 | TP | FP | FN | 该类缺失的图数 |",
               "|---|---|---|---|---|---|---|---|---|---|"]
        for name in ("Area", "Yaoping"):
            v = res[name]
            out.append(f"| {name} | **{v['Dice']:.4f}** | {v['IoU']:.4f} | "
                       f"{v['Precision']:.4f} | {v['Recall']:.4f} | {v['n_pos_px']:,} | "
                       f"{v['n_tp']:,} | {v['n_fp']:,} | {v['n_fn']:,} | "
                       f"{v['n_images_absent']} |")
        out += ["", f"**宏平均 Dice = {res['_macro_dice']:.4f}**", ""]
        return out

    if test_tta is not None:
        L += table(test_tta, f"逐类别掩膜指标（含 TTA，交付配置）")
        L += table(test, "对照：不含 TTA")
        L += [f"TTA 带来的宏平均增益：**{test_tta['_macro_dice'] - test['_macro_dice']:+.4f}**", ""]
    else:
        L += table(test, "逐类别掩膜指标")

    L += [f"权重：`{run_dir / 'best.pt'}`", f"可视化：`{run_dir / 'test_vis'}`", ""]
    txt = "\n".join(L)
    (run_dir / "test_report.md").write_text(txt, encoding="utf-8")
    print("\n" + txt)
    return run_dir


def main() -> None:
    ap = argparse.ArgumentParser(description="训练双通道语义分割网络")
    ap.add_argument("--encoder", default=C.ENCODER, choices=["resnet18", "resnet34"])
    ap.add_argument("--epochs", type=int, default=C.EPOCHS)
    ap.add_argument("--batch", type=int, default=C.BATCH_SIZE)
    ap.add_argument("--lr", type=float, default=C.LR)
    ap.add_argument("--weight-decay", type=float, default=C.WEIGHT_DECAY)
    ap.add_argument("--img-size", type=int, default=C.IMG_SIZE)
    ap.add_argument("--patience", type=int, default=C.PATIENCE)
    ap.add_argument("--workers", type=int, default=C.NUM_WORKERS)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log-every", type=int, default=100)
    ap.add_argument("--no-amp", dest="amp", action="store_false", default=C.AMP)
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()
    train(args)


if __name__ == "__main__":
    main()
