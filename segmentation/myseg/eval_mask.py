"""
myseg/eval_mask.py — 只看 mask 区域识别质量

不做直径测量、不做药片-抑菌圈配对，只回答一个问题：
**模型把 Area（抑菌圈）和 Yaoping（药片）这两类区域分割得准不准。**

    python -m myseg.eval_mask --weights runs_unet/<run>/best.pt
    python -m myseg.eval_mask --weights ... --split val          # 评验证集
    python -m myseg.eval_mask --weights ... --no-vis             # 不出可视化

产出（写在 export/mask_eval/<权重名>_<split>/）
    report.md          逐类别指标 + 混淆计数，可直接贴进稿子
    report.json        同上的机器可读版
    metrics.csv        每张图一行的指标，便于统计检验
    vis/<图名>.jpg     四联图：原图 | 人工 | 预测 | 差异图

指标说明
--------
逐类别（前景为正类）：
    Dice / IoU      区域重叠程度，论文里报的主指标
    Precision       预测的区域里有多少是真的
    Recall          真实的区域里有多少被找到
    F1              由像素级混淆计数推出
    Acc             整图像素准确率（背景占绝大多数，参考意义有限）
    n_pos_px        该类的正像素总数（分母，用来判断指标是否稳）
    n_images_absent 该图里本来就没有这个类别的图片数

差异图配色：
    绿 = 命中(TP)   红 = 误检(FP)   蓝 = 漏检(FN)
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from myseg import config as C
    from myseg import metrics as M
    from myseg.dataset import SegDataset
    from myseg.io_utils import imread, imwrite
    from myseg.model import build_model
else:
    from myseg import config as C
    from myseg import metrics as M
    from myseg.dataset import SegDataset
    from myseg.io_utils import imread, imwrite
    from myseg.model import build_model


# ────────────────────────────────────────────────
#  可视化：原图 | 人工 | 预测 | 差异
# ────────────────────────────────────────────────
COL_ZONE = (117, 158, 29)      # BGR 绿
COL_DISK = (165, 95, 24)       # BGR 蓝


def _denorm(x: np.ndarray) -> np.ndarray:
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    mean = np.asarray(C.IMAGENET_MEAN, dtype=np.float32).reshape(3, 1, 1)
    std = np.asarray(C.IMAGENET_STD, dtype=np.float32).reshape(3, 1, 1)
    img = np.clip((x * std + mean) * 255.0, 0, 255).astype(np.uint8)
    import cv2
    return cv2.cvtColor(np.transpose(img, (1, 2, 0)), cv2.COLOR_RGB2BGR)


def _outline(base: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """把 (K,H,W) 的 0/1 mask 描边到 base 上。"""
    import cv2
    vis = base.copy()
    for c, color in ((0, COL_ZONE), (1, COL_DISK)):
        if c >= mask.shape[0]:
            continue
        m = (mask[c] > 0.5).astype(np.uint8)
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if cnts:
            cv2.drawContours(vis, cnts, -1, color, 3)
    return vis


def _diff_map(base: np.ndarray, pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """差异图：绿=命中 红=误检 蓝=漏检。两类合并显示。"""
    import cv2
    p = (pred > C.MASK_THRESH).any(axis=0)
    g = (gt > 0.5).any(axis=0)
    tp = p & g
    fp = p & ~g
    fn = ~p & g
    dim = (base.astype(np.float32) * 0.45).astype(np.uint8)
    out = dim.copy()
    out[tp] = (60, 200, 60)
    out[fp] = (40, 40, 230)
    out[fn] = (230, 140, 30)
    hit, miss, extra = int(tp.sum()), int(fn.sum()), int(fp.sum())
    denom = hit + miss + extra
    iou = hit / denom if denom else 1.0
    for i, (txt, col) in enumerate([
        (f"TP {hit}px", (60, 200, 60)),
        (f"FP {extra}px", (40, 40, 230)),
        (f"FN {miss}px", (230, 140, 30)),
        (f"IoU {iou:.3f}", (255, 255, 255)),
    ]):
        cv2.putText(out, txt, (10, 26 + i * 24), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(out, txt, (10, 26 + i * 24), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, col, 1, cv2.LINE_AA)
    return out


def _panel(tiles: list[tuple[str, np.ndarray]], title: str) -> np.ndarray:
    import cv2
    h = max(t.shape[0] for _, t in tiles)
    norm = []
    for name, t in tiles:
        if t.shape[0] != h:
            s = h / t.shape[0]
            t = cv2.resize(t, (int(t.shape[1] * s), h))
        t = t.copy()
        cv2.rectangle(t, (0, 0), (t.shape[1], 40), (22, 22, 22), -1)
        cv2.putText(t, name, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                    (255, 255, 255), 2, cv2.LINE_AA)
        norm.append(t)
    row = np.hstack(norm)
    cv2.rectangle(row, (0, 0), (row.shape[1], 44), (10, 10, 10), -1)
    cv2.putText(row, title, (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.85,
                (0, 255, 255), 2, cv2.LINE_AA)
    return row


# ────────────────────────────────────────────────
#  评估
# ────────────────────────────────────────────────
@torch.no_grad()
def run(args) -> Path:
    from torch.utils.data import DataLoader

    device = torch.device("cuda" if (torch.cuda.is_available() and not args.cpu) else "cpu")
    weights = Path(args.weights)
    if not weights.is_file():
        raise SystemExit(f"找不到权重：{weights}")

    ckpt = torch.load(weights, map_location="cpu", weights_only=False)
    encoder = ckpt.get("encoder", C.ENCODER)
    img_size = ckpt.get("img_size", C.IMG_SIZE)
    model = build_model(encoder, pretrained=False).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    # 测试集一律用原图；val 默认也用原图（增强版指标会偏乐观，需显式指定）
    use_aug = args.use_aug
    ds = SegDataset(args.split, use_aug=use_aug, img_size=img_size)
    loader = DataLoader(ds, batch_size=args.batch, shuffle=False,
                        num_workers=args.workers,
                        pin_memory=(device.type == "cuda"))

    out_dir = C.EXPORT / "mask_eval" / f"{weights.stem}_{args.split}"
    vis_dir = out_dir / "vis"
    if not args.no_vis:
        vis_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 74)
    print(f"权重      : {weights}")
    print(f"编码器    : {encoder}   输入尺寸: {img_size}")
    print(f"权重来自  : epoch {ckpt.get('epoch')}（val Dice {ckpt.get('val_dice')}）")
    print(f"评估集    : {args.split}  {len(ds)} 张（来源 {ds.source}）")
    print(f"设备      : {device}")
    print(f"输出      : {out_dir}")
    print("=" * 74)

    # 逐类别累计混淆计数（跨图直接把 tp/fp/fn 相加，再算指标 —— 比逐图平均更稳）
    acc = {c: dict(tp=0, fp=0, fn=0, tn=0) for c in range(C.NUM_CLASSES)}
    per_image: list[dict] = []
    absent = {c: 0 for c in range(C.NUM_CLASSES)}
    n_vis = 0

    for batch in loader:
        x = batch["image"].to(device, non_blocking=True)
        prob = torch.sigmoid(model(x)).float().cpu().numpy()
        gt = batch["mask"].numpy()

        for i in range(prob.shape[0]):
            name = batch["name"][i]
            counts = M.confusion_counts(prob[i], gt[i])
            row = {"image": name, "plate": name.split("-", 1)[1].split("_")[0]}
            for c, (tp, fp, fn, tn) in enumerate(counts):
                for k, v in zip(("tp", "fp", "fn", "tn"), (tp, fp, fn, tn)):
                    acc[c][k] += v
                eps = 1e-9
                row[f"{C.CLASS_NAMES[c]}_IoU"] = round(tp / (tp + fp + fn + eps), 6)
                row[f"{C.CLASS_NAMES[c]}_Dice"] = round(2 * tp / (2 * tp + fp + fn + eps), 6)
                row[f"{C.CLASS_NAMES[c]}_P"] = round(tp / (tp + fp + eps), 6)
                row[f"{C.CLASS_NAMES[c]}_R"] = round(tp / (tp + fn + eps), 6)
                if (gt[i][c] > 0.5).sum() == 0:
                    absent[c] += 1
            per_image.append(row)

            if not args.no_vis:
                base = _denorm(batch["image"][i])
                tile = _panel([
                    ("original", base),
                    ("ground truth", _outline(base, gt[i])),
                    ("prediction", _outline(base, prob[i])),
                    ("difference", _diff_map(base, prob[i], gt[i])),
                ], name)
                imwrite(vis_dir / f"{name}.jpg", tile, quality=90)
                n_vis += 1

    # ── 汇总指标 ──
    summary = {}
    for c in range(C.NUM_CLASSES):
        t = acc[c]
        tp, fp, fn, tn = t["tp"], t["fp"], t["fn"], t["tn"]
        eps = 1e-9
        summary[C.CLASS_NAMES[c]] = {
            "IoU": tp / (tp + fp + fn + eps),
            "Dice": 2 * tp / (2 * tp + fp + fn + eps),
            "Precision": tp / (tp + fp + eps),
            "Recall": tp / (tp + fn + eps),
            "F1": 2 * tp / (2 * tp + fp + fn + eps),
            "Accuracy": (tp + tn) / (tp + tn + fp + fn + eps),
            "n_pos_px": tp + fn,
            "n_tp": tp, "n_fp": fp, "n_fn": fn,
            "n_images_absent": absent[c],
        }
    macro_iou = float(np.mean([summary[n]["IoU"] for n in summary]))
    macro_dice = float(np.mean([summary[n]["Dice"] for n in summary]))

    out_dir.mkdir(parents=True, exist_ok=True)

    # metrics.csv
    csv_path = out_dir / "metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as fp:
        w = csv.DictWriter(fp, fieldnames=list(per_image[0].keys()))
        w.writeheader()
        w.writerows(per_image)

    # report.md
    lines = [
        f"# Mask 区域识别评估 — {args.split}",
        "",
        f"- 权重：`{weights}`",
        f"- 编码器：{encoder}，输入尺寸 {img_size}×{img_size}",
        f"- 权重来自 epoch {ckpt.get('epoch')}（训练时验证 Dice {ckpt.get('val_dice')}）",
        f"- 评估图片：{len(ds)} 张（来源 `{ds.source}`，按物理平板划分，与训练集无重叠）",
        f"- 是否增强：{'是（指标偏乐观）' if use_aug else '否，全部为原图'}",
        "",
        "## 逐类别指标",
        "",
        "| 类别 | Dice | IoU | Precision | Recall | 正像素数 | TP | FP | FN | 该类缺失的图数 |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for n, v in summary.items():
        lines.append(
            f"| {n} | **{v['Dice']:.4f}** | {v['IoU']:.4f} | {v['Precision']:.4f} | "
            f"{v['Recall']:.4f} | {v['n_pos_px']:,} | {v['n_tp']:,} | {v['n_fp']:,} | "
            f"{v['n_fn']:,} | {v['n_images_absent']} |"
        )
    lines += [
        "",
        f"- **宏平均 Dice = {macro_dice:.4f}，宏平均 IoU = {macro_iou:.4f}**",
        "",
        "## 逐图指标分布",
        "",
        "| 类别 | Dice 最小 | 中位 | 最大 | Dice<0.5 的图数 |",
        "|---|---|---|---|---|",
    ]
    for c in range(C.NUM_CLASSES):
        n = C.CLASS_NAMES[c]
        vals = np.array([r[f"{n}_Dice"] for r in per_image])
        lines.append(f"| {n} | {vals.min():.4f} | {np.median(vals):.4f} | "
                     f"{vals.max():.4f} | {int((vals < 0.5).sum())}/{len(vals)} |")
    lines += [
        "",
        "## 说明",
        "",
        "- 指标由跨图累计的像素级混淆计数算出（不是逐图平均），更稳。",
        "- 差异图配色：绿色=命中(TP)，红色=误检(FP)，蓝色=漏检(FN)。",
        f"- 逐图 CSV：`{csv_path.name}`" + (f"，可视化 {n_vis} 张在 `vis/`。" if n_vis else "。"),
        "- 「该类缺失的图数」指人工标注里本来就没有这个类别的图片数，"
        "这些图会把该类 Recall 拉低或让指标失真，看指标时要一起看这一列。",
        "",
    ]
    report_md = "\n".join(lines)
    (out_dir / "report.md").write_text(report_md, encoding="utf-8")

    (out_dir / "report.json").write_text(json.dumps({
        "weights": str(weights), "encoder": encoder, "img_size": img_size,
        "checkpoint_epoch": ckpt.get("epoch"),
        "split": args.split, "n_images": len(ds), "use_aug": use_aug,
        "summary": summary, "macro_dice": macro_dice, "macro_iou": macro_iou,
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    print()
    print(report_md)
    return out_dir


def main() -> None:
    ap = argparse.ArgumentParser(description="只评估 mask 区域识别质量")
    ap.add_argument("--weights", type=Path, required=True, help="best.pt 路径")
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--no-vis", action="store_true", help="不出可视化图，快很多")
    ap.add_argument("--use-aug", action="store_true",
                    help="val 用增强集评估（指标会偏乐观，一般不要开）")
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()
