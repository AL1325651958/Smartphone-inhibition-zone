"""
myseg/predict.py — 用训练好的权重对新图片推理

    python -m myseg.predict --source images --weights runs_unet/<run>/best.pt
    python -m myseg.predict --source data/test/images --weights ... --save-csv pred.csv

输出
----
export/predict/<图片名>_overlay.jpg   三联对照图（原图 / 预测轮廓 / 直径标注）
export/predict/measurements.csv       每张图每个药片的直径读数（mm）
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from myseg import config as C
    from myseg import instance as I
    from myseg import metrics as M
    from myseg.model import build_model
    from myseg.io_utils import imread
else:
    from myseg import config as C
    from myseg import instance as I
    from myseg import metrics as M
    from myseg.model import build_model
    from myseg.io_utils import imread

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp"}


def load_model(weights: Path, device: torch.device):
    ckpt = torch.load(weights, map_location=device, weights_only=False)
    encoder = ckpt.get("encoder", C.ENCODER)
    img_size = ckpt.get("img_size", C.IMG_SIZE)
    model = build_model(encoder, pretrained=False).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    meta = {
        "encoder": encoder, "img_size": img_size,
        "epoch": ckpt.get("epoch"),
        "val_dice": ckpt.get("val_dice"),
        "val_iou": ckpt.get("val_iou"),
    }
    return model, meta


@torch.no_grad()
def infer_one(model, img_bgr: np.ndarray, img_size: int, device: torch.device) -> np.ndarray:
    """返回 (K, img_size, img_size) 概率图。"""
    h0, w0 = img_bgr.shape[:2]
    img = cv2.resize(img_bgr, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    mean = np.asarray(C.IMAGENET_MEAN, dtype=np.float32).reshape(3, 1, 1)
    std = np.asarray(C.IMAGENET_STD, dtype=np.float32).reshape(3, 1, 1)
    x = (np.transpose(rgb, (2, 0, 1)) - mean) / std
    t = torch.from_numpy(np.ascontiguousarray(x))[None].to(device)
    prob = torch.sigmoid(model(t)).float()[0].cpu().numpy()
    return prob


def analyze(img_bgr: np.ndarray, prob: np.ndarray, img_size: int) -> dict:
    """
    从概率图得到**实例级**结果：每个药片、每个抑菌圈独立实例，并给出归属关系。

    步骤：
      1. 药片连通域 → 逐实例（实测连通域数 == 真实实例数）
      2. 以药片质心为种子，在抑菌圈区域内做测地距离划分 → 每个药片的圈实例
      3. 用 6 mm 药片内标标定：mm_per_px = 6 / median(各药片等效直径)

    返回的每一行对应**一个药片**（而不是一个圈），没圈的药片也会出现在行里，
    这样「这个药片没有抑菌圈」是可以直接读出来的。
    """
    mask = (prob > C.MASK_THRESH).astype(np.float32)
    inst = I.extract_instances(mask, img_size=img_size)

    disk_d = [d["eq_diameter_px"] for d in inst["disks"]]
    mm_per_px = None
    if disk_d:
        med = float(np.median(disk_d))
        if med > 1e-6:
            mm_per_px = 6.0 / med                     # 6 mm 标准纸片内标

    disk_by_id = {d["id"]: d for d in inst["disks"]}
    zone_by_id = {z["id"]: z for z in inst["zones"]}
    rows = []
    for p in inst["pairs"]:
        d = disk_by_id[p["disk_id"]]
        z = zone_by_id.get(p["zone_id"]) if p["zone_id"] is not None else None
        dpx = d["eq_diameter_px"]
        zpx = z["eq_diameter_px"] if z is not None else None
        rows.append({
            "disk_id": d["id"],
            "disk_cx_px": round(d["centroid"][0], 1),
            "disk_cy_px": round(d["centroid"][1], 1),
            "disk_diameter_px": round(dpx, 2),
            "zone_id": z["id"] if z is not None else None,
            "zone_cx_px": round(z["centroid"][0], 1) if z is not None else None,
            "zone_cy_px": round(z["centroid"][1], 1) if z is not None else None,
            "zone_area_px": round(z["area_px"], 1) if z is not None else None,
            "zone_diameter_px": round(zpx, 2) if zpx is not None else None,
            "disk_diameter_mm": round(dpx * mm_per_px, 3) if mm_per_px else None,
            "zone_diameter_mm": round(zpx * mm_per_px, 3) if (mm_per_px and zpx) else None,
            "zone_to_disk_ratio": round(zpx / dpx, 4) if (zpx and dpx > 1e-6) else None,
            "has_zone": z is not None,
        })

    return {
        "n_disks": inst["n_disks"],
        "n_zones": inst["n_zones"],
        "n_with_zone": sum(1 for r in rows if r["has_zone"]),
        "n_without_zone": sum(1 for r in rows if not r["has_zone"]),
        "mm_per_px": mm_per_px,
        "rows": rows,
        "instances": inst,
        "mask": mask,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="ResNet+U-Net 抑菌圈推理")
    ap.add_argument("--weights", type=Path, required=True, help="best.pt 路径")
    ap.add_argument("--source", type=Path, default=C.SRC_IMAGES,
                    help="图片文件或目录")
    ap.add_argument("--out", type=Path, default=C.EXPORT / "predict")
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()

    if not args.weights.is_file():
        raise SystemExit(f"找不到权重：{args.weights}")

    device = torch.device("cuda" if (torch.cuda.is_available() and not args.cpu) else "cpu")
    model, meta = load_model(args.weights, device)
    img_size = meta["img_size"]
    print(f"权重: {args.weights}")
    print(f"  编码器={meta['encoder']}  img_size={img_size}  "
          f"epoch={meta['epoch']}  val_dice={meta['val_dice']}")
    print(f"设备: {device}\n")

    src = Path(args.source)
    files = [src] if src.is_file() else sorted(
        f for f in src.iterdir() if f.suffix.lower() in IMG_EXT)
    if not files:
        raise SystemExit(f"{src} 里没有图片")

    args.out.mkdir(parents=True, exist_ok=True)
    from myseg.visualize import save_overlay

    all_rows: list[dict] = []
    for f in files:
        img = imread(f)
        if img is None:
            print(f"  跳过（读不了）：{f.name}")
            continue
        prob = infer_one(model, img, img_size, device)
        r = analyze(img, prob, img_size)

        save_overlay(img, prob, np.zeros_like(prob), args.out / f"{f.stem}_overlay.jpg")

        for row in r["rows"]:
            row["file"] = f.name
            all_rows.append(row)

        cal = f"{r['mm_per_px']:.5f} mm/px" if r["mm_per_px"] else "失败(未检出药片)"
        print(f"  {f.name:44s} 药片实例={r['n_disks']:2d}  抑菌圈实例={r['n_zones']:2d}  "
              f"有圈={r['n_with_zone']:2d}  无圈={r['n_without_zone']:2d}  标定={cal}")

    if all_rows:
        csv_path = args.out / "instances.csv"
        cols = ["file", "disk_id", "disk_cx_px", "disk_cy_px", "disk_diameter_px",
                "zone_id", "zone_cx_px", "zone_cy_px", "zone_area_px",
                "zone_diameter_px", "disk_diameter_mm", "zone_diameter_mm",
                "zone_to_disk_ratio", "has_zone"]
        with csv_path.open("w", newline="", encoding="utf-8-sig") as fp:
            w = csv.DictWriter(fp, fieldnames=cols, extrasaction="ignore")
            w.writeheader()
            w.writerows(all_rows)
        n_with = sum(1 for r in all_rows if r["has_zone"])
        print(f"\n实例明细已写入：{csv_path}"
              f"（{len(all_rows)} 个药片实例，其中 {n_with} 个有抑菌圈，"
              f"{len(all_rows) - n_with} 个无圈）")

    (args.out / "run_info.json").write_text(
        json.dumps({"weights": str(args.weights), **meta, "n_images": len(files)},
                   indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"可视化目录：{args.out}")


if __name__ == "__main__":
    main()
