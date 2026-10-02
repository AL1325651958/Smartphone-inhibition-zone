"""
myseg/instance_viz.py — 实例级可视化：每个药片/抑菌圈独立实例 + 归属连线

输出「像 YOLO-seg 那样」的图：
  * 每个实例一种**不同颜色**（不是按类别上色，而是按实例上色）
  * 实例编号直接标在图上（D1/D2… 药片，Z1/Z2… 抑菌圈）
  * 药片与它的抑菌圈之间画一条**连线**，一眼看出归属
  * 无圈的药片用灰色标出（耐药 / 未标注）

面板布局：
    原图 | 语义分割（2 类） | 实例分割（每实例一色 + 归属连线） | 差异图

用法
----
    python -m myseg.instance_viz --weights runs_unet/<run>/best.pt
    python -m myseg.instance_viz --weights ... --split val
    python -m myseg.instance_viz --weights ... --limit 5      # 只出前 5 张
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from myseg import config as C
    from myseg import instance as I
    from myseg import metrics as M
    from myseg.dataset import SegDataset
    from myseg.io_utils import imwrite
    from myseg.model import build_model
else:
    from myseg import config as C
    from myseg import instance as I
    from myseg import metrics as M
    from myseg.dataset import SegDataset
    from myseg.io_utils import imwrite
    from myseg.model import build_model

# 20 种高区分度颜色（BGR），按实例循环取用
PALETTE = [
    (56, 56, 255), (151, 157, 255), (31, 112, 255), (29, 178, 255),
    (49, 210, 207), (10, 249, 72), (23, 204, 146), (134, 219, 61),
    (52, 147, 26), (187, 212, 0), (168, 153, 44), (255, 194, 0),
    (147, 69, 52), (255, 115, 100), (236, 24, 0), (255, 56, 132),
    (133, 0, 82), (255, 56, 203), (200, 149, 255), (199, 55, 255),
]
COL_NODISK = (150, 150, 150)          # 无圈药片：灰


def _denorm(x) -> np.ndarray:
    import cv2
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    mean = np.asarray(C.IMAGENET_MEAN, dtype=np.float32).reshape(3, 1, 1)
    std = np.asarray(C.IMAGENET_STD, dtype=np.float32).reshape(3, 1, 1)
    img = np.clip((x * std + mean) * 255.0, 0, 255).astype(np.uint8)
    return cv2.cvtColor(np.transpose(img, (1, 2, 0)), cv2.COLOR_RGB2BGR)


def _semantic_panel(base: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """按类别上色（对比用）。"""
    import cv2
    vis = base.copy()
    overlay = vis.copy()
    for c, color in ((0, (117, 158, 29)), (1, (165, 95, 24))):
        m = (mask[c] > C.MASK_THRESH).astype(np.uint8)
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if cnts:
            cv2.drawContours(overlay, cnts, -1, color, cv2.FILLED)
            cv2.drawContours(vis, cnts, -1, color, 3)
    return cv2.addWeighted(overlay, 0.3, vis, 0.7, 0)


def _instance_panel(base: np.ndarray, res: dict) -> np.ndarray:
    """
    每实例一色 + 编号 + 药片↔抑菌圈 归属连线。
    """
    import cv2
    vis = base.copy()

    # ── 抑菌圈实例 ──
    for z in res["zones"]:
        color = PALETTE[z["id"] % len(PALETTE)]
        m = z["mask"].astype(np.uint8)
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            continue
        overlay = vis.copy()
        cv2.drawContours(overlay, cnts, -1, color, cv2.FILLED)
        vis = cv2.addWeighted(overlay, 0.30, vis, 0.70, 0)
        cv2.drawContours(vis, cnts, -1, color, 3)

    # ── 药片实例 ──
    zone_of_disk = {p["disk_id"]: p["zone_id"] for p in res["pairs"]}
    for d in res["disks"]:
        zid = zone_of_disk.get(d["id"])
        color = PALETTE[zid % len(PALETTE)] if zid is not None else COL_NODISK
        m = d["mask"].astype(np.uint8)
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if cnts:
            cv2.drawContours(vis, cnts, -1, color, max(2, 0))
            cv2.drawContours(vis, cnts, -1, (255, 255, 255), 1)

    # ── 归属连线：药片中心 ←→ 抑菌圈中心 ──
    disk_by_id = {d["id"]: d for d in res["disks"]}
    for p in res["pairs"]:
        if p["zone_id"] is None:
            continue
        d = disk_by_id[p["disk_id"]]
        z = res["zones"][p["zone_id"]]
        dc = tuple(int(round(v)) for v in d["centroid"])
        zc = tuple(int(round(v)) for v in z["centroid"])
        color = PALETTE[p["zone_id"] % len(PALETTE)]
        cv2.line(vis, dc, zc, (0, 0, 0), 5, cv2.LINE_AA)
        cv2.line(vis, dc, zc, color, 2, cv2.LINE_AA)

    # ── 编号 ──
    for p in res["pairs"]:
        d = disk_by_id[p["disk_id"]]
        dc = tuple(int(round(v)) for v in d["centroid"])
        if p["zone_id"] is None:
            txt, col = f"D{p['disk_id'] + 1}(-)", COL_NODISK
        else:
            z = res["zones"][p["zone_id"]]
            zc = tuple(int(round(v)) for v in z["centroid"])
            col = PALETTE[p["zone_id"] % len(PALETTE)]
            _text(vis, f"Z{p['zone_id'] + 1}", zc, col)
            txt = f"D{p['disk_id'] + 1}->Z{p['zone_id'] + 1}"
        _text(vis, txt, (dc[0] - 40, dc[1] + 26), col)

    cv2.putText(vis, f"disks={res['n_disks']}  zones={res['n_zones']}",
                (10, vis.shape[0] - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (0, 255, 255), 2, cv2.LINE_AA)
    return vis


def _text(img, s, org, color):
    import cv2
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)


def _diff_panel(base: np.ndarray, pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    import cv2
    p = (pred > C.MASK_THRESH).any(axis=0)
    g = (gt > 0.5).any(axis=0)
    tp, fp, fn = p & g, p & ~g, ~p & g
    out = (base.astype(np.float32) * 0.45).astype(np.uint8)
    out[tp] = (60, 200, 60)
    out[fp] = (40, 40, 230)
    out[fn] = (230, 140, 30)
    hit, extra, miss = int(tp.sum()), int(fp.sum()), int(fn.sum())
    den = hit + extra + miss
    items = [(f"TP {hit}px", (60, 200, 60)), (f"FP {extra}px", (40, 40, 230)),
             (f"FN {miss}px", (230, 140, 30)),
             (f"IoU {hit / den if den else 1.0:.3f}", (255, 255, 255))]
    for i, (t, col) in enumerate(items):
        _text(out, t, (10, 26 + i * 24), col)
    return out


def _header(tile: np.ndarray, title: str) -> np.ndarray:
    import cv2
    t = tile.copy()
    cv2.rectangle(t, (0, 0), (t.shape[1], 40), (22, 22, 22), -1)
    cv2.putText(t, title, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                (255, 255, 255), 2, cv2.LINE_AA)
    return t


@torch.no_grad()
def run(args) -> Path:
    from torch.utils.data import DataLoader

    device = torch.device("cuda" if (torch.cuda.is_available() and not args.cpu) else "cpu")
    weights = Path(args.weights)
    ckpt = torch.load(weights, map_location="cpu", weights_only=False)
    encoder, img_size = ckpt.get("encoder", C.ENCODER), ckpt.get("img_size", C.IMG_SIZE)
    model = build_model(encoder, pretrained=False).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    ds = SegDataset(args.split, use_aug=False, img_size=img_size)
    loader = DataLoader(ds, batch_size=args.batch, shuffle=False,
                        num_workers=args.workers)

    out_dir = C.EXPORT / "instance" / f"{weights.stem}_{args.split}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 74)
    print(f"权重      : {weights}")
    print(f"模型      : {encoder} @ {img_size}  (epoch {ckpt.get('epoch')})")
    print(f"评估集    : {args.split}  {len(ds)} 张")
    print(f"输出      : {out_dir}")
    print("=" * 74)

    summary = []
    done = 0
    for batch in loader:
        x = batch["image"].to(device, non_blocking=True)
        prob = torch.sigmoid(model(x)).float().cpu().numpy()
        gt = batch["mask"].numpy()

        for i in range(prob.shape[0]):
            name = batch["name"][i]
            res = I.extract_instances(prob[i], img_size=img_size)
            base = _denorm(batch["image"][i])

            if not args.no_vis:
                tile = np.hstack([
                    _header(base, "original"),
                    _header(_semantic_panel(base, prob[i]), "semantic (2 classes)"),
                    _header(_instance_panel(base, res), "instances + association"),
                    _header(_diff_panel(base, prob[i], gt[i]), "difference"),
                ])
                imwrite(out_dir / f"{name}.jpg", tile, quality=90)

            summary.append({
                "image": name,
                "n_disks": res["n_disks"],
                "n_zones": res["n_zones"],
                "n_pairs": sum(1 for p in res["pairs"] if p["zone_id"] is not None),
                "n_disks_without_zone": sum(1 for p in res["pairs"] if p["zone_id"] is None),
                "disks": [
                    {"id": d["id"], "centroid": [round(v, 1) for v in d["centroid"]],
                     "area_px": round(d["area_px"], 1),
                     "eq_diameter_px": round(d["eq_diameter_px"], 1),
                     "zone_id": next((p["zone_id"] for p in res["pairs"]
                                      if p["disk_id"] == d["id"]), None)}
                    for d in res["disks"]
                ],
                "zones": [
                    {"id": z["id"], "disk_id": z["disk_id"],
                     "centroid": [round(v, 1) for v in z["centroid"]],
                     "area_px": round(z["area_px"], 1),
                     "eq_diameter_px": round(z["eq_diameter_px"], 1)}
                    for z in res["zones"]
                ],
            })
            done += 1
            if done % 10 == 0 or done == len(ds):
                print(f"  已处理 {done}/{len(ds)}")
            if args.limit and done >= args.limit:
                break
        if args.limit and done >= args.limit:
            break

    (out_dir / "instances.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    tot_d = sum(r["n_disks"] for r in summary)
    tot_z = sum(r["n_zones"] for r in summary)
    tot_p = sum(r["n_pairs"] for r in summary)
    tot_n = sum(r["n_disks_without_zone"] for r in summary)
    print()
    print(f"合计：药片 {tot_d} 个，抑菌圈 {tot_z} 个，成功配对 {tot_p} 对，"
          f"无圈药片 {tot_n} 个")
    print(f"实例明细：{out_dir / 'instances.json'}")
    if not args.no_vis:
        print(f"可视化  ：{out_dir}（每实例一色 + 归属连线 + 编号）")
    return out_dir


def main() -> None:
    ap = argparse.ArgumentParser(description="实例级可视化（每实例一色 + 归属连线）")
    ap.add_argument("--weights", type=Path, required=True)
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0, help="只处理前 N 张")
    ap.add_argument("--no-vis", action="store_true", help="只出 instances.json")
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()
