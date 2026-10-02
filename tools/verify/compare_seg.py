"""对比 C# 端（App 里真正跑的 ZoneSegModel）与真值标注的实例级/语义级指标。

用法:
    python compare_seg.py <csharp_out_dir> [<csharp_out_dir_2> ...]

每个目录里应有 Verify.exe 产出的:
    seg_csharp.json            每图实例列表
    masks/<stem>__<Class>_<id>.png   640×640 二值掩膜

真值来自 `Antibacterial zone mask/labels/<stem>.txt`（YOLO-seg，类别 0=Area/抑菌圈，1=Yaoping/药片）。
比较在 **640×640 模型尺度** 上进行：真值多边形坐标按 (640/原图宽, 640/原图高) 缩放，
与 App 的「整图直接 resize 到 640」完全一致。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]          # 工作区根
LABELS = ROOT / "Antibacterial zone mask" / "labels"
CLASSES = {0: "Area", 1: "Yaoping"}
IOU_MATCH = 0.5


def read_yolo_seg(path: Path, w: int, h: int):
    out = []
    if not path.is_file():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) < 7:
            continue
        cls = int(float(parts[0]))
        xy = np.asarray([float(v) for v in parts[1:]], dtype=np.float32)
        if xy.size % 2:
            xy = xy[:-1]
        pts = xy.reshape(-1, 2)
        pts[:, 0] *= w
        pts[:, 1] *= h
        out.append((cls, pts))
    return out


def frame_rotation(stem: str) -> int:
    """返回把「App/SkiaSharp 解出的原始帧」旋转到「正立帧」的 np.rot90 次数 k。

    原因：本轮测试图里有大量 EXIF Orientation=6 的手机照片。
      · OpenCV imdecode（= 标注/semseg 训练时的解码方式）会自动应用 EXIF → 正立帧
      · SkiaSharp SKBitmap.Decode（= App 的解码方式）不应用 EXIF → 原始存储帧
    两者相差一个 90° 旋转；由于「先旋转再拉伸到 640」与「先拉伸再旋转」等价，
    预测掩膜只要旋转同样的 k 就能与真值（正立帧归一化坐标）比较。
    """
    raw = None
    for cand in (ROOT / "Antibacterial zone mask" / "data" / "test" / "images" / f"{stem}.jpg",
                 ROOT / "Antibacterial zone mask" / "images" / f"{stem}.jpg"):
        if cand.is_file():
            raw = cv2.imdecode(np.fromfile(str(cand), dtype=np.uint8),
                               cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
            up = cv2.imdecode(np.fromfile(str(cand), dtype=np.uint8), cv2.IMREAD_COLOR)
            break
    if raw is None:
        return 0
    best, bk = None, 0
    for k in range(4):
        r = np.rot90(raw, k)
        if r.shape != up.shape:
            continue
        d = float(np.mean(np.abs(r.astype(np.int16) - up.astype(np.int16))))
        if best is None or d < best:
            best, bk = d, k
    return bk


def gt_instances(stem: str, size: int = 640):
    """真值实例掩膜（640 尺度）——标注是归一化坐标，直接乘 640 即可（正立帧 → 640 正方形）。"""
    out = {0: [], 1: []}
    p = LABELS / f"{stem}.txt"
    if not p.is_file():
        return out
    for line in p.read_text(encoding="utf-8").splitlines():
        v = line.split()
        if len(v) < 7:
            continue
        cls = int(float(v[0]))
        xy = np.asarray([float(x) for x in v[1:]], dtype=np.float32)
        if xy.size % 2:
            xy = xy[:-1]
        pts = xy.reshape(-1, 2) * size
        m = np.zeros((size, size), np.uint8)
        cv2.fillPoly(m, [np.round(pts).astype(np.int32)], 1)
        if cls in out:
            out[cls].append(m)
    return out


def iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = int(np.logical_and(a > 0, b > 0).sum())
    union = int(np.logical_or(a > 0, b > 0).sum())
    return inter / union if union else 0.0


def match_instances(preds, gts, thr=IOU_MATCH):
    """贪心按 IoU 匹配；返回 (tp, fp, fn, 匹配 IoU 列表)"""
    used = [False] * len(gts)
    matched_ious = []
    tp = 0
    for pm in preds:
        best, bi = 0.0, -1
        for j, gm in enumerate(gts):
            if used[j]:
                continue
            v = iou(pm, gm)
            if v > best:
                best, bi = v, j
        if best >= thr and bi >= 0:
            used[bi] = True
            tp += 1
            matched_ious.append(best)
    return tp, len(preds) - tp, len(gts) - tp, matched_ious


def evaluate(out_dir: Path):
    rows = json.loads((out_dir / "seg_csharp.json").read_text(encoding="utf-8"))
    agg = {c: dict(tp=0, fp=0, fn=0, ious=[], pred=0, gt=0) for c in (0, 1)}
    sem = {c: dict(inter=0, union=0, pred=0, gt=0) for c in (0, 1)}
    per_image = []

    for row in rows:
        stem = row["image"]
        k_rot = frame_rotation(stem)
        gts = gt_instances(stem)
        preds = {0: [], 1: []}
        for p in row["instances"]:
            cid = p["cls_id"]
            mp = out_dir / "masks" / f"{stem}__{p['cls']}_{p['disk_index']}.png"
            m = cv2.imdecode(np.fromfile(str(mp), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
            if m is None:
                continue
            mb = (m > 127).astype(np.uint8)
            if k_rot:
                mb = np.rot90(mb, k_rot)   # 原始帧 → 正立帧（与真值同一坐标系）
            preds[cid].append(mb)

        line = {"image": stem, "rot": k_rot,
                "n_pred": {CLASSES[c]: len(preds[c]) for c in (0, 1)},
                "n_gt": {CLASSES[c]: len(gts.get(c, [])) for c in (0, 1)}}
        for c in (0, 1):
            tp, fp, fn, ious = match_instances(preds[c], gts.get(c, []))
            a = agg[c]
            a["tp"] += tp; a["fp"] += fp; a["fn"] += fn
            a["ious"] += ious
            a["pred"] += len(preds[c]); a["gt"] += len(gts.get(c, []))
            # 语义（并集）
            pu = np.zeros((640, 640), np.uint8)
            for m in preds[c]:
                pu |= m
            gu = np.zeros((640, 640), np.uint8)
            for m in gts.get(c, []):
                gu |= m
            s = sem[c]
            s["inter"] += int(np.logical_and(pu > 0, gu > 0).sum())
            s["union"] += int(np.logical_or(pu > 0, gu > 0).sum())
            s["pred"] += int((pu > 0).sum()); s["gt"] += int((gu > 0).sum())
            line[f"{CLASSES[c]}_iou_matched"] = round(float(np.mean(ious)), 4) if ious else None

        # 药片↔圈 归属：预测圈所属药片(disk_index) 是否与真值中「包含该药片」的圈一致
        per_image.append(line)

    print(f"\n{'='*78}\n{out_dir.name}   (C# 端，直接在 App 的预处理/后处理路径上跑的)\n{'='*78}")
    print(f"{'类别':<10}{'pred':>6}{'gt':>6}{'TP':>6}{'FP':>6}{'FN':>6}"
          f"{'P':>8}{'R':>8}{'F1':>8}{'匹配IoU':>9}")
    for c in (0, 1):
        a = agg[c]
        p = a["tp"] / max(a["tp"] + a["fp"], 1)
        r = a["tp"] / max(a["tp"] + a["fn"], 1)
        f = 2 * p * r / max(p + r, 1e-9)
        mi = float(np.mean(a["ious"])) if a["ious"] else float("nan")
        print(f"{CLASSES[c]:<10}{a['pred']:>6}{a['gt']:>6}{a['tp']:>6}{a['fp']:>6}{a['fn']:>6}"
              f"{p:>8.4f}{r:>8.4f}{f:>8.4f}{mi:>9.4f}")

    print("\n语义（两类掩膜并集，640 尺度）：")
    for c in (0, 1):
        s = sem[c]
        dice = 2 * s["inter"] / max(s["pred"] + s["gt"], 1)
        iou_v = s["inter"] / max(s["union"], 1)
        print(f"  {CLASSES[c]:<9} Dice={dice:.4f}  IoU={iou_v:.4f}")

    bad = [l for l in per_image
           if l["n_pred"]["Yaoping"] != l["n_gt"]["Yaoping"]
           or l["n_pred"]["Area"] != l["n_gt"]["Area"]]
    print(f"\n实例数与真值不一致的图（{len(bad)}/{len(per_image)}）：")
    for l in bad[:15]:
        print(f"  {l['image']:<18} pred {l['n_pred']}  gt {l['n_gt']}")
    return per_image


if __name__ == "__main__":
    for d in sys.argv[1:]:
        evaluate(Path(d))
