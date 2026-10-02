"""
myseg/metrics.py — 分割指标与直径测量误差

分两层
------
1. **像素层**（回应审稿人对「分类别指标」的要求）
   逐类别 IoU / Dice / Precision / Recall / Accuracy，基于混淆计数，不用 sklearn。

2. **实例层 / 测量层**（这才是本研究真正的主指标）
   从预测 mask 拆连通域 → 药片与抑菌圈配对 → 算直径 → 与人工标注比：
     MAE / RMSE / 偏倚 / 95% LoA / 落在 ±1、±2 mm 内的比例
   直径以「药片 6 mm 内标」标定：mm_per_px = 6.0 / median(预测药片直径 px)。
"""
from __future__ import annotations

import numpy as np

try:
    from myseg import config as C
    from myseg import labels as L
except ImportError:                                     # 允许直接运行
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from myseg import config as C
    from myseg import labels as L


# ────────────────────────────────────────────────
#  1. 像素层指标
# ────────────────────────────────────────────────
def confusion_counts(pred: np.ndarray, target: np.ndarray, thresh: float = C.MASK_THRESH):
    """
    pred / target: (K, H, W) 概率或 0/1。
    返回每类的 (tp, fp, fn, tn)。
    """
    out = []
    for c in range(pred.shape[0]):
        p = pred[c] > thresh
        t = target[c] > 0.5
        tp = int(np.logical_and(p, t).sum())
        fp = int(np.logical_and(p, ~t).sum())
        fn = int(np.logical_and(~p, t).sum())
        tn = int(np.logical_and(~p, ~t).sum())
        out.append((tp, fp, fn, tn))
    return out


def pixel_metrics(pred: np.ndarray, target: np.ndarray, thresh: float = C.MASK_THRESH) -> dict:
    """返回各类的 IoU / Dice / Precision / Recall / Accuracy。"""
    res = {}
    for c, (tp, fp, fn, tn) in enumerate(confusion_counts(pred, target, thresh)):
        eps = 1e-9
        res[C.CLASS_NAMES.get(c, str(c))] = {
            "IoU":       tp / (tp + fp + fn + eps),
            "Dice":      2 * tp / (2 * tp + fp + fn + eps),
            "Precision": tp / (tp + fp + eps),
            "Recall":    tp / (tp + fn + eps),
            "Accuracy":  (tp + tn) / (tp + tn + fp + fn + eps),
            "n_pos_px":  tp + fn,
        }
    return res


def to_binary(mask: np.ndarray, thresh: float = C.MASK_THRESH) -> np.ndarray:
    """
    把概率图或 0/1 掩膜统一成 float32 的 0/1。

    ⚠️ 必须显式调用：measure / measure_zone_diameters 内部用连通域掩膜做射线判断，
    若传入未二值化的概率，判断会被概率值污染，直径读数会严重偏大（实测差约 3 倍）。
    """
    return (np.asarray(mask) > thresh).astype(np.float32)


def aggregate_pixel_metrics(all_res: list[dict]) -> dict:
    """
    跨样本汇总：按每类的正像素数加权平均（像素多的类权重更高，符合分割惯例）。

    all_res 是多个样本的 pixel_metrics 输出，形如
        {'Area': {...}, 'Yaoping': {...}}
    每个类名下是 {'IoU','Dice','Precision','Recall','Accuracy','n_pos_px'}。
    """
    if not all_res:
        return {}
    names = list(all_res[0].keys())
    out: dict = {}
    for name in names:
        metric_keys = [k for k in all_res[0][name].keys() if k != "n_pos_px"]
        weights = [max(r[name]["n_pos_px"], 1) for r in all_res]      # 按样本取权重
        den = float(sum(weights))
        agg = {k: sum(r[name][k] * w for r, w in zip(all_res, weights)) / max(den, 1e-9)
               for k in metric_keys}
        agg["n_pos_px"] = int(sum(r[name]["n_pos_px"] for r in all_res))
        out[name] = agg
    return out


# ────────────────────────────────────────────────
#  2. 从 mask 得到直径读数
# ────────────────────────────────────────────────
def measure(mask: np.ndarray, img_size: int = C.IMG_SIZE,
            min_area: int = C.MIN_BLOB_AREA) -> dict:
    """
    从 (K,H,W) 的 0/1 mask 提取每个药片的抑菌圈直径（像素，img_size 尺度）。

    测量遵循论文定义（±15° 扇区、方向为皿心→药片、取扇区内方向的均值），
    见 labels.measure_zone_diameters 的文档。

    标定：mm_per_px = 6.0 / median(各药片等效直径)     ← 6 mm 标准纸片内标

    注意：mask 必须是 0/1。传概率请先过 to_binary()，否则读数会严重偏大。
    """
    mask = to_binary(mask)                                # 入口统一二值化，防误用
    res = L.measure_zone_diameters(mask[1], mask[0], min_area=min_area)   # 1=Yaoping, 0=Area
    disks = [it["disk"] for it in res["disks"]]
    disk_d = [d["eq_diameter_px"] for d in disks]

    mm_per_px = None
    if disk_d:
        med = float(np.median(disk_d))
        if med > 1e-6:
            mm_per_px = 6.0 / med                            # 6 mm 标准纸片内标

    measured = [it for it in res["disks"] if it["zone_diameter_px"] is not None]
    return {
        "disk_diameters_px": disk_d,
        "zone_diameters_px": [it["zone_diameter_px"] for it in measured],
        "pairs": res["disks"],                               # 每个药片一条，含未测到的
        "mm_per_px": mm_per_px,
        "n_disks": res["n_disks"],
        "n_zones": res["n_zones"],
        "n_measured": res["n_measured"],
        "plate_center": res["plate_center"],
    }


def _greedy_match(pred_d: list[float], gt_d: list[float]) -> list[tuple[int, int]]:
    """
    把预测直径与人工直径按数值最近配对（贪心）。
    两侧数量不一致时，多的一方不参与误差统计（单独报告）。
    """
    pairs: list[tuple[int, int]] = []
    used = set()
    for i, pv in enumerate(pred_d):
        best, bd = None, None
        for j, gv in enumerate(gt_d):
            if j in used:
                continue
            d = abs(pv - gv)
            if bd is None or d < bd:
                best, bd = j, d
        if best is not None:
            used.add(best)
            pairs.append((i, best))
    return pairs


# ────────────────────────────────────────────────
#  3. 测量层指标
# ────────────────────────────────────────────────
def measurement_error(pred_mm: list[float], gt_mm: list[float]) -> dict:
    """MAE / RMSE / 偏倚 / 95% LoA / 容差命中率。"""
    n = min(len(pred_mm), len(gt_mm))
    if n == 0:
        return {"n": 0}
    p = np.asarray(pred_mm[:n], dtype=np.float64)
    g = np.asarray(gt_mm[:n], dtype=np.float64)
    d = p - g
    bias = float(d.mean())
    sd = float(d.std(ddof=1)) if n > 1 else 0.0
    return {
        "n": int(n),
        "MAE": float(np.abs(d).mean()),
        "RMSE": float(np.sqrt((d ** 2).mean())),
        "bias": bias,
        "sd_diff": sd,
        "LoA_low": bias - 1.96 * sd,
        "LoA_high": bias + 1.96 * sd,
        "pct_within_0.5mm": float((np.abs(d) <= 0.5).mean() * 100),
        "pct_within_1mm": float((np.abs(d) <= 1.0).mean() * 100),
        "pct_within_2mm": float((np.abs(d) <= 2.0).mean() * 100),
    }


def evaluate_sample(pred_mask: np.ndarray, gt_mask: np.ndarray,
                    img_size: int = C.IMG_SIZE) -> dict:
    """
    单样本评估：像素指标 + 直径误差（以人工药片直径做标定，保证两者同一尺子）。
    """
    px = pixel_metrics(pred_mask, gt_mask)

    pr = measure(pred_mask, img_size)
    gt = measure(gt_mask, img_size)

    # 用人工标注的药片直径做标定，pred 与 gt 才能在同一物理尺度上比
    gt_disk = gt["disk_diameters_px"]
    mm_per_px = 6.0 / float(np.median(gt_disk)) if gt_disk else None

    out = {"pixel": px, "n_pred_zones": pr["n_zones"], "n_gt_zones": gt["n_zones"],
           "mm_per_px": mm_per_px,
           "pred_zone_px": pr["zone_diameters_px"], "gt_zone_px": gt["zone_diameters_px"]}

    if mm_per_px:
        pred_mm = [d * mm_per_px for d in pr["zone_diameters_px"]]
        gt_mm = [d * mm_per_px for d in gt["zone_diameters_px"]]
        m = _greedy_match(pred_mm, gt_mm)
        out["zone_error"] = measurement_error([pred_mm[i] for i, _ in m],
                                              [gt_mm[j] for _, j in m])
        out["n_matched"] = len(m)
        out["n_unmatched_pred"] = len(pred_mm) - len(m)
        out["n_unmatched_gt"] = len(gt_mm) - len(m)
    return out
