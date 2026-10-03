"""
myseg/instance.py — 实例级输出：每个药片、每个抑菌圈单独成实例，并给出归属关系

为什么需要这一层
----------------
模型输出的是**语义**掩膜（每类一张图），不是实例掩膜。
实测本项目数据：

    类别        人工标注多边形数(=真实实例数)   语义掩膜连通域数
    Yaoping     140                              140     ← 天然分离，连通域即实例
    Area        116                               45     ← 相邻的圈融合成一整块

也就是说药片本来就是一个个独立实例，而**抑菌圈会融合**，必须拆开才能回答
「这个圈是哪个药片的」。

拆分方案：把药片当种子
----------------------
每个抑菌圈区域里恰好含有一个药片，所以用**药片质心作为种子**，在圈区域内做
测地（沿区域内路径）最近种子划分：

  * 每个圈像素归属于「沿区域内路径最近的那个药片」
  * 不越出圈的真实外缘 —— 外边界完全保留，只在相邻圈相接处切开
  * 没有圈的药片（耐药）自然拿不到任何区域，如实输出「无圈」

这比在整幅图上做普通 Voronoi 好：普通 Voronoi 的分界线是直线，会切进圈内部、
且不受圈边界约束。

产出
----
    {
      'disks':  [{'id':0, 'mask':..., 'centroid':..., 'area_px':...}, ...],
      'zones':  [{'id':0, 'mask':..., 'centroid':..., 'area_px':..., 'disk_id':0}, ...],
      'pairs':  [{'disk_id':0, 'zone_id':0|None}, ...],
      'n_disks': int, 'n_zones': int,
    }
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


def _disk_instances(disk_mask: np.ndarray, min_area: int) -> list[dict]:
    """
    药片实例：连通域即实例（实测 20 张图无一例外，连通域数 == 真实实例数）。
    """
    import cv2
    comps = L.components(disk_mask, min_area=min_area)
    m = (disk_mask > 0.5).astype(np.uint8)
    _, lab = cv2.connectedComponents(m, connectivity=8)

    out: list[dict] = []
    for i, c in enumerate(comps):
        cx, cy = c["centroid"]
        ci, cj = int(round(cy)), int(round(cx))
        if not (0 <= ci < m.shape[0] and 0 <= cj < m.shape[1]) or lab[ci, cj] == 0:
            continue
        inst = (lab == lab[ci, cj]).astype(np.uint8)
        out.append({
            "id": i,
            "mask": inst,
            "centroid": (float(cx), float(cy)),
            "area_px": float(inst.sum()),
            "eq_diameter_px": L.equivalent_diameter(float(inst.sum())),
            "bbox": c["bbox"],
        })
    return out


def _geodesic_partition(zone_mask: np.ndarray, seeds: list[tuple[float, float]],
                        valid_region: np.ndarray | None = None) -> np.ndarray:
    """
    在 zone_mask 内部，按「沿区域内的测地距离」把每个像素分给最近的种子。

    实现方式：从所有种子同时做多源 BFS（每次扩张 1 像素），先到者占据。
    这样得到的是区域内的最短路径划分，不会越过区域外缘。
    返回 int32 标签图：-1 = 不属于任何种子（区域外），否则为种子下标。
    """
    from collections import deque

    h, w = zone_mask.shape[:2]
    allowed = (zone_mask > 0.5)
    if valid_region is not None:
        allowed &= (valid_region > 0.5)

    labels = np.full((h, w), -1, dtype=np.int32)
    q: deque = deque()

    for sid, (sx, sy) in enumerate(seeds):
        si, sj = int(round(sy)), int(round(sx))
        if not (0 <= si < h and 0 <= sj < w):
            continue
        if not allowed[si, sj]:
            # 种子必须落在区域里；不在就找区域内离它最近的点
            ys, xs = np.nonzero(allowed)
            if ys.size == 0:
                continue
            k = np.argmin((ys - si) ** 2 + (xs - sj) ** 2)
            si, sj = int(ys[k]), int(xs[k])
        if labels[si, sj] == -1:
            labels[si, sj] = sid
            q.append((si, sj))

    # 4 邻域多源 BFS
    while q:
        i, j = q.popleft()
        cur = labels[i, j]
        for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            ni, nj = i + di, j + dj
            if 0 <= ni < h and 0 <= nj < w and allowed[ni, nj] and labels[ni, nj] == -1:
                labels[ni, nj] = cur
                q.append((ni, nj))

    return labels


def extract_instances(mask: np.ndarray, min_area: int = C.MIN_BLOB_AREA,
                      img_size: int = C.IMG_SIZE) -> dict:
    """
    从语义掩膜 (2,H,W) 提取实例级结果并建立归属关系。

    参数
    ----
    mask : (2,H,W)，通道 0=Area(抑菌圈)，通道 1=Yaoping(药片)。0/1 或概率都行。

    返回见模块文档。
    """
    mask = (np.asarray(mask) > C.MASK_THRESH).astype(np.uint8)
    zone_sem = mask[0]
    disk_sem = mask[1]

    disks = _disk_instances(disk_sem, min_area)
    if not disks:
        return {"disks": [], "zones": [], "pairs": [],
                "n_disks": 0, "n_zones": 0}

    seeds = [d["centroid"] for d in disks]
    lab = _geodesic_partition(zone_sem, seeds)

    zones: list[dict] = []
    pairs: list[dict] = []
    for d in disks:
        sid = d["id"]
        zm = (lab == sid).astype(np.uint8)
        area = int(zm.sum())
        if area < min_area:
            pairs.append({"disk_id": sid, "zone_id": None,
                          "reason": "该药片周围没有抑菌圈（未分割到区域）"})
            continue
        ys, xs = np.nonzero(zm)
        zone = {
            "id": len(zones),
            "mask": zm,
            "centroid": (float(xs.mean()), float(ys.mean())),
            "area_px": float(area),
            "eq_diameter_px": L.equivalent_diameter(float(area)),
            "disk_id": sid,
        }
        pairs.append({"disk_id": sid, "zone_id": zone["id"],
                      "zone_to_disk_area_ratio": round(area / max(d["area_px"], 1e-6), 3)})
        zones.append(zone)

    return {
        "disks": disks, "zones": zones, "pairs": pairs,
        "n_disks": len(disks), "n_zones": len(zones),
        "instance_labels": lab,
    }


# ────────────────────────────────────────────────
#  与人工实例对比（客观打分用）
# ────────────────────────────────────────────────
def instance_iou(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    inter = float(np.logical_and(pred_mask > 0, gt_mask > 0).sum())
    union = float(np.logical_or(pred_mask > 0, gt_mask > 0).sum())
    return inter / union if union else 0.0


def match_instances(pred_masks: list[np.ndarray], gt_masks: list[np.ndarray],
                    iou_thresh: float = 0.5) -> dict:
    """
    贪心 IoU 匹配（仅用于评估，不参与推理）。返回匹配对与贪心指标。
    贪心匹配对实例分割评估足够，且避免引入额外依赖。
    """
    if not pred_masks or not gt_masks:
        return {"matched": [], "n_pred": len(pred_masks), "n_gt": len(gt_masks),
                "mean_iou_matched": 0.0, "tp": 0, "fp": len(pred_masks),
                "fn": len(gt_masks), "precision": 0.0, "recall": 0.0, "f1": 0.0}

    iou = np.zeros((len(pred_masks), len(gt_masks)), dtype=np.float64)
    for i, p in enumerate(pred_masks):
        for j, g in enumerate(gt_masks):
            iou[i, j] = instance_iou(p, g)

    used_p, used_g, matched = set(), set(), []
    order = np.argsort(-iou, axis=None)
    for flat in order:
        i, j = divmod(int(flat), len(gt_masks))
        if i in used_p or j in used_g or iou[i, j] < iou_thresh:
            continue
        used_p.add(i)
        used_g.add(j)
        matched.append((i, j, float(iou[i, j])))

    tp = len(matched)
    fp = len(pred_masks) - tp
    fn = len(gt_masks) - tp
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    return {
        "matched": matched, "n_pred": len(pred_masks), "n_gt": len(gt_masks),
        "tp": tp, "fp": fp, "fn": fn,
        "precision": prec, "recall": rec,
        "f1": 2 * prec * rec / max(prec + rec, 1e-9),
        "mean_iou_matched": float(np.mean([m[2] for m in matched])) if matched else 0.0,
        "iou_matrix": iou,
    }
