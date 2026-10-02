"""
reference_instances.py — 语义 2 通道 → 实例化参考实现（纯 NumPy / OpenCV，不依赖 torch）

位置：Antibacterial zone/_new_models/reference_instances.py
本脚本是 **C# 端（.NET MAUI + Microsoft.ML.OnnxRuntime）后处理的逐条参考**，
两侧必须按同一约定实现，否则实例结果对不上。

接口约定（与 C# 端一致）
-----------------------
模型：best.onnx，输入 float32 [1,3,640,640]（0..255 RGB，原图直接双线性 resize，无 letterbox），
      输出 float32 [1,2,640,640]（已 sigmoid 的概率），通道 0 = Area(抑菌圈)，通道 1 = Yaoping(药片)。

1) 阈值：两通道都用 **0.45** 得到二值掩膜。
2) 药片实例：`diskProb > 0.45` 做 **8 连通**连通域
   （等价 `cv2.connectedComponentsWithStats(..., connectivity=8)`），
   面积 < **30 px**（640 尺度）丢弃；每个连通域 = 一个药片实例，
   置信度 = 该连通域内 `diskProb` 的均值。
3) 抑菌圈实例：不用连通域（相邻圈会融合），用「药片播种 + 测地分区」：
   - 种子 = 每个药片连通域的质心四舍五入到整数像素，按连通域 label 升序全部入队（多源 BFS）
   - 搜索空间 = `zoneProb > 0.45` 的像素集合；**8 邻域** BFS，步长按像素计数（FIFO）
   - 每个 zone 像素归属「最早到达它的种子」，平局由队列顺序（= label 升序）决定
   - 药片 k 的圈实例 = 归属给它且 `zoneProb > 0.45` 的像素集合；
     该集合为空或面积 < 30 px → `has_zone=False`，不产生圈实例
   - 不与任何种子连通的 zone 像素记为 `orphan_zone_px`，不产生实例
4) 实例指标：预测实例掩膜 vs 真值多边形（`labels/<stem>.txt` 在**原始分辨率**下 fillPoly），
   两边都**最近邻缩放到 640×640** 再算 IoU，IoU ≥ 0.5 视为匹配（贪心）。

用法
----
    cd "Antibacterial zone mask"
    python "..\Antibacterial zone\_new_models\reference_instances.py" \
        --onnx "..\Antibacterial zone\_new_models\best.onnx" \
        --source data/test/images \
        --out "..\Antibacterial zone\_new_models\instances"
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

# ── 与 semseg/config.py 严格一致（本脚本不 import torch，故此处显式声明） ──
THRESH = 0.45
MIN_AREA = 30          # 640 尺度下的最小实例面积（px）
SEED_SEARCH_RADIUS = 32  # 种子回退方窗半径（px, 640 尺度），与 ZoneSegModel.cs 一致
IMG_SIZE = 640
CLASS_NAMES = {0: "Area", 1: "Yaoping"}
ZONE, DISK = 0, 1
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp"}

THIS = Path(__file__).resolve()
# 本文件位于 <workspace>/Antibacterial zone/_new_models/，
# 而模型包位于 <workspace>/Antibacterial zone mask/（两者是同级目录）
_CANDS = [THIS.parent.parent.parent / "Antibacterial zone mask",   # <workspace>/Antibacterial zone mask
          THIS.parent.parent / "Antibacterial zone mask"]
MASK_ROOT = next((p for p in _CANDS if (p / "semseg").is_dir()), _CANDS[0])
SRC_IMAGES = MASK_ROOT / "images"
SRC_LABELS = MASK_ROOT / "labels"
SPLIT_JSON = MASK_ROOT / "semseg" / "split.json"


# ════════════════════════════════════════════════
#  图像读写（中文路径安全：不做 cv2.imread/imwrite 直接调用）
# ════════════════════════════════════════════════
def imread(path) -> np.ndarray | None:
    p = Path(path)
    if not p.is_file():
        return None
    buf = np.fromfile(str(p), dtype=np.uint8)
    return cv2.imdecode(buf, cv2.IMREAD_COLOR) if buf.size else None


def imwrite(path, img: np.ndarray, quality: int = 92) -> bool:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    params = ([int(cv2.IMWRITE_JPEG_QUALITY), int(quality)]
              if p.suffix.lower() in (".jpg", ".jpeg") else [])
    ok, buf = cv2.imencode(p.suffix or ".png", img, params)
    if ok:
        buf.tofile(str(p))
    return bool(ok)


# ════════════════════════════════════════════════
#  预处理 / 推理
# ════════════════════════════════════════════════
def preprocess(img_bgr: np.ndarray, size: int = IMG_SIZE) -> np.ndarray:
    """原图 → [1,3,S,S] float32 的 0..255 RGB（与 dataset.py 的直接 resize 等价）。"""
    img = cv2.resize(img_bgr, (size, size), interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
    return rgb.transpose(2, 0, 1)[None]


class Backend:
    """ONNX Runtime 后端（默认）。也支持 --weights 走 PyTorch，便于对照。"""

    def __init__(self, onnx: Path | None = None, weights: Path | None = None,
                 cpu: bool = False):
        self.kind = "onnx" if onnx is not None else "torch"
        if onnx is not None:
            import onnxruntime as ort
            avail = ort.get_available_providers()
            prov = ["CPUExecutionProvider"] if cpu else (
                ["CUDAExecutionProvider", "CPUExecutionProvider"]
                if "CUDAExecutionProvider" in avail else ["CPUExecutionProvider"])
            self.sess = ort.InferenceSession(str(onnx), providers=prov)
            self.in_name = self.sess.get_inputs()[0].name
            self.backend = f"onnxruntime {ort.__version__} ({', '.join(prov)})"
        else:
            import torch
            sys.path.insert(0, str(MASK_ROOT))
            from semseg.model import build_model
            ck = torch.load(weights, map_location="cpu", weights_only=False)
            self.torch = torch
            self.device = torch.device(
                "cpu" if (cpu or not torch.cuda.is_available()) else "cuda")
            net = build_model(ck.get("encoder", "resnet18"), pretrained=False)
            net.load_state_dict(ck["model_state"])
            self.net = net.to(self.device).eval()
            self.backend = f"pytorch ({self.device})"

    def _run(self, x: np.ndarray) -> np.ndarray:
        """x: [1,3,S,S] float32 0..255 → 概率 [2,S,S]。"""
        if self.kind == "onnx":
            return self.sess.run(None, {self.in_name: x})[0][0]
        with self.torch.no_grad():
            t = self.torch.from_numpy(np.ascontiguousarray(x)).to(self.device)
            return self.torch.sigmoid(self.net(t)).float().cpu().numpy()[0]

    def prob(self, img_bgr: np.ndarray, tta: bool = True,
             size: int = IMG_SIZE) -> np.ndarray:
        """TTA 定义与 semseg/infer.py `_forward` 一致：rot90 k=1,2,3 + 水平翻转，5 视图平均。"""
        x = preprocess(img_bgr, size)
        outs = [self._run(x)]
        if tta:
            for k in (1, 2, 3):
                xr = np.rot90(x, k, axes=(2, 3))
                outs.append(np.rot90(self._run(np.ascontiguousarray(xr)), -k, axes=(1, 2)))
            xf = x[:, :, :, ::-1]
            outs.append(self._run(np.ascontiguousarray(xf))[:, :, ::-1])
        return np.mean(outs, axis=0).astype(np.float32)


# ════════════════════════════════════════════════
#  实例化：药片 = 连通域；抑菌圈 = 药片播种 + 测地分区
# ════════════════════════════════════════════════
def _geodesic_bfs(allowed: np.ndarray, seeds: list[tuple[int, int]],
                  max_rounds: int = 20000) -> tuple[np.ndarray, int]:
    """
    8 邻域多源 FIFO BFS（步长按像素计数，等权）。

    allowed : bool (H,W)，搜索空间
    seeds   : [(row, col), ...]，按 label 升序传入；全部先标记再扩张，
              因此「平局由队列顺序决定」= 同一波前内 row-major 先到者占据，
              符合 FIFO BFS 语义。

    实现说明：用 NumPy 分轮扩张，每一轮把当前 wavefront 的 8 邻域中尚未占用的
    像素一次性分配给「本轮所有来源」。同一轮内多个来源竞争同一像素时按
    row-major 顺序取第一个来源，与逐个出队的 FIFO BFS 结果一致（同一轮内
    所有来源的当前距离相等）。这样避免了 40 万像素的 Python 级 deque 循环。

    返回 (labels, n_rounds)，labels 为 int32：-1 = 未分配，>=0 = 种子下标。
    """
    h, w = allowed.shape
    labels = np.full((h, w), -1, dtype=np.int32)
    for sid, (si, sj) in enumerate(seeds):
        if 0 <= si < h and 0 <= sj < w and allowed[si, sj] and labels[si, sj] == -1:
            labels[si, sj] = sid
    wave = labels >= 0
    if not wave.any():
        return labels, 0

    rounds = 0
    padded = np.full((h + 2, w + 2), -1, dtype=np.int32)
    while True:
        prev = wave
        new = np.zeros_like(wave)
        for di in (-1, 0, 1):                      # 先求 8 邻域并集
            for dj in (-1, 0, 1):
                if di == 0 and dj == 0:
                    continue
                new |= np.roll(np.roll(prev, di, axis=0), dj, axis=1)
        new &= allowed & ~wave
        if not new.any():
            break
        # 顺序敏感赋值：同一波前内按 (di,dj) 顺序依次填充（row-major 语义，平局由顺序决定）
        padded[1:-1, 1:-1] = labels
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                if di == 0 and dj == 0:
                    continue
                if not new.any():
                    break
                src = padded[1 + di:1 + di + h, 1 + dj:1 + dj + w]
                take = new & (src >= 0)
                if not take.any():
                    continue
                labels[take] = src[take]
                new &= ~take
        wave = labels >= 0
        rounds += 1
        if rounds >= max_rounds:
            break
    return labels, rounds


def _bbox_of(mask: np.ndarray) -> list[int]:
    ys, xs = np.nonzero(mask)
    if ys.size == 0:
        return [0, 0, 0, 0]
    return [int(xs.min()), int(ys.min()),
            int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)]


def eq_diameter(area_px: float) -> float:
    """
    等效圆直径 d = 2√(A/π)。

    口径说明：本脚本里 `*_px` 一律是 **640 模型尺度**的像素；
    而 C# 端 `SegInstance.EquivalentDiameter()` 用的是**原图尺度**像素面积
    （mask 像素数 / (scaleX*scaleY)），两者差一个各向异性缩放因子。
    做 C#↔Python 对比时请按同一尺度换算，勿直接比这两个字段。
    """
    return float(2.0 * np.sqrt(max(area_px, 0.0) / np.pi))


def _seed_point(allowed: np.ndarray, cx: float, cy: float,
                radius: int = SEED_SEARCH_RADIUS) -> tuple[int, int] | None:
    """
    种子定位，与 ZoneSegModel.cs 的 FindSeed 逐条一致：

      1. sx = Math.Round(cx), sy = Math.Round(cy)  —— 银行家舍入（.NET Math.Round 默认
         ToEven；Python3 内置 round() 同样是 round-half-to-even，两者一致）
      2. clamp 到 [0, 639]
      3. 该像素在 zone 掩膜内 → 直接用
      4. 否则在 ±radius 的**方窗**内找最近 zone 像素：平方距离比较，
         dy 外层、dx 内层（行优先），严格小于才更新 → 平局取行优先先出现者
      5. 方窗内找不到 → 返回 None（该药片 has_zone=false，不入队、不产生圈实例）
    """
    h, w = allowed.shape
    sx = min(max(int(round(cx)), 0), w - 1)
    sy = min(max(int(round(cy)), 0), h - 1)
    if allowed[sy, sx]:
        return sy, sx
    best, bx, by = None, -1, -1
    for dy in range(-radius, radius + 1):
        ny = sy + dy
        if ny < 0 or ny >= h:
            continue
        for dx in range(-radius, radius + 1):
            nx = sx + dx
            if nx < 0 or nx >= w:
                continue
            if not allowed[ny, nx]:
                continue
            d = dx * dx + dy * dy
            if best is None or d < best:
                best, bx, by = d, nx, ny
    return None if bx < 0 else (by, bx)


def extract_instances(prob: np.ndarray, thresh: float = THRESH,
                      min_area: int = MIN_AREA) -> dict:
    """
    prob : (2,H,W) 概率图（640 尺度），通道 0 = Area，通道 1 = Yaoping。
    返回磁盘/圈实例、归属关系、孤儿圈像素计数、诊断量。
    """
    prob = np.asarray(prob, dtype=np.float32)
    zone_p, disk_p = prob[ZONE], prob[DISK]
    zone_bin = (zone_p > thresh)
    disk_bin = (disk_p > thresh).astype(np.uint8)

    # ── 药片：8 连通连通域 ──
    n_lab, labels, stats, cents = cv2.connectedComponentsWithStats(
        disk_bin, connectivity=8)
    disks: list[dict] = []
    for lab in range(1, n_lab):                         # label 升序
        area = int(stats[lab, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        m = (labels == lab)
        disks.append({
            "id": len(disks),
            "src_label": int(lab),
            "class": "Yaoping",
            "bbox": [int(stats[lab, cv2.CC_STAT_LEFT]), int(stats[lab, cv2.CC_STAT_TOP]),
                     int(stats[lab, cv2.CC_STAT_WIDTH]), int(stats[lab, cv2.CC_STAT_HEIGHT])],
            "area_px": area,
            "eq_diameter_px": round(eq_diameter(area), 3),
            "centroid": [round(float(cents[lab][0]), 3), round(float(cents[lab][1]), 3)],
            "mean_prob": round(float(disk_p[m].mean()), 6),
            "has_zone": False,
            "zone_id": None,
            "_mask": m.astype(np.uint8),
        })
    # 被 min_area 丢掉的药片小连通域（失败模式诊断）
    tiny_disks = [{"bbox": [int(stats[l, cv2.CC_STAT_LEFT]), int(stats[l, cv2.CC_STAT_TOP]),
                            int(stats[l, cv2.CC_STAT_WIDTH]), int(stats[l, cv2.CC_STAT_HEIGHT])],
                   "area_px": int(stats[l, cv2.CC_STAT_AREA]),
                   "centroid": [round(float(cents[l][0]), 2), round(float(cents[l][1]), 2)],
                   "max_prob": round(float(disk_p[labels == l].max()), 4)}
                  for l in range(1, n_lab)
                  if int(stats[l, cv2.CC_STAT_AREA]) < min_area]

    if not disks:
        return {"disks": [], "zones": [], "pairs": [],
                "n_disks": 0, "n_zones": 0, "n_zone_components": int(
                    cv2.connectedComponents(zone_bin.astype(np.uint8),
                                            connectivity=8)[0] - 1),
                "orphan_zone_px": int(zone_bin.sum()), "tiny_disks_dropped": tiny_disks,
                "bfs_rounds": 0, "zone_px_total": int(zone_bin.sum())}

    # ── 种子：药片质心四舍五入到整数像素，label 升序全部先入队 ──
    seeds: list[tuple[int, int] | None] = []
    allowed = zone_bin.copy()
    for d in disks:
        seeds.append(_seed_point(allowed, d["centroid"][0], d["centroid"][1]))

    valid_seeds = [(i, s) for i, s in enumerate(seeds) if s is not None]
    lab_map, rounds = (_geodesic_bfs(allowed, [s for _, s in valid_seeds])
                       if valid_seeds else (np.full(allowed.shape, -1, np.int32), 0))
    # 内部种子槽位 → disk id
    slot2disk = {slot: d_id for slot, (d_id, _) in enumerate(valid_seeds)}

    claimed = (lab_map >= 0)
    orphan_px = int((allowed & ~claimed).sum())
    n_zone_comp = int(cv2.connectedComponents(
        zone_bin.astype(np.uint8), connectivity=8)[0] - 1)

    zones: list[dict] = []
    pairs: list[dict] = []
    did2slot = {d_id: slot for slot, d_id in slot2disk.items()}
    for d in disks:
        slot = did2slot.get(d["id"])
        zm_u8 = ((lab_map == slot).astype(np.uint8) if slot is not None
                 else np.zeros(lab_map.shape, np.uint8))
        area = int(zm_u8.sum())
        if area < min_area:
            pairs.append({"disk_id": d["id"], "zone_id": None, "has_zone": False,
                          "zone_area_px": area,
                          "reason": ("该药片周围没有抑菌圈（未分割到区域）"
                                     if area == 0 else f"分到的圈面积 {area} < {min_area} px")})
            continue
        ys, xs = np.nonzero(zm_u8)
        zone = {
            "id": len(zones),
            "class": "Area",
            "disk_id": d["id"],
            "bbox": _bbox_of(zm_u8),
            "area_px": area,
            "eq_diameter_px": round(eq_diameter(area), 3),
            "centroid": [round(float(xs.mean()), 3), round(float(ys.mean()), 3)],
            "mean_prob": round(float(zone_p[zm_u8 > 0].mean()), 6),
            "_mask": zm_u8,
        }
        d["has_zone"] = True
        d["zone_id"] = zone["id"]
        pairs.append({"disk_id": d["id"], "zone_id": zone["id"], "has_zone": True,
                      "zone_area_px": area,
                      "zone_to_disk_area_ratio": round(area / max(d["area_px"], 1), 3)})
        zones.append(zone)
    del lab_map

    return {
        "disks": disks, "zones": zones, "pairs": pairs,
        "n_disks": len(disks), "n_zones": len(zones),
        "n_zone_components": n_zone_comp,
        "orphan_zone_px": orphan_px,
        "tiny_disks_dropped": tiny_disks,
        "bfs_rounds": rounds,
        "zone_px_total": int(allowed.sum()),
    }


# ════════════════════════════════════════════════
#  真值：labels/<stem>.txt → 实例掩膜（原始分辨率）
# ════════════════════════════════════════════════
def read_yolo_seg(path: Path, img_w: int, img_h: int):
    out = []
    p = Path(path)
    if not p.is_file():
        return out
    for line in p.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) < 7:
            continue
        try:
            cls = int(float(parts[0]))
            coords = np.asarray([float(v) for v in parts[1:]], dtype=np.float32)
        except ValueError:
            continue
        if coords.size % 2:
            coords = coords[:-1]
        pts = coords.reshape(-1, 2)
        pts[:, 0] *= img_w
        pts[:, 1] *= img_h
        out.append((cls, pts))
    return out


def gt_instances(stem: str, w: int, h: int, size: int = IMG_SIZE):
    """
    真值实例掩膜。按 C# 端约定口径：在**原始分辨率**下 fillPoly 渲染，
    再**最近邻**缩放到 640×640 后参与 IoU 匹配。只保留 640 掩膜（省内存）。
    返回 {cls: [mask640, ...]}，mask 为 uint8 0/1。
    """
    inst = read_yolo_seg(SRC_LABELS / f"{stem}.txt", w, h)
    out: dict[int, list[np.ndarray]] = {ZONE: [], DISK: []}
    for cls, pts in inst:
        if cls not in out or len(pts) < 3:
            continue
        full = np.zeros((h, w), np.uint8)
        cv2.fillPoly(full, [np.round(pts).astype(np.int32)], 1)
        small = cv2.resize(full, (size, size), interpolation=cv2.INTER_NEAREST)
        out[cls].append((small > 0).astype(np.uint8))
    return out


# ════════════════════════════════════════════════
#  实例匹配（贪心 IoU，COCO segm AP@0.5 口径）
# ════════════════════════════════════════════════
def iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = float(np.logical_and(a, b).sum())
    union = float(np.logical_or(a, b).sum())
    return inter / union if union else 0.0


def match_greedy(preds: list[dict], gts: list[np.ndarray], iou_thresh: float = 0.5):
    """preds: [{'mask':bool...}]；gts: [bool mask]。返回 (matched, mean_iou)。"""
    if not preds or not gts:
        return [], 0.0
    M = np.zeros((len(preds), len(gts)), np.float64)
    for i, p in enumerate(preds):
        for j, g in enumerate(gts):
            M[i, j] = iou(p["mask"], g)
    used_p, used_g, matched = set(), set(), []
    for flat in np.argsort(-M, axis=None):
        i, j = divmod(int(flat), len(gts))
        if i in used_p or j in used_g or M[i, j] < iou_thresh:
            continue
        used_p.add(i)
        used_g.add(j)
        matched.append({"pred": i, "gt": j, "iou": float(M[i, j])})
    mean_iou = float(np.mean([m["iou"] for m in matched])) if matched else 0.0
    return matched, mean_iou


def pair_stats(matched, n_pred: int, n_gt: int) -> dict:
    tp = len(matched)
    fp, fn = n_pred - tp, n_gt - tp
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    return {"n_pred": n_pred, "n_gt": n_gt, "tp": tp, "fp": fp, "fn": fn,
            "precision": round(prec, 4), "recall": round(rec, 4),
            "f1": round(2 * prec * rec / (prec + rec), 4) if (prec + rec) else 0.0}


# ════════════════════════════════════════════════
#  可视化
# ════════════════════════════════════════════════
def _palette(n: int) -> list[tuple[int, int, int]]:
    import colorsys
    out = []
    for i in range(max(n, 1)):
        r, g, b = colorsys.hsv_to_rgb((i * 0.61803) % 1.0, 0.85, 0.95)
        out.append((int(b * 255), int(g * 255), int(r * 255)))   # BGR
    return out


def draw_instances(img_bgr: np.ndarray, res: dict, size: int = IMG_SIZE) -> np.ndarray:
    base = cv2.resize(img_bgr, (size, size), interpolation=cv2.INTER_LINEAR)
    vis = (base.astype(np.float32) * 0.45).astype(np.uint8)
    cols = _palette(max(res["n_disks"], res["n_zones"], 1))
    for z in res["zones"]:
        vis[z["_mask"]] = cols[z["id"] % len(cols)]
    for d in res["disks"]:
        c = cols[d["id"] % len(cols)]
        vis[d["_mask"]] = (int(c[0] * 0.6), int(c[1] * 0.6), int(c[2] * 0.6))
    vis = cv2.addWeighted(vis, 0.65, base, 0.35, 0)
    for z in res["zones"]:
        cx, cy = int(z["centroid"][0]), int(z["centroid"][1])
        cv2.putText(vis, f"Z{z['id']}", (cx - 12, cy + 4), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(vis, f"Z{z['id']}", (cx - 12, cy + 4), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (0, 0, 0), 1, cv2.LINE_AA)
    for d in res["disks"]:
        cx, cy = int(d["centroid"][0]), int(d["centroid"][1])
        tag = f"D{d['id']}" + (f"->Z{d['zone_id']}" if d["has_zone"] else "(-)")
        col = (0, 255, 0) if d["has_zone"] else (128, 128, 128)
        cv2.putText(vis, tag, (cx - 14, cy - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(vis, tag, (cx - 14, cy - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    col, 1, cv2.LINE_AA)
    return vis


def draw_gt(img_bgr: np.ndarray, gti: dict, size: int = IMG_SIZE) -> np.ndarray:
    base = cv2.resize(img_bgr, (size, size), interpolation=cv2.INTER_LINEAR)
    vis = (base.astype(np.float32) * 0.45).astype(np.uint8)
    cols = _palette(max(len(gti[ZONE]), len(gti[DISK]), 1))
    for i, m in enumerate(gti[ZONE]):
        vis[m > 0] = cols[i % len(cols)]
    for i, m in enumerate(gti[DISK]):
        c = cols[i % len(cols)]
        vis[m > 0] = (int(c[0] * 0.6), int(c[1] * 0.6), int(c[2] * 0.6))
    return cv2.addWeighted(vis, 0.65, base, 0.35, 0)


# ════════════════════════════════════════════════
#  主流程
# ════════════════════════════════════════════════
def main() -> None:
    ap = argparse.ArgumentParser(description="语义掩膜 → 实例 + 归属（C# 端参考实现）")
    ap.add_argument("--onnx", type=Path, default=None)
    ap.add_argument("--weights", type=Path, default=None)
    ap.add_argument("--source", type=Path, required=True,
                    help="图片目录（或单张图片）")
    ap.add_argument("--out", type=Path, required=True, help="输出目录")
    ap.add_argument("--thresh", type=float, default=THRESH)
    ap.add_argument("--min-area", type=int, default=MIN_AREA)
    ap.add_argument("--split", default=None,
                    help="只跑 split.json 里的这个 split（test/val/train）")
    ap.add_argument("--no-tta", dest="tta", action="store_false", default=True)
    ap.add_argument("--no-eval", dest="eval", action="store_false", default=True,
                    help="不做与 labels/ 的实例指标（只出实例 JSON）")
    ap.add_argument("--no-vis", dest="vis", action="store_false", default=True)
    ap.add_argument("--masks", action="store_true",
                    help="额外导出每实例一张 640×640 二值掩膜 PNG（0/255），"
                         "命名 <stem>__<ClassName>_<disk_id>.png，与 C# 端一致")
    ap.add_argument("--dump-all", action="store_true",
                    help="额外导出汇总 instances.json（20 张图一个文件）")
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()

    if args.onnx is None and args.weights is None:
        raise SystemExit("必须指定 --onnx 或 --weights")

    out_dir = Path(args.out)
    (out_dir / "instances").mkdir(parents=True, exist_ok=True)
    if args.vis:
        (out_dir / "vis").mkdir(parents=True, exist_ok=True)
    if args.masks:
        (out_dir / "masks").mkdir(parents=True, exist_ok=True)

    src = Path(args.source)
    files = [src] if src.is_file() else sorted(
        f for f in src.iterdir() if f.suffix.lower() in IMG_EXT)
    if args.split:
        names = set(json.loads(SPLIT_JSON.read_text(encoding="utf-8"))[args.split])
        files = [f for f in files if f.stem in names]
    if not files:
        raise SystemExit(f"{src} 里没有图片")

    be = Backend(onnx=args.onnx, weights=args.weights, cpu=args.cpu)
    print("=" * 78)
    print(f"后端      : {be.backend}")
    print(f"阈值      : {args.thresh}   最小实例面积: {args.min_area} px (640 尺度)")
    print(f"TTA       : {'开（5 视图）' if args.tta else '关'}")
    print(f"图片      : {len(files)} 张")
    print(f"输出      : {out_dir}")
    print("=" * 78)

    per_image: list[dict] = []
    agg = {c: {"n_pred": 0, "n_gt": 0, "tp": 0, "fp": 0, "fn": 0,
               "iou_sum": 0.0, "n_matched": 0} for c in (ZONE, DISK)}
    pair_ok = pair_tot = 0
    ms_all: list[float] = []
    diagnostics: list[dict] = []
    all_images: list[dict] = []

    for f in files:
        img = imread(f)
        if img is None:
            print(f"  跳过（读不了）：{f.name}")
            continue
        h0, w0 = img.shape[:2]
        t0 = time.perf_counter()
        prob = be.prob(img, tta=args.tta)
        ms = (time.perf_counter() - t0) * 1000
        ms_all.append(ms)

        res = extract_instances(prob, args.thresh, args.min_area)

        # ── 与 labels/ 的实例级比较 ──
        ev = None
        if args.eval:
            gti = gt_instances(f.stem, w0, h0)
            ev = {}
            for cls, key in ((ZONE, "zones"), (DISK, "disks")):
                preds = [{"mask": z["_mask"]} for z in res[key]]
                gts = gti[cls]
                matched, miou = match_greedy(preds, gts, 0.5)
                st = pair_stats(matched, len(preds), len(gts))
                st["mean_iou_matched"] = round(miou, 4)
                ev[CLASS_NAMES[cls]] = st
                a = agg[cls]
                a["n_pred"] += st["n_pred"]
                a["n_gt"] += st["n_gt"]
                a["tp"] += st["tp"]
                a["fp"] += st["fp"]
                a["fn"] += st["fn"]
                if matched:
                    a["iou_sum"] += sum(m["iou"] for m in matched)
                    a["n_matched"] += len(matched)

            # 归属正确率：预测的每个圈实例，是否与「含它那个药片」的真值圈重叠最多
            gt_disk_masks = gti[DISK]
            gt_zone_masks = gti[ZONE]
            # 真值归属：药片 j 的质心落在哪个真值圈里（可能多个 → 取面积最小的那个）
            disk2gtzone = {}
            for j, dm in enumerate(gt_disk_masks):
                ys, xs = np.nonzero(dm)
                cy, cx = int(round(ys.mean())), int(round(xs.mean()))
                cands = [k for k, zm in enumerate(gt_zone_masks) if zm[cy, cx]]
                if cands:
                    disk2gtzone[j] = min(cands, key=lambda k: int(gt_zone_masks[k].sum()))
            for z in res["zones"]:
                did = z["disk_id"]
                # 该药片 → 真值药片（IoU 最大的，≥0.5 才认）
                pd = res["disks"][did]["_mask"]
                if gt_disk_masks:
                    ious = [iou(pd, g) for g in gt_disk_masks]
                    j = int(np.argmax(ious))
                    if ious[j] >= 0.5 and j in disk2gtzone:
                        pair_tot += 1
                        k = disk2gtzone[j]
                        if iou(z["_mask"], gt_zone_masks[k]) >= 0.5:
                            pair_ok += 1
            ev["_gt_counts"] = {"Area": len(gt_zone_masks), "Yaoping": len(gt_disk_masks)}
            if gt_zone_masks:
                union_zone = np.any([m > 0 for m in gt_zone_masks], axis=0).astype(np.uint8)
            else:
                union_zone = np.zeros((IMG_SIZE, IMG_SIZE), np.uint8)
            ev["_gt_zone_components"] = int(cv2.connectedComponents(
                union_zone, connectivity=8)[0] - 1)

        # ── 诊断 ──
        diag = {
            "image": f.stem,
            "n_disks_pred": res["n_disks"],
            "n_zones_pred": res["n_zones"],
            "n_zone_components": res["n_zone_components"],
            "orphan_zone_px": res["orphan_zone_px"],
            "tiny_disks_dropped": len(res["tiny_disks_dropped"]),
            "tiny_disks": res["tiny_disks_dropped"],
            "n_disks_no_zone": sum(1 for d in res["disks"] if not d["has_zone"]),
            "bfs_rounds": res["bfs_rounds"],
            "zone_px_total": res["zone_px_total"],
            "infer_ms": round(ms, 1),
        }
        if args.eval:
            diag["gt_counts"] = ev["_gt_counts"]
            diag["gt_zone_components"] = ev["_gt_zone_components"]
            diag["gt_zone_components_minus_pred"] = (
                ev["_gt_zone_components"] - res["n_zone_components"])
            diag["eval"] = {k: v for k, v in ev.items() if not k.startswith("_")}
        diagnostics.append(diag)

        # ── 落盘：实例 JSON + 每实例掩膜 PNG ──
        def clean(inst):
            return {k: v for k, v in inst.items() if not k.startswith("_")}

        scale_xy = (IMG_SIZE / w0) * (IMG_SIZE / h0)     # 640 面积 → 原图面积
        flat_instances: list[dict] = []
        for d in res["disks"]:
            it = clean(d)
            it["class_id"] = DISK
            it["disk_index"] = d["id"]
            it["zone_id"] = d["zone_id"]
            it["eq_diameter_px_orig"] = round(
                float(2.0 * np.sqrt(max(d["area_px"] / scale_xy, 0.0) / np.pi)), 3)
            it["eq_diameter_px_640"] = d["eq_diameter_px"]
            flat_instances.append(it)
        for z in res["zones"]:
            it = clean(z)
            it["class_id"] = ZONE
            it["disk_index"] = z["disk_id"]          # 圈实例的 disk_index = 所属药片
            it["zone_id"] = z["id"]
            it["has_zone"] = True
            it["eq_diameter_px_orig"] = round(
                float(2.0 * np.sqrt(max(z["area_px"] / scale_xy, 0.0) / np.pi)), 3)
            it["eq_diameter_px_640"] = z["eq_diameter_px"]
            flat_instances.append(it)

        if args.masks:
            for src_inst in list(res["disks"]) + list(res["zones"]):
                m = src_inst.get("_mask")
                if m is None:
                    continue
                imwrite(out_dir / "masks"
                        / f"{f.stem}__{src_inst['class']}_{src_inst.get('disk_id', src_inst['id'])}.png",
                        np.where(m > 0, 255, 0).astype(np.uint8))

        doc = {
            "image": f.name,
            "stem": f.stem,
            "orig_size": [w0, h0],
            "model_input": [IMG_SIZE, IMG_SIZE],
            "thresh": args.thresh,
            "min_area_px": args.min_area,
            "tta": bool(args.tta),
            "infer_ms": round(ms, 1),
            "n_disks": res["n_disks"],
            "n_zones": res["n_zones"],
            "n_zone_components": res["n_zone_components"],
            "orphan_zone_px": res["orphan_zone_px"],
            "disks": [clean(d) for d in res["disks"]],
            "zones": [clean(z) for z in res["zones"]],
            "pairs": res["pairs"],
            "instances": [{k: v for k, v in it.items() if not k.startswith("_")}
                          for it in flat_instances],
        }
        if args.eval:
            doc["gt_eval"] = {k: v for k, v in ev.items() if not k.startswith("_")}
            doc["gt_counts"] = ev["_gt_counts"]
        (out_dir / "instances" / f"{f.stem}.json").write_text(
            json.dumps(doc, indent=2, ensure_ascii=False), encoding="utf-8")

        all_images.append({
            "image": f.stem,
            "width": w0, "height": h0,
            "ms": round(ms, 1),
            "n_disk": res["n_disks"], "n_zone": res["n_zones"],
            "n_zone_components": res["n_zone_components"],
            "orphan_zone_px": res["orphan_zone_px"],
            "instances": [{k: v for k, v in it.items() if not k.startswith("_")}
                          for it in flat_instances],
        })

        if args.vis:
            vis = draw_instances(img, res)
            if args.eval:
                vis = np.concatenate([vis, draw_gt(img, gti)], axis=1)
            imwrite(out_dir / "vis" / f"{f.stem}.jpg", vis)

        per_image.append({k: v for k, v in diag.items()})

        extra = ""
        if args.eval:
            extra = (f" | GT 药片={ev['_gt_counts']['Yaoping']:2d} 圈={ev['_gt_counts']['Area']:3d}"
                     f" 圈连通域={ev['_gt_zone_components']:2d}")
        print(f"  {f.stem:20s} 药片={res['n_disks']:2d} 圈={res['n_zones']:2d}"
              f" 圈连通域={res['n_zone_components']:2d} 孤儿px={res['orphan_zone_px']:6d}"
              f" 无圈药片={diag['n_disks_no_zone']}{extra}  {ms:.0f}ms")

    # ── 汇总 ──
    summary = {
        "backend": be.backend,
        "thresh": args.thresh, "min_area_px": args.min_area, "tta": bool(args.tta),
        "n_images": len(per_image),
        "infer_ms_median": round(float(np.median(ms_all)), 1) if ms_all else None,
        "instance_metrics": {},
        "association": {"n_evaluable_zones": pair_tot, "n_correct": pair_ok,
                        "accuracy": round(pair_ok / pair_tot, 4) if pair_tot else None},
        "per_image": per_image,
    }
    for cls in (ZONE, DISK):
        a = agg[cls]
        tp = a["tp"]
        prec = tp / (tp + a["fp"]) if (tp + a["fp"]) else 0.0
        rec = tp / (tp + a["fn"]) if (tp + a["fn"]) else 0.0
        summary["instance_metrics"][CLASS_NAMES[cls]] = {
            "n_pred": a["n_pred"], "n_gt": a["n_gt"], "tp": tp,
            "fp": a["fp"], "fn": a["fn"],
            "precision": round(prec, 4), "recall": round(rec, 4),
            "f1": round(2 * prec * rec / (prec + rec), 4) if (prec + rec) else 0.0,
            "mean_iou_matched": round(a["iou_sum"] / a["n_matched"], 4)
            if a["n_matched"] else 0.0,
        }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    if args.dump_all:
        (out_dir / "instances.json").write_text(
            json.dumps(all_images, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"汇总实例：{out_dir / 'instances.json'}（{len(all_images)} 张图）")

    # CSV
    with open(out_dir / "per_image.csv", "w", newline="", encoding="utf-8-sig") as fh:
        wr = csv.writer(fh)
        wr.writerow(["image", "n_disks_pred", "n_zones_pred", "n_zone_components",
                     "gt_disks", "gt_zones", "gt_zone_components",
                     "orphan_zone_px", "tiny_disks_dropped", "disks_no_zone",
                     "gt_components_minus_pred", "Area_tp", "Area_fp", "Area_fn",
                     "Yaoping_tp", "Yaoping_fp", "Yaoping_fn", "infer_ms"])
        for d in diagnostics:
            e = d.get("eval", {})
            gc = d.get("gt_counts", {})
            wr.writerow([d["image"], d["n_disks_pred"], d["n_zones_pred"],
                         d["n_zone_components"], gc.get("Yaoping", ""), gc.get("Area", ""),
                         d.get("gt_zone_components", ""), d["orphan_zone_px"],
                         d["tiny_disks_dropped"], d["n_disks_no_zone"],
                         d.get("gt_zone_components_minus_pred", ""),
                         e.get("Area", {}).get("tp", ""), e.get("Area", {}).get("fp", ""),
                         e.get("Area", {}).get("fn", ""),
                         e.get("Yaoping", {}).get("tp", ""),
                         e.get("Yaoping", {}).get("fp", ""),
                         e.get("Yaoping", {}).get("fn", ""), d["infer_ms"]])

    print("\n" + "=" * 78)
    print("实例级指标（IoU≥0.5 贪心匹配，跨图累计；GT = labels/*.txt 多边形数）")
    for name, v in summary["instance_metrics"].items():
        print(f"  {name:8s} 预测={v['n_pred']:4d}  GT={v['n_gt']:4d}  TP={v['tp']:4d}  "
              f"P={v['precision']:.4f}  R={v['recall']:.4f}  F1={v['f1']:.4f}  "
              f"匹配IoU={v['mean_iou_matched']:.4f}")
    a = summary["association"]
    print(f"  归属正确率: {a['n_correct']}/{a['n_evaluable_zones']} = {a['accuracy']}")
    print(f"  单张耗时中位数: {summary['infer_ms_median']} ms（TTA={summary['tta']}）")
    print(f"汇总: {out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
