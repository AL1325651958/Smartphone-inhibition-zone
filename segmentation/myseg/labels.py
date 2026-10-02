"""
myseg/labels.py — YOLO-seg 标签读写、多边形栅格化、几何测量

YOLO-seg 标签格式（每行一个实例）：
    <class_id> <x1> <y1> <x2> <y2> ... <xn> <yn>
坐标是【归一化】的，范围 0–1。

本项目两类：
    0 = Area     抑菌圈区域
    1 = Yaoping  药片（6 mm 标准纸片，用作内标）
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

NORM_EPS = 1e-6


# ────────────────────────────────────────────────
#  读
# ────────────────────────────────────────────────
def read_label(path: Path, img_w: int, img_h: int) -> list[tuple[int, np.ndarray]]:
    """读取标签，返回 [(class_id, points_px[N,2]), ...]。像素坐标。"""
    out: list[tuple[int, np.ndarray]] = []
    if not Path(path).is_file():
        return out
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 7:                     # class + 至少 3 个点
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


def read_label_norm(path: Path) -> list[tuple[int, np.ndarray]]:
    """读取标签，返回 [(class_id, points_norm[N,2])]，坐标保持归一化。"""
    out: list[tuple[int, np.ndarray]] = []
    if not Path(path).is_file():
        return out
    with open(path, encoding="utf-8") as f:
        for line in f:
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
            out.append((cls, coords.reshape(-1, 2)))
    return out


# ────────────────────────────────────────────────
#  写
# ────────────────────────────────────────────────
def write_label(path: Path, instances: list[tuple[int, np.ndarray]],
                img_w: int, img_h: int) -> None:
    """把像素坐标多边形写成 YOLO-seg 标签（归一化，裁剪到 [0,1]）。"""
    lines: list[str] = []
    for cls, pts in instances:
        if pts is None or len(pts) < 3:
            continue
        n = pts.astype(np.float64).copy()
        n[:, 0] = np.clip(n[:, 0] / max(img_w, 1), 0.0, 1.0)
        n[:, 1] = np.clip(n[:, 1] / max(img_h, 1), 0.0, 1.0)
        flat = " ".join(f"{v:.6f}" for v in n.reshape(-1))
        lines.append(f"{int(cls)} {flat}")
    Path(path).write_text("\n".join(lines), encoding="utf-8")


# ────────────────────────────────────────────────
#  栅格化
# ────────────────────────────────────────────────
def rasterize(instances: list[tuple[int, np.ndarray]], img_w: int, img_h: int,
              num_classes: int = 2) -> np.ndarray:
    """
    把多边形实例栅格化成 (num_classes, H, W) 的 0/1 float32 mask。
    同类多实例会合并（本项目每类语义唯一，实例后面靠连通域拆开）。
    """
    mask = np.zeros((num_classes, img_h, img_w), dtype=np.float32)
    for cls, pts in instances:
        if not (0 <= cls < num_classes) or len(pts) < 3:
            continue
        poly = np.round(pts).astype(np.int32)
        cv2.fillPoly(mask[cls], [poly], 1.0)
    return mask


# ────────────────────────────────────────────────
#  几何测量（从 mask 得到实际直径）
# ────────────────────────────────────────────────
def equivalent_diameter(area_px: float) -> float:
    """等效圆直径：面积相同的圆直径 = 2*sqrt(A/pi)。"""
    return float(2.0 * np.sqrt(max(area_px, 0.0) / np.pi))


def components(mask: np.ndarray, min_area: int = 30) -> list[dict]:
    """
    对单通道 0/1 mask 做连通域分析，返回每个连通域的几何量。
    面积小于 min_area 的丢弃。

    每个元素同时保留
      component_mask : 该连通域在自身 bbox 内的 0/1 掩膜（供局部扇区射线采样用）
      mask_h / mask_w: 原 mask 的尺寸
    """
    m = (mask > 0.5).astype(np.uint8)
    mask_h, mask_w = m.shape[:2]
    n, lab, stats, cents = cv2.connectedComponentsWithStats(m, connectivity=8)
    out: list[dict] = []
    for i in range(1, n):                          # 0 是背景
        area = float(stats[i, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        x, y = int(stats[i, cv2.CC_STAT_LEFT]), int(stats[i, cv2.CC_STAT_TOP])
        w, h = int(stats[i, cv2.CC_STAT_WIDTH]), int(stats[i, cv2.CC_STAT_HEIGHT])
        comp = (lab[y:y + h, x:x + w] == i).astype(np.uint8)
        cnts, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not cnts:
            continue
        c = max(cnts, key=cv2.contourArea) + np.array([x, y], dtype=np.int32)
        (cx, cy), r = cv2.minEnclosingCircle(c.astype(np.float32))
        out.append({
            "area_px": area,
            "eq_diameter_px": equivalent_diameter(area),
            "centroid": (float(cents[i][0]), float(cents[i][1])),
            "enc_circle": (float(cx), float(cy), float(r)),
            "bbox": (x, y, w, h),
            "contour": c,
            "component_mask": comp,
            "mask_h": mask_h,
            "mask_w": mask_w,
        })
    out.sort(key=lambda d: -d["area_px"])
    return out


def estimate_plate_center(disks: list[dict]) -> tuple[float, float] | None:
    """
    估计培养皿几何中心：取全部药片质心的中位数（对个别漏检/误检稳健）。
    论文原话：“the antibiotic-disk masks and their spatial distribution were first
    used to estimate the geometric centre of the Petri dish”。
    """
    if not disks:
        return None
    pts = np.array([d["centroid"] for d in disks], dtype=np.float64)
    return float(np.median(pts[:, 0])), float(np.median(pts[:, 1]))


def _sector_outer_radius(zone_mask: np.ndarray, cx: float, cy: float,
                         angle: float, r_start: float, r_max: float,
                         half_angle_deg: float, n_rays: int) -> np.ndarray:
    """
    在以 angle 为中心、±half_angle_deg 的扇区内，从 (cx,cy) 向外发射 n_rays 条射线，
    返回每条射线打到 zone_mask 的【最外缘】距离（未命中的记为 nan）。

    论文：“Radial intersections with the disk and inhibition-zone masks were sampled
    within this sector.”
    """
    h, w = zone_mask.shape[:2]
    half = np.deg2rad(half_angle_deg)
    angles = np.linspace(angle - half, angle + half, n_rays)
    radii = np.arange(max(r_start, 1.0), r_max, 1.0)
    if radii.size == 0:
        return np.full(n_rays, np.nan)

    ca, sa = np.cos(angles), np.sin(angles)
    xs = np.rint(cx + np.outer(ca, radii)).astype(np.int32)
    ys = np.rint(cy + np.outer(sa, radii)).astype(np.int32)
    inb = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
    hits = np.where(inb, zone_mask[np.clip(ys, 0, h - 1), np.clip(xs, 0, w - 1)], 0)

    out = np.full(n_rays, np.nan)
    for k in range(n_rays):
        idx = np.flatnonzero(hits[k] > 0)
        if idx.size:
            out[k] = radii[idx[-1]]                     # 该方向的外缘距离
    return out


def measure_zone_diameters(disk_mask: np.ndarray, zone_mask: np.ndarray,
                           min_area: int = 30, half_angle_deg: float = 15.0,
                           n_rays: int = 41) -> dict:
    """
    按论文定义测量每个药片的抑菌圈直径（像素）。

    论文原文（Manuscript “Calibration, Measurement, and Application Workflow”）：
      “For each disk--zone pair, a local sector with an angular range of ±15° was
       defined along the radial direction between the disk centre and the estimated
       centre of the Petri dish. Radial intersections with the disk and
       inhibition-zone masks were sampled within this sector. Representative pixel
       diameters were calculated as the mean across the sampled directions.”

    关键点：扇区方向是「皿心 → 药片」，即采样的是**朝皿外那一侧**的边界。
    这正好绕开相邻抑菌圈的干扰（朝内的射线会打到邻片的圈或融合边界上，导致读数偏大）。

    直径定义：D_zone = 2 × mean(扇区内各射线从药片中心到圈外缘的距离)

    返回
    ----
    {
      'plate_center': (px, py) 或 None,
      'disks': [ {disk, zone_diameter_px, n_rays_used, angle_deg, ...}, ... ],
      'n_disks': int, 'n_zones': int, 'n_measured': int,
    }
    """
    disks = components(disk_mask, min_area=min_area)
    zones = components(zone_mask, min_area=min_area)
    center = estimate_plate_center(disks)

    items: list[dict] = []
    if center is None or not zones:
        for d in disks:
            items.append({"disk": d, "zone_diameter_px": None, "n_rays_used": 0,
                          "angle_deg": None, "r_outer_px": None})
        return {"plate_center": center, "disks": items, "n_disks": len(disks),
                "n_zones": len(zones), "n_measured": 0}

    pcx, pcy = center
    h, w = zone_mask.shape[:2]
    r_max = float(np.hypot(w, h))                       # 足够覆盖整图

    for d in disks:
        dcx, dcy = d["centroid"]
        r_disk = d["eq_diameter_px"] / 2.0

        # 扇区方向 = 皿心 → 药片
        vx, vy = dcx - pcx, dcy - pcy
        if abs(vx) < 1e-9 and abs(vy) < 1e-9:
            items.append({"disk": d, "zone_diameter_px": None, "n_rays_used": 0,
                          "angle_deg": None, "r_outer_px": None})
            continue
        angle = float(np.arctan2(vy, vx))

        # 药片中心必须在某个圈内，否则视为无圈（耐药 / 未标注）
        inside = zone_mask[int(round(dcy)), int(round(dcx))] > 0 \
            if 0 <= int(round(dcy)) < h and 0 <= int(round(dcx)) < w else False
        if not inside:
            items.append({"disk": d, "zone_diameter_px": None, "n_rays_used": 0,
                          "angle_deg": float(np.rad2deg(angle)), "r_outer_px": None})
            continue

        outer = _sector_outer_radius(zone_mask, dcx, dcy, angle,
                                     r_start=r_disk, r_max=r_max,
                                     half_angle_deg=half_angle_deg, n_rays=n_rays)
        good = outer[~np.isnan(outer)]
        good = good[good > r_disk * 0.95]               # 越过药片本体的才算圈

        if good.size < 3:
            items.append({"disk": d, "zone_diameter_px": None, "n_rays_used": 0,
                          "angle_deg": float(np.rad2deg(angle)), "r_outer_px": None})
            continue

        r_outer = float(np.mean(good))                  # 论文：取扇区内方向的均值
        items.append({
            "disk": d,
            "zone_diameter_px": float(2.0 * r_outer),   # 直径 = 2 × 外缘半径
            "r_outer_px": r_outer,
            "n_rays_used": int(good.size),
            "angle_deg": float(np.rad2deg(angle)),
        })

    measured = [it for it in items if it["zone_diameter_px"] is not None]
    return {"plate_center": center, "disks": items, "n_disks": len(disks),
            "n_zones": len(zones), "n_measured": len(measured)}
