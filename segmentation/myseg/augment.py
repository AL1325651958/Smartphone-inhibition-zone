"""
myseg/augment.py — 训练集 / 验证集数据增强（纯 OpenCV + NumPy，无外部增强库）

按需求：**训练集和验证集都做增强，测试集保持原图不动**。

为什么不用 albumentations
-------------------------
它的依赖链里有 C 扩展（pydantic-core / stringzilla），在本机下载极慢；
而且这类库在 Windows 非 ASCII 路径上有踩坑风险（本项目路径含「抑菌区域分析_返修」）。
本模块的增强逻辑清晰可控，自己写反而更稳、更快、无依赖。

几何一致性保证
--------------
所有几何变换都用「一个 2×3（或 3×3）矩阵」同时作用到图片和标注点上：
  图片：cv2.warpAffine / cv2.warpPerspective
  点  ：cv2.transform / cv2.perspectiveTransform（同一矩阵）
这样多边形与像素**严格对齐**，不会出现标注漂移。

设计要点
--------
1. 增强在「原图分辨率」上做，多边形随之同步变换 —— 本项目最终要输出像素距离，
   在原分辨率变换不引入额外的坐标换算误差。
2. 边界用 BORDER_REPLICATE 而不是黑色：黑边会被模型当成「圈外」特征，有害。
3. 光度类增强（亮度/HSV/模糊/噪声/CLAHE）只改像素，点不动。
4. 每个样本生成后做合法性检查：多边形点数够、面积不为 0、不占满整图。

用法
----
    python -m myseg.augment                    # 增强 train 和 val
    python -m myseg.augment --splits train     # 只增强训练集
    python -m myseg.augment --preview          # 只生成一张对齐检查图
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from myseg import config as C
    from myseg import labels as L
    from myseg.io_utils import imread, imwrite
else:
    from myseg import config as C
    from myseg import labels as L
    from myseg.io_utils import imread, imwrite


# ────────────────────────────────────────────────
#  几何：同一个矩阵作用于图片与点
# ────────────────────────────────────────────────
def _affine_matrix(w: int, h: int, rng: np.random.Generator) -> np.ndarray:
    """绕图像中心旋转 + 缩放 + 平移，返回 2×3 矩阵。"""
    angle = rng.uniform(-180.0, 180.0)
    scale = rng.uniform(0.90, 1.10)
    tx = rng.uniform(-0.05, 0.05) * w
    ty = rng.uniform(-0.05, 0.05) * h
    m = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, scale)
    m[0, 2] += tx
    m[1, 2] += ty
    return m


def _perspective_matrix(w: int, h: int, rng: np.random.Generator,
                        strength: float = 0.04) -> np.ndarray:
    """轻微透视变换（模拟拍摄角度偏差），返回 3×3 矩阵。"""
    s = strength * min(w, h)
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst = src + rng.uniform(-s, s, size=(4, 2)).astype(np.float32)
    return cv2.getPerspectiveTransform(src, dst)


def _apply_affine(img: np.ndarray, kpts: list[np.ndarray], m: np.ndarray):
    h, w = img.shape[:2]
    out = cv2.warpAffine(img, m, (w, h), flags=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_REPLICATE)
    new = []
    for p in kpts:
        if len(p) == 0:
            new.append(p)
            continue
        q = cv2.transform(p.reshape(-1, 1, 2).astype(np.float32), m).reshape(-1, 2)
        new.append(q)
    return out, new


def _apply_perspective(img: np.ndarray, kpts: list[np.ndarray], m: np.ndarray):
    h, w = img.shape[:2]
    out = cv2.warpPerspective(img, m, (w, h), flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_REPLICATE)
    new = []
    for p in kpts:
        if len(p) == 0:
            new.append(p)
            continue
        q = cv2.perspectiveTransform(p.reshape(-1, 1, 2).astype(np.float32), m).reshape(-1, 2)
        new.append(q)
    return out, new


# ────────────────────────────────────────────────
#  光度
# ────────────────────────────────────────────────
def _photometric(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    out = img.astype(np.float32)

    if rng.random() < 0.8:                                  # 亮度 / 对比度
        out = out * (1.0 + rng.uniform(-0.20, 0.20)) + rng.uniform(-25.0, 25.0)

    if rng.random() < 0.5:                                  # 色调 / 饱和度 / 明度
        hsv = cv2.cvtColor(np.clip(out, 0, 255).astype(np.uint8), cv2.COLOR_BGR2HSV)
        hsv = hsv.astype(np.int16)
        hsv[..., 0] = (hsv[..., 0] + int(rng.integers(-5, 6))) % 180
        hsv[..., 1] = np.clip(hsv[..., 1] * (1.0 + rng.uniform(-0.30, 0.30)), 0, 255)
        hsv[..., 2] = np.clip(hsv[..., 2] * (1.0 + rng.uniform(-0.20, 0.20)), 0, 255)
        out = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR).astype(np.float32)

    if rng.random() < 0.2:                                  # 轻微模糊
        k = int(rng.choice([3, 5]))
        out = cv2.GaussianBlur(out, (k, k), 0)

    if rng.random() < 0.2:                                  # 噪声
        out = out + rng.normal(0.0, rng.uniform(2.0, 8.0), out.shape).astype(np.float32)

    if rng.random() < 0.2:                                  # CLAHE（L 通道）
        lab = cv2.cvtColor(np.clip(out, 0, 255).astype(np.uint8), cv2.COLOR_BGR2LAB)
        lab[..., 0] = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(lab[..., 0])
        out = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR).astype(np.float32)

    return np.clip(out, 0, 255).astype(np.uint8)


# ────────────────────────────────────────────────
#  一次增强
# ────────────────────────────────────────────────
def augment_once(image: np.ndarray, instances: list[tuple[int, np.ndarray]],
                 rng: np.random.Generator
                 ) -> tuple[np.ndarray, list[tuple[int, np.ndarray]]] | None:
    """执行一次增强。退化（点被挤没 / 面积异常）返回 None。"""
    h, w = image.shape[:2]
    cls_list = [c for c, _ in instances]
    kpts = [p.astype(np.float32).copy() for _, p in instances]

    img = image
    if rng.random() < 0.5:                                  # 水平翻转
        img = cv2.flip(img, 1)
        for p in kpts:
            p[:, 0] = (w - 1) - p[:, 0]
    if rng.random() < 0.5:                                  # 垂直翻转
        img = cv2.flip(img, 0)
        for p in kpts:
            p[:, 1] = (h - 1) - p[:, 1]
    if rng.random() < 0.5:                                  # 转 90°（圆对称）
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        for p in kpts:
            x, y = p[:, 0].copy(), p[:, 1].copy()
            p[:, 0] = (h - 1) - y
            p[:, 1] = x
        h, w = img.shape[:2]

    if rng.random() < 0.85:                                 # 仿射
        img, kpts = _apply_affine(img, kpts, _affine_matrix(w, h, rng))

    if rng.random() < 0.30:                                 # 透视
        img, kpts = _apply_perspective(img, kpts, _perspective_matrix(w, h, rng))

    img = _photometric(img, rng)

    ha, wa = img.shape[:2]
    out: list[tuple[int, np.ndarray]] = []
    for cls, p in zip(cls_list, kpts):
        if len(p) < 3:
            continue
        q = np.clip(p, [0.0, 0.0], [float(wa - 1), float(ha - 1)]).astype(np.float32)
        area = float(cv2.contourArea(q.reshape(-1, 1, 2)))
        if area <= 1.0 or area / float(wa * ha) > 0.98:     # 退化 / 占满整图
            return None
        out.append((cls, q))

    return (img, out) if out else None


def build_pipeline(seed: int = C.AUG_SEED) -> np.random.Generator:
    """兼容旧接口：本模块用随机数生成器代替 pipeline 对象。"""
    return np.random.default_rng(seed)


# ────────────────────────────────────────────────
#  预览：确认多边形跟着图一起动
# ────────────────────────────────────────────────
def preview(seed: int = C.AUG_SEED) -> Path:
    src_img_dir = C.DATA / "train" / "images"
    src_lbl_dir = C.DATA / "train" / "labels"
    stems = sorted(p.stem for p in src_img_dir.glob("*.jpg"))
    if not stems:
        raise SystemExit("训练集为空，先运行 python -m myseg.split_dataset")

    rng = np.random.default_rng(seed)

    first = imread(src_img_dir / f"{stems[0]}.jpg")
    h, w = first.shape[:2]
    panels = [("original", first, L.read_label(src_lbl_dir / f"{stems[0]}.txt", w, h))]

    for i in range(3):
        got = None
        for _ in range(30):
            stem = stems[int(rng.integers(len(stems)))]
            img = imread(src_img_dir / f"{stem}.jpg")
            hh, ww = img.shape[:2]
            inst = L.read_label(src_lbl_dir / f"{stem}.txt", ww, hh)
            got = augment_once(img, inst, rng)
            if got is not None:
                break
        if got is not None:
            panels.append((f"aug{i + 1}", got[0], got[1]))

    target_h = 520
    tiles = []
    for name, img, inst in panels:
        vis = img.copy()
        for cls, pts in inst:
            color = (117, 158, 29) if cls == 0 else (165, 95, 24)     # BGR
            cv2.polylines(vis, [np.round(pts).astype(np.int32)], True, color, 4)
            cx, cy = pts.mean(axis=0)
            cv2.putText(vis, C.CLASS_NAMES[cls], (int(cx) - 40, int(cy)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)
        s = target_h / vis.shape[0]
        vis = cv2.resize(vis, (int(vis.shape[1] * s), target_h))
        cv2.rectangle(vis, (0, 0), (vis.shape[1], 46), (25, 25, 25), -1)
        cv2.putText(vis, name, (12, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                    (255, 255, 255), 2, cv2.LINE_AA)
        tiles.append(vis)

    max_w = max(t.shape[1] for t in tiles)
    tiles = [cv2.copyMakeBorder(t, 0, 0, 0, max_w - t.shape[1],
                                cv2.BORDER_CONSTANT, value=(30, 30, 30)) for t in tiles]
    grid = np.vstack([np.hstack(tiles[:2]), np.hstack(tiles[2:4])])

    out = C.RUNS / "augment_preview.jpg"
    out.parent.mkdir(parents=True, exist_ok=True)
    imwrite(out, grid, quality=92)
    return out


# ────────────────────────────────────────────────
#  批量增强
# ────────────────────────────────────────────────
def build_split(split: str, per_image: int, seed: int = C.AUG_SEED) -> dict:
    src_img_dir = C.DATA / split / "images"
    src_lbl_dir = C.DATA / split / "labels"
    dst_img_dir = C.DATA / f"{split}_aug" / "images"
    dst_lbl_dir = C.DATA / f"{split}_aug" / "labels"
    dst_img_dir.mkdir(parents=True, exist_ok=True)
    dst_lbl_dir.mkdir(parents=True, exist_ok=True)

    for d in (dst_img_dir, dst_lbl_dir):                    # 清掉上一轮产物
        for f in d.iterdir():
            f.unlink()

    rng = np.random.default_rng(seed)
    stems = sorted(p.stem for p in src_img_dir.glob("*.jpg"))
    stats = {"copied": 0, "augmented": 0, "failed": 0}

    for stem in stems:
        img = imread(src_img_dir / f"{stem}.jpg")
        if img is None:
            print(f"  无法读取 {stem}.jpg")
            continue
        h, w = img.shape[:2]
        instances = L.read_label(src_lbl_dir / f"{stem}.txt", w, h)

        imwrite(dst_img_dir / f"{stem}.jpg", img, quality=95)       # 原图照抄
        L.write_label(dst_lbl_dir / f"{stem}.txt", instances, w, h)
        stats["copied"] += 1

        ok, tries = 0, 0
        while ok < per_image and tries < per_image * 8:
            tries += 1
            got = augment_once(img, instances, rng)
            if got is None:
                stats["failed"] += 1
                continue
            aimg, ainst = got
            ah, aw = aimg.shape[:2]
            name = f"{stem}_aug{ok:03d}"
            imwrite(dst_img_dir / f"{name}.jpg", aimg, quality=95)
            L.write_label(dst_lbl_dir / f"{name}.txt", ainst, aw, ah)
            ok += 1
            stats["augmented"] += 1

        if ok < per_image:
            print(f"  {stem}: 只生成 {ok}/{per_image}")

    return stats


def main() -> None:
    ap = argparse.ArgumentParser(description="训练/验证集数据增强（纯 OpenCV）")
    ap.add_argument("--splits", nargs="+", default=["train", "val"],
                    choices=["train", "val"], help="要增强哪些划分（test 一律不动）")
    ap.add_argument("--per-image", type=int, default=C.AUG_PER_IMAGE)
    ap.add_argument("--seed", type=int, default=C.AUG_SEED)
    ap.add_argument("--preview", action="store_true", help="只生成对齐检查图")
    args = ap.parse_args()

    if args.preview:
        out = preview(args.seed)
        print(f"对齐检查图已保存：{out}")
        print("请打开确认多边形与图中抑菌圈/药片对齐。")
        return

    print(f"每张原图生成 {args.per_image} 张增强图（另保留 1 张原图）\n")
    total = 0
    for split in args.splits:
        print(f"[{split}]")
        st = build_split(split, args.per_image, args.seed)
        n = len(list((C.DATA / f"{split}_aug" / "images").glob("*.jpg")))
        print(f"  原图 {st['copied']} + 增强 {st['augmented']} = 共 {n} 张"
              f"（退化丢弃 {st['failed']} 次）")
        total += n

    n_test = len(list((C.DATA / "test" / "images").glob("*.jpg")))
    print(f"\n测试集保持原图不动：{n_test} 张")
    print(f"训练+验证合计可用 {total} 张")
    print("\n建议先跑 python -m myseg.augment --preview 看一眼对齐。")


if __name__ == "__main__":
    main()
