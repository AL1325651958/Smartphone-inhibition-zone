"""
semseg.dataset — Dataset：图 → 图像 + 两类掩膜

目标只有两个通道：
    semantic[0] = 抑菌圈区域（该类全部多边形的并集）
    semantic[1] = 药片      （同上）

每个多边形单独栅格化再取并集 —— 这也是唯一正确的做法：
类别掩膜只取决于该类自己的多边形，与「药片↔抑菌圈怎么配对」无关。
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from semseg import config as C
    from semseg.io_utils import imread
else:
    from semseg import config as C
    from semseg.io_utils import imread


# ────────────────────────────────────────────────
#  标签读写
# ────────────────────────────────────────────────
def read_polygon_seg(path: Path, img_w: int, img_h: int) -> list[tuple[int, np.ndarray]]:
    """读多边形分割标签，返回 [(class_id, points_px[N,2]), ...]。"""
    out: list[tuple[int, np.ndarray]] = []
    p = Path(path)
    if not p.is_file():
        return out
    with open(p, encoding="utf-8") as f:
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
            pts = coords.reshape(-1, 2)
            pts[:, 0] *= img_w
            pts[:, 1] *= img_h
            out.append((cls, pts))
    return out


def write_polygon_seg(path: Path, instances: list[tuple[int, np.ndarray]],
                   img_w: int, img_h: int) -> None:
    lines = []
    for cls, pts in instances:
        if pts is None or len(pts) < 3:
            continue
        n = pts.astype(np.float64).copy()
        n[:, 0] = np.clip(n[:, 0] / max(img_w, 1), 0.0, 1.0)
        n[:, 1] = np.clip(n[:, 1] / max(img_h, 1), 0.0, 1.0)
        lines.append(f"{int(cls)} " + " ".join(f"{v:.6f}" for v in n.reshape(-1)))
    Path(path).write_text("\n".join(lines), encoding="utf-8")


def build_semantic(instances, w: int, h: int, out_size: int) -> np.ndarray:
    """把两类多边形分别栅格化到 out_size 尺度，返回 (2, out_size, out_size) 的 0/1。"""
    full = np.zeros((C.NUM_CLASSES, h, w), np.uint8)
    for cls, pts in instances:
        if not (0 <= cls < C.NUM_CLASSES) or len(pts) < 3:
            continue
        cv2.fillPoly(full[cls], [np.round(pts).astype(np.int32)], 1)
    if (w, h) == (out_size, out_size):
        return full.astype(np.float32)
    return np.stack([
        cv2.resize(full[c], (out_size, out_size), interpolation=cv2.INTER_NEAREST)
        for c in range(C.NUM_CLASSES)
    ]).astype(np.float32)


def _worker_init(_worker_id):
    """每个 DataLoader worker 把 OpenCV 线程压到 1，避免多 worker 互相争抢。"""
    cv2.setNumThreads(1)


# ────────────────────────────────────────────────
#  Dataset
# ────────────────────────────────────────────────
class SegDataset(Dataset):
    """
    split  : 'train' | 'val' | 'test'
    use_aug: 是否用增强目录 data/<split>_aug（不存在则自动回退原图）
    """

    def __init__(self, split: str, use_aug: bool = True, img_size: int = C.IMG_SIZE):
        self.split = split
        self.img_size = img_size
        aug = C.DATA / f"{split}_aug"
        raw = C.DATA / split
        if use_aug and (aug / "images").is_dir() and any((aug / "images").glob("*.jpg")):
            self.img_dir, self.lbl_dir, self.source = (
                aug / "images", aug / "labels", f"{split}_aug")
        else:
            self.img_dir, self.lbl_dir, self.source = (
                raw / "images", raw / "labels", split)
        self.files = sorted(self.img_dir.glob("*.jpg"))
        if not self.files:
            raise SystemExit(
                f"{self.img_dir} 没有图片。先运行：\n"
                f"  python -m semseg.prepare       （划分 + 增强）")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int):
        p = self.files[idx]
        img = imread(p)
        if img is None:
            raise RuntimeError(f"无法读取 {p}")
        h0, w0 = img.shape[:2]
        inst = read_polygon_seg(self.lbl_dir / f"{p.stem}.txt", w0, h0)
        sem = build_semantic(inst, w0, h0, self.img_size)

        img_r = cv2.resize(img, (self.img_size, self.img_size),
                           interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(img_r, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        mean = np.asarray(C.IMAGENET_MEAN, dtype=np.float32).reshape(3, 1, 1)
        std = np.asarray(C.IMAGENET_STD, dtype=np.float32).reshape(3, 1, 1)
        x = (np.transpose(rgb, (2, 0, 1)) - mean) / std

        return {
            "image": torch.from_numpy(np.ascontiguousarray(x)),
            "semantic": torch.from_numpy(sem),
            "name": p.stem,
        }
