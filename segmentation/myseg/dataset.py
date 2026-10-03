"""
myseg/dataset.py — PyTorch Dataset

从 data/<split>/{images,labels} 读图与多边形分割标签，栅格化成
(num_classes, IMG_SIZE, IMG_SIZE) 的多通道 mask。

坐标约定
--------
- 原图 H0×W0 上的多边形（像素）→ 乘以 IMG_SIZE/H0 与 IMG_SIZE/W0 直接缩放。
- 归一化标签不变，所以本模块在【归一化坐标】上做缩放最省事：
    px_x = norm_x * W0  →  norm_x 不变  →  网络输出坐标下的 px_x' = norm_x * IMG_SIZE
  也就是说直接把归一化坐标乘 IMG_SIZE 即可，与 mask 的 resize 完全一致。
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

try:
    from myseg import config as C
    from myseg import labels as L
    from myseg.io_utils import imread
except ImportError:                                     # 允许直接运行
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from myseg import config as C
    from myseg import labels as L
    from myseg.io_utils import imread


class SegDataset(Dataset):
    """
    参数
    ----
    split : 'train' | 'val' | 'test'
        - train / val 默认读增强目录 data/<split>_aug
        - test 读原图目录 data/test
    use_aug : bool
        是否优先使用增强目录（目录不存在时自动回退到原图）
    img_size : int
    augment_online : bool
        是否再做一次在线增强（离线增强之外的可选补充，默认关闭以避免双重增强）
    """

    def __init__(self, split: str, use_aug: bool = True, img_size: int = C.IMG_SIZE,
                 augment_online: bool = False):
        self.split = split
        self.img_size = img_size
        self.augment_online = augment_online

        aug_dir = C.DATA / f"{split}_aug"
        raw_dir = C.DATA / split
        if use_aug and (aug_dir / "images").is_dir() and any((aug_dir / "images").glob("*.jpg")):
            self.img_dir = aug_dir / "images"
            self.lbl_dir = aug_dir / "labels"
            self.source = f"{split}_aug"
        else:
            self.img_dir = raw_dir / "images"
            self.lbl_dir = raw_dir / "labels"
            self.source = split

        self.files = sorted(self.img_dir.glob("*.jpg"))
        if not self.files:
            raise SystemExit(
                f"{self.img_dir} 里没有图片。先运行：\n"
                f"  python -m myseg.split_dataset\n"
                f"  python -m myseg.augment        （train/val 增强）"
            )

        # 在线增强（可选）
        self.online = None
        if augment_online:
            from myseg.augment import build_pipeline
            self.online = build_pipeline()

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int):
        img_path = self.files[idx]
        img = imread(img_path)
        if img is None:
            raise RuntimeError(f"无法读取 {img_path}")
        h0, w0 = img.shape[:2]

        instances = L.read_label(self.lbl_dir / f"{img_path.stem}.txt", w0, h0)

        if self.online is not None:
            from myseg.augment import augment_once
            got = augment_once(img, instances, self.online)
            if got is not None:
                img, instances = got

        # 统一到 img_size × img_size（按各自轴缩放；归一化坐标因此保持不变）
        img_r = cv2.resize(img, (self.img_size, self.img_size),
                           interpolation=cv2.INTER_LINEAR)

        mask = np.zeros((C.NUM_CLASSES, self.img_size, self.img_size), dtype=np.float32)
        sx = self.img_size / max(w0, 1)
        sy = self.img_size / max(h0, 1)
        for cls, pts in instances:
            if not (0 <= cls < C.NUM_CLASSES) or len(pts) < 3:
                continue
            p = pts.astype(np.float32).copy()
            p[:, 0] *= sx
            p[:, 1] *= sy
            cv2.fillPoly(mask[cls], [np.round(p).astype(np.int32)], 1.0)

        # BGR → RGB → CHW → 归一化
        rgb = cv2.cvtColor(img_r, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        mean = np.asarray(C.IMAGENET_MEAN, dtype=np.float32).reshape(3, 1, 1)
        std = np.asarray(C.IMAGENET_STD, dtype=np.float32).reshape(3, 1, 1)
        x = np.transpose(rgb, (2, 0, 1))
        x = (x - mean) / std

        return {
            "image": torch.from_numpy(np.ascontiguousarray(x)),
            "mask": torch.from_numpy(mask),
            "name": img_path.stem,
            "size": (h0, w0),
        }


def class_stats(dataset: SegDataset) -> dict:
    """统计各 split 的实例数与像素占比，用于检查类别不平衡。"""
    inst_cnt = {c: 0 for c in range(C.NUM_CLASSES)}
    px_cnt = {c: 0 for c in range(C.NUM_CLASSES)}
    total_px = 0
    for f in dataset.files:
        img = imread(f)
        h0, w0 = img.shape[:2]
        inst = L.read_label(dataset.lbl_dir / f"{f.stem}.txt", w0, h0)
        m = L.rasterize(inst, w0, h0, C.NUM_CLASSES)
        total_px += h0 * w0
        for c in range(C.NUM_CLASSES):
            inst_cnt[c] += sum(1 for cls, _ in inst if cls == c)
            px_cnt[c] += int(m[c].sum())
    return {
        "n_images": len(dataset.files),
        "instances": inst_cnt,
        "pixel_frac": {c: px_cnt[c] / max(total_px, 1) for c in range(C.NUM_CLASSES)},
    }
