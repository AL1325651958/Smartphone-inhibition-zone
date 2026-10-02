"""
myseg/visualize.py — 结果可视化

save_overlay : 把预测 mask 与人工 mask 叠在原图上，三联对照
plot_curves  : 训练曲线（loss / Dice / IoU / 直径 MAE）
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

try:
    from myseg import config as C
    from myseg.io_utils import imwrite
except ImportError:                                     # 允许直接运行
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from myseg import config as C
    from myseg.io_utils import imwrite

# BGR
COL_ZONE = (117, 158, 29)      # 绿  抑菌圈
COL_DISK = (165, 95, 24)       # 蓝  药片


def _denorm(x: np.ndarray) -> np.ndarray:
    """(3,H,W) 归一化张量 → (H,W,3) uint8 BGR。"""
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    mean = np.asarray(C.IMAGENET_MEAN, dtype=np.float32).reshape(3, 1, 1)
    std = np.asarray(C.IMAGENET_STD, dtype=np.float32).reshape(3, 1, 1)
    img = x * std + mean
    img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    return cv2.cvtColor(np.transpose(img, (1, 2, 0)), cv2.COLOR_RGB2BGR)


def _draw_mask(base: np.ndarray, mask: np.ndarray, thresh: float = C.MASK_THRESH,
               fill_alpha: float = 0.25) -> np.ndarray:
    """把 (K,H,W) 概率图描边+半透明填充到 base 上。"""
    vis = base.copy()
    overlay = vis.copy()
    for c, color in ((0, COL_ZONE), (1, COL_DISK)):
        if c >= mask.shape[0]:
            continue
        m = (mask[c] > thresh).astype(np.uint8)
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            continue
        cv2.drawContours(overlay, cnts, -1, color, thickness=cv2.FILLED)
        cv2.drawContours(vis, cnts, -1, color, 3)
    return cv2.addWeighted(overlay, fill_alpha, vis, 1 - fill_alpha, 0)


def save_overlay(img_tensor, pred_prob: np.ndarray, gt_mask: np.ndarray, out_path: Path,
                 title: str | None = None) -> None:
    """
    三联图：原图 | 人工标注 | 模型预测
    img_tensor : (3,H,W) 归一化张量，或 (H,W,3) uint8 BGR
    pred_prob  : (K,H,W) 概率
    gt_mask    : (K,H,W) 0/1
    """
    base = _denorm(img_tensor) if not isinstance(img_tensor, np.ndarray) \
        else img_tensor.copy()

    panels = [
        base,
        _draw_mask(base, gt_mask.astype(np.float32), thresh=0.5),
        _draw_mask(base, pred_prob),
    ]
    labels = ["original", "ground truth", "prediction"]
    out = []
    for p, lab in zip(panels, labels):
        p = p.copy()
        cv2.rectangle(p, (0, 0), (p.shape[1], 46), (25, 25, 25), -1)
        cv2.putText(p, lab, (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.85,
                    (255, 255, 255), 2, cv2.LINE_AA)
        out.append(p.astype(np.float32))
    grid = np.hstack(out).astype(np.uint8)

    if title:
        cv2.putText(grid, title, (12, grid.shape[0] - 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    imwrite(out_path, grid)


def plot_curves(history: list[dict], out_path: Path) -> None:
    """训练曲线。matplotlib 失败时静默跳过（不影响训练主流程）。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ep = [h["epoch"] for h in history]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    axes[0, 0].plot(ep, [h["train_loss"] for h in history], color="#c0392b")
    axes[0, 0].set_title("train loss"); axes[0, 0].set_xlabel("epoch")

    axes[0, 1].plot(ep, [h["val_dice"] for h in history], label="Dice", color="#27ae60")
    axes[0, 1].plot(ep, [h["val_iou"] for h in history], label="IoU", color="#2980b9")
    axes[0, 1].set_title("validation (mean over classes)")
    axes[0, 1].set_xlabel("epoch"); axes[0, 1].legend()

    axes[1, 0].plot(ep, [h["val_zone_MAE"] for h in history], color="#8e44ad")
    axes[1, 0].set_title("validation zone diameter MAE (mm)")
    axes[1, 0].set_xlabel("epoch")

    names = list(history[0]["val_pixel"].keys())
    for i, n in enumerate(names):
        axes[1, 1].plot(ep, [h["val_pixel"][n]["IoU"] for h in history], label=n)
    axes[1, 1].set_title("per-class IoU"); axes[1, 1].set_xlabel("epoch")
    axes[1, 1].legend()

    for ax in axes.ravel():
        ax.grid(alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
