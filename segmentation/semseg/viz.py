"""
semseg.viz — 掩膜可视化与训练曲线
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from semseg import config as C
    from semseg.io_utils import imwrite
else:
    from semseg import config as C
    from semseg.io_utils import imwrite

COL = {C.ZONE: (117, 158, 29), C.DISK: (165, 95, 24)}     # BGR：圈绿、药片蓝


def denorm(x) -> np.ndarray:
    """ImageNet 归一化张量 或 直接 BGR uint8 图 → BGR uint8。"""
    if isinstance(x, np.ndarray) and x.dtype == np.uint8:
        return x
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    mean = np.asarray(C.IMAGENET_MEAN, dtype=np.float32).reshape(3, 1, 1)
    std = np.asarray(C.IMAGENET_STD, dtype=np.float32).reshape(3, 1, 1)
    img = np.clip((x * std + mean) * 255.0, 0, 255).astype(np.uint8)
    return cv2.cvtColor(np.transpose(img, (1, 2, 0)), cv2.COLOR_RGB2BGR)


def _text(img, s, org, color, scale=0.6, thick=2):
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thick + 2,
                cv2.LINE_AA)
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def _outline(base, mask, thickness=3):
    vis = base.copy()
    for c in range(mask.shape[0]):
        m = (mask[c] > 0.5).astype(np.uint8)
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if cnts:
            cv2.drawContours(vis, cnts, -1, COL[c], thickness)
    return vis


def _filled(base, mask, alpha=0.30):
    vis = base.copy()
    overlay = vis.copy()
    for c in range(mask.shape[0]):
        m = (mask[c] > 0.5).astype(np.uint8)
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if cnts:
            cv2.drawContours(overlay, cnts, -1, COL[c], cv2.FILLED)
            cv2.drawContours(vis, cnts, -1, COL[c], 3)
    return cv2.addWeighted(overlay, alpha, vis, 1 - alpha, 0)


def _header(tile, title):
    t = tile.copy()
    cv2.rectangle(t, (0, 0), (t.shape[1], 40), (22, 22, 22), -1)
    _text(t, title, (10, 28), (255, 255, 255), 0.75)
    return t


def save_panel(img_tensor, prob: np.ndarray, gt: np.ndarray, out_path: Path) -> None:
    """四联图：原图 | 人工标注 | 预测 | 差异（绿=命中 红=误检 蓝=漏检）"""
    base = denorm(img_tensor)
    p = (prob > C.THRESH).any(axis=0)
    g = (gt > 0.5).any(axis=0)
    tp, fp, fn = p & g, p & ~g, ~p & g
    diff = (base.astype(np.float32) * 0.45).astype(np.uint8)
    diff[tp] = (60, 200, 60)
    diff[fp] = (40, 40, 230)
    diff[fn] = (230, 140, 30)
    hit, extra, miss = int(tp.sum()), int(fp.sum()), int(fn.sum())
    den = hit + extra + miss
    for i, (t, col) in enumerate([
        (f"TP {hit}px", (60, 200, 60)), (f"FP {extra}px", (40, 40, 230)),
        (f"FN {miss}px", (230, 140, 30)),
        (f"IoU {hit/den if den else 1.0:.3f}", (255, 255, 255)),
    ]):
        _text(diff, t, (10, 26 + i * 24), col, 0.55, 1)

    tile = np.hstack([
        _header(base, "original"),
        _header(_outline(base, gt), "ground truth"),
        _header(_filled(base, prob), "prediction"),
        _header(diff, "difference"),
    ])
    imwrite(out_path, tile, quality=90)


def plot_curves(history: list[dict], out_path: Path) -> None:
    import io
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    ep = [h["epoch"] for h in history]
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))

    ax[0].plot(ep, [h["loss"] for h in history], color="#c0392b")
    ax[0].set_title("train loss")

    ax[1].plot(ep, [h["val"]["Area"]["Dice"] for h in history], label="Area")
    ax[1].plot(ep, [h["val"]["Yaoping"]["Dice"] for h in history], label="Yaoping")
    ax[1].plot(ep, [h["val"]["_macro_dice"] for h in history], label="macro",
               linestyle="--", color="k")
    ax[1].set_title("validation Dice")
    ax[1].legend()

    ax[2].plot(ep, [h["val"]["Area"]["IoU"] for h in history], label="Area")
    ax[2].plot(ep, [h["val"]["Yaoping"]["IoU"] for h in history], label="Yaoping")
    ax[2].set_title("validation IoU")
    ax[2].legend()

    for a in ax:
        a.set_xlabel("epoch")
        a.grid(alpha=0.3)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=140)
    buf.seek(0)
    arr = np.asarray(Image.open(buf).convert("RGB"))
    imwrite(out_path, cv2.cvtColor(arr, cv2.COLOR_RGB2BGR), quality=92)
    plt.close(fig)
