"""
semseg.metrics — 逐类别分割指标

只看掩膜质量，按类分别报告：

    Dice        2TP / (2TP + FP + FN)   ← 主指标
    IoU         TP / (TP + FP + FN)
    Precision   TP / (TP + FP)
    Recall      TP / (TP + FN)

指标由**跨图累计的像素计数**算出，而不是逐图平均 —— 后者会被小目标的图带偏。
另外统计「该类缺失的图数」：人工标注里本来就没有这个类的图片，看到这列才知道
Recall 为什么会被拉低。
"""
from __future__ import annotations

import numpy as np

try:
    from semseg import config as C
except ImportError:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from semseg import config as C


def confusion_counts(pred_prob: np.ndarray, target: np.ndarray,
                     thresh: float = C.THRESH) -> list[tuple[int, int, int, int]]:
    """返回每类的 (tp, fp, fn, tn)。pred_prob/target 形状 (K,H,W)。"""
    out = []
    for c in range(pred_prob.shape[0]):
        p = pred_prob[c] > thresh
        g = target[c] > 0.5
        out.append((
            int(np.logical_and(p, g).sum()),
            int(np.logical_and(p, ~g).sum()),
            int(np.logical_and(~p, g).sum()),
            int(np.logical_and(~p, ~g).sum()),
        ))
    return out


def metrics_from_counts(counts: list[tuple[int, int, int, int]],
                        absent: dict[int, int] | None = None) -> dict:
    out = {}
    for c, (tp, fp, fn, tn) in enumerate(counts):
        eps = 1e-9
        out[C.CLASS_NAMES.get(c, str(c))] = {
            "Dice": 2 * tp / (2 * tp + fp + fn + eps),
            "IoU": tp / (tp + fp + fn + eps),
            "Precision": tp / (tp + fp + eps),
            "Recall": tp / (tp + fn + eps),
            "Accuracy": (tp + tn) / (tp + tn + fp + fn + eps),
            "n_pos_px": tp + fn,
            "n_tp": tp, "n_fp": fp, "n_fn": fn,
            "n_images_absent": (absent or {}).get(c, 0),
        }
    return out


def macro_dice(m: dict) -> float:
    return float(np.mean([v["Dice"] for v in m.values()]))
