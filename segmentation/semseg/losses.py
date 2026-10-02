"""
semseg.losses — 分割损失

前景极不平衡：药片只占约 1.2% 像素、抑菌圈约 13%，其余都是背景。
只用 BCE 会退化成"全部预测成背景"（药片通道 Dice=0，但 BCE 却很小）。

所以用两项组合：

    L = w_bce * BCE(带正样本加权)  +  (1 - w_bce) * SoftDice

* BCE 负责逐像素的稳定梯度；药片通道加正样本权重，进一步补偿前景稀少。
* SoftDice 直接优化重叠率，对类别不平衡不敏感 —— 这是把药片拉回来的关键。
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from semseg import config as C
except ImportError:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from semseg import config as C


class BCEDiceLoss(nn.Module):
    def __init__(self, bce_weight: float = C.BCE_WEIGHT,
                 disk_pos_weight: float = C.DISK_POS_WEIGHT):
        super().__init__()
        self.bce_weight = bce_weight
        # 通道 0(抑菌圈) 与 通道 1(药片) 各一个正样本权重
        self.register_buffer("pos_weight",
                             torch.tensor([1.0, disk_pos_weight], dtype=torch.float32))

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pw = self.pos_weight.to(logits.device).view(1, -1, 1, 1)
        bce = F.binary_cross_entropy_with_logits(logits, target, pos_weight=pw)

        prob = torch.sigmoid(logits)
        dims = (0, 2, 3)
        inter = (prob * target).sum(dims)
        denom = prob.sum(dims) + target.sum(dims)
        dice = 1.0 - (2.0 * inter + 1e-6) / (denom + 1e-6)   # 每类一个值
        return self.bce_weight * bce + (1.0 - self.bce_weight) * dice.mean()


class TotalLoss(nn.Module):
    """返回总损失与两个分量（用于日志）。"""

    def __init__(self):
        super().__init__()
        self.crit = BCEDiceLoss()

    def forward(self, logits, target):
        loss = self.crit(logits, target)
        with torch.no_grad():
            prob = torch.sigmoid(logits)
            dims = (0, 2, 3)
            inter = (prob * target).sum(dims)
            denom = prob.sum(dims) + target.sum(dims)
            dice = (2.0 * inter + 1e-6) / (denom + 1e-6)
        return loss, {C.CLASS_NAMES[0]: float(dice[0]), C.CLASS_NAMES[1]: float(dice[1])}
