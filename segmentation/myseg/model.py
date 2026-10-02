"""
myseg/model.py — ResNet 编码器 + U-Net 解码器（原生 PyTorch）

为什么是这个组合
----------------
本项目只有 78 张原图（增强后约 1100 张）。从零训练分割网络在这么小的数据上
必然过拟合，所以用 **ImageNet 预训练权重**做迁移学习。

编码器用 torchvision 的 ResNet（不依赖任何检测框架），解码器是自己写的
U-Net：逐级上采样 + 与编码器同尺度特征做跳跃连接，最后 1×1 卷积输出
NUM_CLASSES 个通道，配 sigmoid 得到每个类别的概率图。

编码器各阶段输出通道（ResNet18/34 相同）：
    stem        64   (H/2)
    layer1      64   (H/4)
    layer2     128   (H/8)
    layer3     256   (H/16)
    layer4     512   (H/32)

解码器从 H/32 逐级上采样回 H，跳跃连接取自 stem / layer1 / layer2 / layer3。
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

try:
    from myseg import config as C
except ImportError:                                     # 允许直接运行
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from myseg import config as C


def _is_resnet34(name: str) -> bool:
    return "34" in name


class ResNetEncoder(nn.Module):
    """torchvision ResNet，去掉最后的全局池化与全连接，暴露中间特征图。"""

    def __init__(self, name: str = "resnet34", pretrained: bool = True):
        super().__init__()
        self.name = name
        if name == "resnet18":
            w = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
            net = models.resnet18(weights=w)
            self.out_channels = (64, 64, 128, 256, 512)
        elif name == "resnet34":
            w = models.ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
            net = models.resnet34(weights=w)
            self.out_channels = (64, 64, 128, 256, 512)
        else:
            raise ValueError(f"不支持的编码器：{name}（可用 resnet18 / resnet34）")

        self.stem = nn.Sequential(net.conv1, net.bn1, net.relu)   # H/2, 64
        self.maxpool = net.maxpool
        self.layer1 = net.layer1                                   # H/4, 64
        self.layer2 = net.layer2                                   # H/8, 128
        self.layer3 = net.layer3                                   # H/16, 256
        self.layer4 = net.layer4                                   # H/32, 512

    def forward(self, x):
        s = self.stem(x)                # H/2
        x = self.maxpool(s)             # H/4
        c1 = self.layer1(x)             # H/4
        c2 = self.layer2(c1)            # H/8
        c3 = self.layer3(c2)            # H/16
        c4 = self.layer4(c3)            # H/32
        return [s, c1, c2, c3, c4]


class DoubleConv(nn.Module):
    """(conv 3×3 → BN → ReLU) × 2"""

    def __init__(self, cin: int, cmid: int, cout: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(cin, cmid, 3, padding=1, bias=False),
            nn.BatchNorm2d(cmid),
            nn.ReLU(inplace=True),
            nn.Conv2d(cmid, cout, 3, padding=1, bias=False),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class DecoderBlock(nn.Module):
    """上采样 → 拼接跳跃特征 → 双卷积"""

    def __init__(self, cin: int, cskip: int, cout: int):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.conv = DoubleConv(cin + cskip, cout, cout)

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:                # 奇偶尺寸兜底
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear",
                              align_corners=False)
        return self.conv(torch.cat([x, skip], dim=1))


class ResNetUNet(nn.Module):
    """
    ResNet 编码器 + U-Net 解码器。输出 (B, NUM_CLASSES, H, W) 的 logits。
    用 logits + BCEWithLogitsLoss（数值更稳），推理时再 sigmoid。
    """

    def __init__(self, encoder: str = C.ENCODER, num_classes: int = C.NUM_CLASSES,
                 pretrained: bool = True):
        super().__init__()
        self.encoder = ResNetEncoder(encoder, pretrained=pretrained)
        c_s, c1, c2, c3, c4 = self.encoder.out_channels      # 64, 64, 128, 256, 512
        self.encoder_name = encoder

        # 逐级上采样：H/32 → H/16 → H/8 → H/4 → H/2
        self.dec4 = DecoderBlock(c4, c3, 256)                # → H/16, 256
        self.dec3 = DecoderBlock(256, c2, 128)               # → H/8,  128
        self.dec2 = DecoderBlock(128, c1, 64)                # → H/4,  64
        self.dec1 = DecoderBlock(64, c_s, 64)                # → H/2,  64
        self.final_up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.head = nn.Sequential(
            DoubleConv(64, 64, 64),
            nn.Conv2d(64, num_classes, 1),
        )
        self.num_classes = num_classes

    def forward(self, x):
        s, c1, c2, c3, c4 = self.encoder(x)
        d = self.dec4(c4, c3)
        d = self.dec3(d, c2)
        d = self.dec2(d, c1)
        d = self.dec1(d, s)
        d = self.final_up(d)
        return self.head(d)                                  # logits


def build_model(encoder: str = C.ENCODER, pretrained: bool = True) -> ResNetUNet:
    return ResNetUNet(encoder=encoder, num_classes=C.NUM_CLASSES, pretrained=pretrained)


def count_parameters(model: nn.Module) -> tuple[int, int]:
    """返回 (总参数量, 可训练参数量)。"""
    total = sum(p.numel() for p in model.parameters())
    train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, train


if __name__ == "__main__":
    import time
    for name in ("resnet18", "resnet34"):
        m = build_model(name, pretrained=False)
        t, tr = count_parameters(m)
        x = torch.randn(2, 3, C.IMG_SIZE, C.IMG_SIZE)
        m.eval()
        with torch.no_grad():
            t0 = time.perf_counter()
            y = m(x)
            dt = (time.perf_counter() - t0) * 1000
        print(f"{name:10s} params={t/1e6:6.2f}M  trainable={tr/1e6:6.2f}M  "
              f"in={tuple(x.shape)} out={tuple(y.shape)}  cpu_fwd={dt:.0f}ms")
