"""
semseg.model — 双通道语义分割网络

    ResNet 编码器 + U-Net 解码器 + 2 通道分割头

输出 (B, 2, H, W) 的 logits：通道 0 = 抑菌圈，通道 1 = 药片。
推理时各自 sigmoid 得到概率图，阈值化即得掩膜。

为什么是这个结构
----------------
* 数据只有 38 张训练图（5 块平板），从零训练必然过拟合 → 用 ImageNet 预训练
  的 ResNet 做迁移
* 分割头做在全分辨率上，显存开销大 → 用 1×1 降维 + 深度可分离卷积
  （早期版本用双 3×3 全卷积，640×640 上直接 OOM）
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

try:
    from semseg import config as C
except ImportError:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from semseg import config as C


class ResNetEncoder(nn.Module):
    OUT_CHANNELS = (64, 64, 128, 256, 512)

    def __init__(self, name: str = "resnet34", pretrained: bool = True):
        super().__init__()
        if name == "resnet18":
            w = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
            net = models.resnet18(weights=w)
        elif name == "resnet34":
            w = models.ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
            net = models.resnet34(weights=w)
        else:
            raise ValueError(f"不支持的编码器：{name}")
        self.name = name
        self.stem = nn.Sequential(net.conv1, net.bn1, net.relu)
        self.pool = net.maxpool
        self.layer1, self.layer2 = net.layer1, net.layer2
        self.layer3, self.layer4 = net.layer3, net.layer4

    def forward(self, x):
        s = self.stem(x)                 # H/2
        c1 = self.layer1(self.pool(s))   # H/4
        c2 = self.layer2(c1)             # H/8
        c3 = self.layer3(c2)             # H/16
        c4 = self.layer4(c3)             # H/32
        return s, c1, c2, c3, c4


class DoubleConv(nn.Module):
    def __init__(self, cin: int, cmid: int, cout: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(cin, cmid, 3, padding=1, bias=False),
            nn.BatchNorm2d(cmid), nn.ReLU(inplace=True),
            nn.Conv2d(cmid, cout, 3, padding=1, bias=False),
            nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class DecoderBlock(nn.Module):
    def __init__(self, cin: int, cskip: int, cout: int):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.conv = DoubleConv(cin + cskip, cout, cout)

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear",
                              align_corners=False)
        return self.conv(torch.cat([x, skip], dim=1))


class UNetDecoder(nn.Module):
    def __init__(self, encoder_channels=ResNetEncoder.OUT_CHANNELS):
        super().__init__()
        c_s, c1, c2, c3, c4 = encoder_channels
        self.d4 = DecoderBlock(c4, c3, 256)
        self.d3 = DecoderBlock(256, c2, 128)
        self.d2 = DecoderBlock(128, c1, 64)
        self.d1 = DecoderBlock(64, c_s, 64)
        self.final_up = nn.Upsample(scale_factor=2, mode="bilinear",
                                    align_corners=False)
        self.out_channels = 64

    def forward(self, feats):
        s, c1, c2, c3, c4 = feats
        d = self.d4(c4, c3)
        d = self.d3(d, c2)
        d = self.d2(d, c1)
        d = self.d1(d, s)
        return self.final_up(d)


class SegNet(nn.Module):
    """双通道语义分割。输出 logits (B, 2, H, W)。"""

    def __init__(self, encoder: str = C.ENCODER, num_classes: int = C.NUM_CLASSES,
                 pretrained: bool = True, head_width: int = 32):
        super().__init__()
        self.encoder_name = encoder
        self.encoder = ResNetEncoder(encoder, pretrained=pretrained)
        self.decoder = UNetDecoder()
        c = self.decoder.out_channels

        # 轻量分割头：1×1 降维 → 3×3 深度可分离 → 1×1 输出
        self.head = nn.Sequential(
            nn.Conv2d(c, head_width, 1, bias=False),
            nn.BatchNorm2d(head_width),
            nn.ReLU(inplace=True),
            nn.Conv2d(head_width, head_width, 3, padding=1, groups=head_width,
                      bias=False),
            nn.BatchNorm2d(head_width),
            nn.ReLU(inplace=True),
            nn.Conv2d(head_width, num_classes, 1),
        )
        nn.init.zeros_(self.head[-1].bias)
        nn.init.normal_(self.head[-1].weight, std=0.01)
        self.num_classes = num_classes

    def forward(self, x):
        return self.head(self.decoder(self.encoder(x)))

    @torch.no_grad()
    def predict(self, x):
        """返回概率图 (B, K, H, W)。"""
        return torch.sigmoid(self(x))


def build_model(encoder: str = C.ENCODER, pretrained: bool = True) -> SegNet:
    return SegNet(encoder=encoder, pretrained=pretrained)


def count_parameters(model: nn.Module) -> tuple[int, int]:
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
        print(f"{name:9s} 参数={t/1e6:6.2f}M  输出={tuple(y.shape)}  cpu前向={dt:.0f}ms")
