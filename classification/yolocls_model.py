"""YOLO-style classification network (YOLOv8/YOLO11-cls backbone) in plain PyTorch.

Mirrors the structure ultralytics uses for `*-cls` models:
  Conv(3x3, BN, SiLU) -> C2f (CSP bottleneck with 2 splits) -> SPPF -> head
Differences from the earlier PillCNN:
  * BatchNorm2d instead of GroupNorm (YOLO uses BN; fine at these batch sizes)
  * CSP-style C2f blocks with n scaled by depth, widths scaled by width multiple
  * SPPF (three chained 5x5 max-pools) before the classifier
  * a 1x1 conv classifier head after dropout, as in YOLO-cls

Model sizes follow the YOLO naming so `yolov8n-cls`-like and larger variants are
available via `scale`.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

# depth, width multiples roughly as in ultralytics for n/s/m
SCALES = {
    "n": (1 / 3, 0.25),
    "s": (1 / 3, 0.50),
    "m": (2 / 3, 0.75),
}


def autopad(k, p=None):
    if p is not None:
        return p
    if isinstance(k, (tuple, list)):
        return tuple(x // 2 for x in k)
    return k // 2


class Conv(nn.Module):
    """Conv2d + BatchNorm + SiLU (YOLO's standard block)."""

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2, eps=0.001, momentum=0.03)
        self.act = nn.SiLU() if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class Bottleneck(nn.Module):
    """Standard YOLO bottleneck: two 3x3 convs with a residual shortcut."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3)):
        super().__init__()
        c_ = c2 // 2
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        y = self.cv2(self.cv1(x))
        return x + y if self.add else y


class C2f(nn.Module):
    """CSP bottleneck with two convolutions and n bottlenecks (YOLOv8 C2f)."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(
            Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)))
            for _ in range(n))

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class SPPF(nn.Module):
    """Spatial pyramid pooling - fast (three 5x5 max-pools)."""

    def __init__(self, c1, c2, k=5):
        super().__init__()
        c_ = c1 // 2
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        y = [self.cv1(x)]
        y.extend(self.m(y[-1]) for _ in range(3))
        return self.cv2(torch.cat(y, 1))


class YOLOCls(nn.Module):
    """YOLO-style classifier.

    Layout (input 112 px -> feature map 4x4 before the head):
      stem conv s2, conv s2, C2f, conv s2, C2f, conv s2, C2f,
      conv s2, C2f, SPPF, dropout, 1x1 conv -> classes, global average pool
    """

    def __init__(self, ncls: int = 7, scale: str = "n", in_ch: int = 3,
                 dropout: float = 0.2):
        super().__init__()
        depth, width = SCALES[scale]
        w = lambda c: max(16, int(round(c * width / 8) * 8))   # noqa: E731
        n = lambda d: max(1, round(d * depth))                 # noqa: E731

        self.stem = nn.Sequential(
            Conv(in_ch, w(64), 3, 2),          # 112 -> 56
            Conv(w(64), w(128), 3, 2),         # 56 -> 28
            C2f(w(128), w(128), n(3), True),
        )
        self.stage2 = nn.Sequential(
            Conv(w(128), w(256), 3, 2),        # 28 -> 14
            C2f(w(256), w(256), n(6), True),
        )
        self.stage3 = nn.Sequential(
            Conv(w(256), w(512), 3, 2),        # 14 -> 7
            C2f(w(512), w(512), n(6), True),
        )
        self.stage4 = nn.Sequential(
            Conv(w(512), w(1024), 3, 2),       # 7 -> 4
            C2f(w(1024), w(1024), n(3), True),
        )
        self.sppf = SPPF(w(1024), w(1024), 5)
        self.drop = nn.Dropout(dropout)
        self.cls = nn.Conv2d(w(1024), ncls, 1)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out",
                                        nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.stem(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.sppf(x)
        x = self.drop(x)
        x = self.cls(x)
        return F.adaptive_avg_pool2d(x, 1).flatten(1)


def count_params(m):
    return sum(p.numel() for p in m.parameters())


if __name__ == "__main__":
    for s in ("n", "s", "m"):
        m = YOLOCls(7, s)
        import torch as t
        y = m(t.zeros(2, 3, 112, 112))
        print(f"scale {s}: params {count_params(m)/1e6:.2f}M  out {tuple(y.shape)}")
