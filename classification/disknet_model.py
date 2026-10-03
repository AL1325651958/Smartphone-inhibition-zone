"""A new classifier for this task: multi-scale detail-preserving CNN.

The task is unusual, so the design targets it directly rather than copying a
generic backbone:

  * the only discriminative signal is the printed abbreviation, a few pixels
    tall, so early stages keep resolution (a single stride-4 stem, no aggressive
    early downsampling) and the final feature map stays at 7x7 rather than 4x4;
  * disks are photographed at different orientations and scales across batches,
    so features from all four stages are fused instead of using only the last;
  * batch appearance varies a lot, so normalisation is BatchNorm (as in standard CNN blocks)
    with SiLU activations, plus squeeze-excitation for channel re-weighting;
  * printed strokes are thin, so a high-pass (difference-of-Gaussians) branch is
    computed and injected early, giving the network an explicit edge signal.

Ideas borrowed from common CNN classification designs: BN+SiLU blocks, CSP-style split and
concatenate, SPPF for context, 1x1 conv classifier head. The arrangement,
the multi-scale pooling fusion, the learned scale gate and the high-pass stem
are specific to this problem.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

VARIANTS = {
    "tiny": dict(w=(24, 48, 96, 192), blocks=(1, 1, 2, 1)),
    "small": dict(w=(32, 64, 128, 256), blocks=(2, 2, 2, 2)),
    "base": dict(w=(48, 96, 192, 384), blocks=(2, 3, 3, 2)),
}


class ConvBNAct(nn.Module):
    """Conv -> BatchNorm -> SiLU basic block."""

    def __init__(self, c1, c2, k=3, s=1, g=1):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, k // 2, groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2, eps=1e-3, momentum=0.1)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class SE(nn.Module):
    """Squeeze-excitation channel attention."""

    def __init__(self, c, r=8):
        super().__init__()
        h = max(8, c // r)
        self.fc1 = nn.Conv2d(c, h, 1)
        self.fc2 = nn.Conv2d(h, c, 1)

    def forward(self, x):
        s = F.adaptive_avg_pool2d(x, 1)
        return x * torch.sigmoid(self.fc2(F.silu(self.fc1(s))))


class ResBlock(nn.Module):
    """Two 3x3 convs with a residual shortcut and squeeze-excitation."""

    def __init__(self, c, drop=0.0):
        super().__init__()
        self.cv1 = ConvBNAct(c, c, 3)
        self.cv2 = nn.Conv2d(c, c, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(c, eps=1e-3, momentum=0.1)
        self.se = SE(c)
        self.drop = nn.Dropout2d(drop) if drop > 0 else nn.Identity()

    def forward(self, x):
        y = self.cv2(self.cv1(x))
        y = self.se(self.bn2(y))
        return F.silu(self.drop(y) + x)


class Transition(nn.Module):
    """Stride-2 downsample with a 3x3 conv (kept simple; no max-pool blurring)."""

    def __init__(self, c1, c2):
        super().__init__()
        self.conv = ConvBNAct(c1, c2, 3, 2)

    def forward(self, x):
        return self.conv(x)


class SPPF(nn.Module):
    """Spatial pyramid pooling (context at three receptive fields)."""

    def __init__(self, c1, c2, k=5):
        super().__init__()
        c_ = c1 // 2
        self.cv1 = ConvBNAct(c1, c_, 1, 1)
        self.cv2 = ConvBNAct(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(k, 1, k // 2)

    def forward(self, x):
        y = [self.cv1(x)]
        y.extend(self.m(y[-1]) for _ in range(3))
        return self.cv2(torch.cat(y, 1))


class MultiScaleHead(nn.Module):
    """Pool every stage to one token and fuse them with a learned scale gate."""

    def __init__(self, widths, ncls, token=128, drop=0.2):
        super().__init__()
        self.proj = nn.ModuleList([
            nn.Sequential(nn.Conv2d(c, token, 1, bias=False),
                          nn.BatchNorm2d(token, eps=1e-3, momentum=0.1),
                          nn.SiLU(inplace=True)) for c in widths])
        self.gate = nn.Sequential(
            nn.Linear(token * len(widths), token), nn.SiLU(),
            nn.Linear(token, len(widths)), nn.Softmax(dim=1))
        self.drop = nn.Dropout(drop)
        self.fc = nn.Linear(token, ncls)

    def forward(self, feats):
        toks = [F.adaptive_avg_pool2d(p(f), 1).flatten(1)
                for p, f in zip(self.proj, feats)]
        stack = torch.stack(toks, dim=1)                    # B x S x token
        w = self.gate(torch.cat(toks, dim=1)).unsqueeze(-1)  # B x S x 1
        z = (stack * w).sum(1)
        return self.fc(self.drop(z))


class HighPass(nn.Module):
    """Fixed difference-of-Gaussians branch: gives the net an explicit edge map."""

    def __init__(self, c_in=3, c_out=16):
        super().__init__()
        self.c_out = c_out
        self.register_buffer("k_small", self._gauss(5, 1.0), persistent=False)
        self.register_buffer("k_large", self._gauss(9, 2.5), persistent=False)
        self.proj = ConvBNAct(c_in * 2, c_out, 1, 1)

    @staticmethod
    def _gauss(k, sigma):
        ax = torch.arange(k, dtype=torch.float32) - (k - 1) / 2
        g = torch.exp(-(ax ** 2) / (2 * sigma ** 2))
        g = g / g.sum()
        return g.view(1, 1, 1, k)

    def forward(self, x):
        c = x.shape[1]
        # pad with zeros between channel blocks so the kernel stays separable
        ks = self.k_small.squeeze().view(1, 1, 1, -1).repeat(c, 1, 1, 1)
        kl = self.k_large.squeeze().view(1, 1, 1, -1).repeat(c, 1, 1, 1)
        hc = c * ks.shape[2]
        blur_s = F.conv2d(F.pad(x, (2, 2, 0, 0), mode="replicate"),
                          ks, groups=c)
        blur_l = F.conv2d(F.pad(x, (4, 4, 0, 0), mode="replicate"),
                          kl, groups=c)
        del hc
        return self.proj(torch.cat([x - blur_s, blur_s - blur_l], dim=1))


class DiskNet(nn.Module):
    """Multi-scale detail-preserving classifier for disk abbreviations.

    Input 112x112 -> stages at 56 / 28 / 14 / 7 px. All four stages feed the
    head, so the small printed glyphs are still represented at 56 px.
    """

    def __init__(self, ncls: int = 7, variant: str = "small", drop=0.05,
                 head_drop=0.3, token=128):
        super().__init__()
        cfg = VARIANTS[variant]
        w = cfg["w"]
        blocks = cfg["blocks"]

        self.highpass = HighPass(3, 16)
        self.stem = nn.Sequential(
            ConvBNAct(3 + 16, w[0], 3, 2),      # 112 -> 56
            ConvBNAct(w[0], w[0], 3, 1),
        )
        self.stage1 = nn.Sequential(*[ResBlock(w[0], drop) for _ in range(blocks[0])])
        self.down2 = Transition(w[0], w[1])
        self.stage2 = nn.Sequential(*[ResBlock(w[1], drop) for _ in range(blocks[1])])
        self.down3 = Transition(w[1], w[2])
        self.stage3 = nn.Sequential(*[ResBlock(w[2], drop) for _ in range(blocks[2])])
        self.down4 = Transition(w[2], w[3])
        self.stage4 = nn.Sequential(*[ResBlock(w[3], drop) for _ in range(blocks[3])])
        self.sppf = SPPF(w[3], w[3])
        self.head = MultiScaleHead(list(w), ncls, token=token, drop=head_drop)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out",
                                        nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        hp = self.highpass(x)
        x = self.stem(torch.cat([x, hp], dim=1))     # 56
        f1 = self.stage1(x)                          # 56
        f2 = self.stage2(self.down2(f1))             # 28
        f3 = self.stage3(self.down3(f2))             # 14
        f4 = self.sppf(self.stage4(self.down4(f3)))  # 7
        return self.head([f1, f2, f3, f4])


def count_params(m):
    return sum(p.numel() for p in m.parameters())


if __name__ == "__main__":
    for v in VARIANTS:
        m = DiskNet(7, v)
        y = m(torch.zeros(2, 3, 112, 112))
        print(f"variant {v:6}: params {count_params(m)/1e6:.2f}M  out {tuple(y.shape)}")
