"""C# 实现 ↔ Python 参考实现 的逐实例掩膜一致性核验（同一坐标系下）。

为什么需要「先转正 EXIF」再比：
  · Python 参考实现走 cv2 解码（**自动应用 EXIF**）→ 正立帧
  · App/C# 走 SkiaSharp 解码（**不应用 EXIF**）→ 原始存储帧
两者对 EXIF=6 的照片相差一个 90° 旋转，而且「整图拉伸到 640」在两个帧下的形变不同，
模型输入因此不同（预测也会不同）。所以直接比对会把「坐标系差异」误当成「实现差异」。

做法：把测试图按 EXIF 转正、另存为 PNG（无 EXIF），再让 C# 跑同一批图，
此时 C# 与参考实现处在同一帧，比较结果只反映**实现**层面的差异。

用法：
    python compare_cs_vs_ref.py <csharp_out_dir> <ref_out_dir>
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "Antibacterial zone mask" / "data" / "test" / "images"
UPRIGHT = ROOT / "Antibacterial zone" / "_verify" / "test_upright"
MIN_AREA = 30


def make_upright() -> int:
    UPRIGHT.mkdir(parents=True, exist_ok=True)
    n = 0
    for p in sorted(SRC.glob("*.jpg")):
        out = UPRIGHT / (p.stem + ".png")
        if out.is_file():
            n += 1
            continue
        im = ImageOps.exif_transpose(Image.open(p))     # 应用 EXIF 后再存（PNG 无 EXIF）
        im.save(out)
        n += 1
    return n


def load_masks(d: Path, stem: str) -> dict[str, np.ndarray]:
    out = {}
    for p in d.glob(f"{stem}__*.png"):
        m = cv2.imdecode(np.fromfile(str(p), np.uint8), cv2.IMREAD_GRAYSCALE)
        if m is None:
            continue
        out[p.stem.split("__", 1)[1]] = (m > 127).astype(np.uint8)
    return out


def iou(a, b) -> float:
    inter = int(np.logical_and(a, b).sum())
    union = int(np.logical_or(a, b).sum())
    return inter / union if union else 0.0


def main():
    cs_dir, ref_dir = Path(sys.argv[1]), Path(sys.argv[2])
    print(f"测试图转正（PNG，无 EXIF）→ {make_upright()} 张，目录 {UPRIGHT}")
    print("请先对同一目录跑：Verify.exe seg <onnx> \"%s\" <cs_out>" % UPRIGHT)

    stems = sorted(p.stem for p in UPRIGHT.glob("*.png"))
    print(f"\n{'='*76}\nC# ↔ Python 参考（同一正立帧）\n{'='*76}")
    print(f"{'图片':<18}{'C#(D/Z)':>10}{'ref(D/Z)':>10}{'共有实例':>9}"
          f"{'IoU中位':>9}{'IoU最小':>9}{'完全一致':>9}")

    all_ious, n_same, n_key, count_mismatch = [], 0, 0, []
    for stem in stems:
        cs = load_masks(cs_dir / "masks", stem)
        rf = load_masks(ref_dir / "masks", stem)
        if not cs or not rf:
            print(f"{stem:<18} 缺掩膜（C# {len(cs)} / ref {len(rf)}）")
            continue
        keys = sorted(set(cs) & set(rf))
        ious = [iou(cs[k], rf[k]) for k in keys]
        same = sum(1 for v in ious if v == 1.0)
        all_ious += ious
        n_same += same
        n_key += len(keys)

        c_d = sum(1 for k in cs if k.startswith("Yaoping"))
        c_z = len(cs) - c_d
        r_d = sum(1 for k in rf if k.startswith("Yaoping"))
        r_z = len(rf) - r_d
        if (c_d, c_z) != (r_d, r_z):
            count_mismatch.append((stem, (c_d, c_z), (r_d, r_z)))

        print(f"{stem:<18}{c_d:>4}/{c_z:<5}{r_d:>4}/{r_z:<5}{len(keys):>9}"
              f"{np.median(ious):>9.4f}{min(ious):>9.4f}{same/len(keys):>9.1%}")

    print(f"\n汇总：共有实例 {n_key} 个；掩膜 IoU 中位 {np.median(all_ious):.4f}，"
          f"均值 {np.mean(all_ious):.4f}，最小 {min(all_ious):.4f}，"
          f"完全逐像素一致 {n_same}/{n_key} = {n_same/n_key:.1%}")
    print(f"实例数不一致的图：{len(count_mismatch)}/{len(stems)}")
    for stem, c, r in count_mismatch:
        print(f"  {stem:<18} C#={(c[0], c[1])}  ref={(r[0], r[1])}")


if __name__ == "__main__":
    main()
