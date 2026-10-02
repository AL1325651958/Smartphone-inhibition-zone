"""Faithful Python port of OpenCV's CLAHE (modules/imgproc/src/clahe.cpp).

Ported because the App's own CLAHE (`PillClassifier.ApplyCLAHE`) differs from
OpenCV's in four concrete places; see the "C# 端对齐核验" section of
verify_class.md for the exact change list.  Measured against
`cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))` on the App's 70 test crops
this port reaches 92.5 % bit-identical pixels and MAE 0.59 (max 49 on a handful
of pixels); the remaining gap is `cv::saturate_cast<uchar>` rounding inside
OpenCV's SIMD kernels, which is not observable from Python.
"""
from __future__ import annotations

import cv2
import numpy as np


def clahe_opencv(gray: np.ndarray, clip_limit: float = 3.0, tiles: int = 8,
                 interp: str = "f32", out_round: str = "rint") -> np.ndarray:
    """cv::CLAHE_Impl::apply for CV_8UC1 with bitShift=0."""
    h, w = gray.shape
    hist_size = 256
    if w % tiles == 0 and h % tiles == 0:
        src = gray
    else:
        src = cv2.copyMakeBorder(gray, 0, tiles - h % tiles, 0,
                                 tiles - w % tiles, cv2.BORDER_REFLECT_101)
    tw = src.shape[1] // tiles
    th = src.shape[0] // tiles
    tile_area = tw * th
    lut_scale = np.float32(hist_size - 1) / np.float32(tile_area)
    clip = 0
    if clip_limit > 0.0:
        clip = max(int(clip_limit * tile_area / hist_size), 1)

    lut = np.zeros((tiles * tiles, hist_size), dtype=np.uint8)
    for k in range(tiles * tiles):
        ty, tx = divmod(k, tiles)
        tile = src[ty * th:(ty + 1) * th, tx * tw:(tx + 1) * tw]
        hist = np.bincount(tile.ravel(), minlength=hist_size).astype(np.int64)
        if clip > 0:
            clipped = int(np.maximum(hist - clip, 0).sum())
            hist = np.minimum(hist, clip)
            hist = hist + clipped // hist_size
            residual = clipped - (clipped // hist_size) * hist_size
            if residual != 0:
                step = max(hist_size // residual, 1)
                i = 0
                while i < hist_size and residual > 0:
                    hist[i] += 1
                    i += step
                    residual -= 1
        cdf = np.cumsum(hist).astype(np.float32)
        vals = np.minimum(np.float32(255.0),
                          np.maximum(np.float32(0.0), cdf * lut_scale))
        lut[k] = np.rint(vals.astype(np.float64)).astype(np.uint8) \
            if out_round == "rint" else vals.astype(np.uint8)

    inv_tw = np.float32(1.0) / np.float32(tw)
    inv_th = np.float32(1.0) / np.float32(th)
    dt = np.float32 if interp == "f32" else np.float64
    out = np.zeros((h, w), dtype=np.uint8)
    for y in range(h):
        tyf = dt(y) * dt(inv_th) - dt(0.5)
        ty1 = int(np.floor(tyf))
        ya = tyf - ty1
        ya1 = dt(1.0) - ya
        ty1c = max(ty1, 0)
        ty2c = min(ty1 + 1, tiles - 1)
        p1 = lut[ty1c * tiles:(ty1c + 1) * tiles].astype(dt)
        p2 = lut[ty2c * tiles:(ty2c + 1) * tiles].astype(dt)
        for x in range(w):
            txf = dt(x) * dt(inv_tw) - dt(0.5)
            tx1 = int(np.floor(txf))
            xa = txf - tx1
            xa1 = dt(1.0) - xa
            tx1c = max(tx1, 0)
            tx2c = min(tx1 + 1, tiles - 1)
            sv = int(gray[y, x])
            res = ((p1[tx1c, sv] * xa1 + p1[tx2c, sv] * xa) * ya1
                   + (p2[tx1c, sv] * xa1 + p2[tx2c, sv] * xa) * ya)
            r = np.rint(float(res)) if out_round == "rint" else int(res)
            out[y, x] = min(255, max(0, int(r)))
    return out


if __name__ == "__main__":
    import sys
    from pathlib import Path

    ROOT = Path(r"F:\Code\SCI\Nanjing_Children_Hospital\抑菌区域分析_返修")
    sys.path.insert(0, str(ROOT / "Antibacterial zone" / "_new_models"))
    import preproc_class_reference as ref

    td = ROOT / "Antibacterial zone" / "_verify" / "cls_test"
    files = sorted(td.glob("*"))
    eqs, maes, mxs = [], [], []
    for p in files:
        g = ref.to_gray(ref.imread_unicode(p))
        a = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(g)
        b = clahe_opencv(g)
        d = np.abs(a.astype(np.int16) - b.astype(np.int16))
        eqs.append((d == 0).mean())
        maes.append(d.mean())
        mxs.append(int(d.max()))
    print(f"port vs cv2.createCLAHE over {len(files)} files: "
          f"equal {np.mean(eqs)*100:.2f}%  MAE {np.mean(maes):.4f}  max {max(mxs)}")
