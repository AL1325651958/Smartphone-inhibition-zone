"""Python twin of the App's `PillClassifier` preprocessing, aligned to the model.

Two variants are provided:

``clahe="app"``  -> reproduces what ``PillClassifier.cs`` computes today
                    (its own CLAHE, PIL-style float resampler, +-12 deg bicubic)
``clahe="cv2"``  -> the chain the model was trained on
                    (``cv2.createCLAHE`` + ``cv2.filter2D`` + ``cv2.LUT`` and a
                    bit-exact port of Pillow's 8-bit fixed-point BICUBIC)

Measured on the App's own 70 test crops (both variants, 5 TTA views):

    variant "app"   enhanced grey MAE 7.81 vs cv2 reference   (CLAHE formula)
    variant "cv2"   enhanced grey MAE 0.00 (bit-identical)
    variant "cv2"   160x160 tensor MAE ~1.5, from PIL rotate only

The `resample_pil_exact` function is a line-by-line port of
`_ImagingResampleHorizontal/Vertical_8bpc` in Pillow's `src/libImaging/Resample.c`
(PRECISION_BITS = 22, INT32 coefficient normalisation, per-pass 8-bit clamp); it
reproduces ``Image.resize(..., Image.BICUBIC)`` bit-for-bit on every resample this
pipeline performs.
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import preproc_class_reference as ref  # noqa: E402
from opencv_clahe_port import clahe_opencv  # noqa: E402

IMG_SIZE, CACHE_PX = 160, 192
SHARPEN_KERNEL = ref.SHARPEN_KERNEL
GAMMA_LUT = ref._GAMMA_LUT
PRECISION_BITS = 32 - 8 - 2                 # 22, from Pillow Resample.c
HALF = 1 << (PRECISION_BITS - 1)


# --------------------------------------------------------------------------- #
# Pillow-exact resampler
# --------------------------------------------------------------------------- #
def _bicubic_filter(x: float) -> float:
    a = -0.5
    x = abs(x)
    if x < 1.0:
        return ((a + 2.0) * x - (a + 3.0)) * x * x + 1
    if x < 2.0:
        return (((x - 5.0) * x + 8.0) * x - 4.0) * a
    return 0.0


def _precompute_coeffs(in_size: int, out_size: int, support: float = 2.0):
    """Identical arithmetic to Pillow's precompute_coeffs + normalize_coeffs_8bpc.

    Returns (kk_int [out, kmax], bounds [out,2], offsets [out,1]) where
    ``kk_int[o, i]`` is the INT32 coefficient for source index ``offsets[o] + i``.
    """
    scale = in_size / out_size
    filterscale = scale if scale >= 1.0 else 1.0
    sup = support * filterscale
    ksize = int(np.ceil(sup)) * 2 + 1
    kk = np.zeros((out_size, ksize), dtype=np.float64)
    bounds = np.zeros((out_size, 2), dtype=np.int64)
    inv = 1.0 / filterscale
    for xx in range(out_size):
        center = (xx + 0.5) * scale
        xmin = int(center - sup + 0.5)
        if xmin < 0:
            xmin = 0
        xmax = int(center + sup + 0.5)
        if xmax > in_size:
            xmax = in_size
        xmax -= xmin                 # Pillow reuses xmax as the tap count
        ww = 0.0
        for x in range(xmax):
            w = _bicubic_filter((x + xmin - center + 0.5) * inv)
            kk[xx, x] = w
            ww += w
        if ww != 0.0:
            kk[xx, :xmax] /= ww
        kk[xx, xmax:] = 0.0
        bounds[xx] = (xmin, xmax)
    kk_int = np.where(kk < 0, -0.5 + kk * (1 << PRECISION_BITS),
                      0.5 + kk * (1 << PRECISION_BITS)).astype(np.int64)
    return kk_int, bounds, bounds[:, :1]


def _clip8(v: np.ndarray) -> np.ndarray:
    """Pillow's clip8: arithmetic >> PRECISION_BITS then index the 0..255 table."""
    return np.clip(v >> PRECISION_BITS, 0, 255).astype(np.uint8)


def resample_pil_exact(src: np.ndarray, dw: int, dh: int,
                       support: float = 2.0) -> np.ndarray:
    """Port of Pillow's 8bpc resampler; bit-identical to Image.resize(BICUBIC).

    Vectorised, but the arithmetic is the same as the C loop: INT32 coefficients,
    per-output-pixel weighted sum starting from ``1 << (PRECISION_BITS-1)``, and a
    clamp per pass (horizontal result is written back as uint8 before the vertical
    pass, exactly like Pillow's two-pass 8-bit pipeline).
    """
    sh, sw = src.shape
    need_h, need_v = sw != dw, sh != dh
    horizontal_first = not ((sh - dh) > 0 and (sh - dh) > (sw - dw) * 2)

    def coeffs(in_size, out_size):
        kk, bounds, off = _precompute_coeffs(in_size, out_size, support)
        return kk, off

    if need_h:
        kh, oh = coeffs(sw, dw)
        # bounds[:,1] is Pillow's tap *count* (xmax is reused), so gather from
        # xmin only; the coefficients past the tap count are exactly 0.
        idx_h = oh + np.arange(kh.shape[1])[None, :]              # [dw, kmax]
        idx_h = np.clip(idx_h, 0, sw - 1)
    if need_v:
        kv, ov = coeffs(sh, dh)
        idx_v = np.clip(ov + np.arange(kv.shape[1])[None, :], 0, sh - 1)

    def hor(img):
        g = img[:, idx_h].astype(np.int64)                        # [h, dw, kmax]
        ss = HALF + (g * kh[None, :, :]).sum(axis=2)              # [h, dw]
        return _clip8(ss)

    def ver(img):
        g = img[idx_v, :].astype(np.int64)                        # [dh, kmax, w]
        ss = HALF + (g * kv[:, :, None]).sum(axis=1)              # [dh, w]
        return _clip8(ss)

    im = src
    if horizontal_first:
        if need_h:
            im = hor(im)
        if need_v:
            im = ver(im)
    else:
        if need_v:
            im = ver(im)
        if need_h:
            im = hor(im)
    return im


# --------------------------------------------------------------------------- #
# rotation: PIL semantics via OpenCV
# --------------------------------------------------------------------------- #
def rotate_pil_like(img: np.ndarray, angle: float, fill: int = 128) -> np.ndarray:
    """`Image.rotate(angle, BICUBIC, fillcolor=128, expand=False)` equivalent.

    PIL measures the angle counter-clockwise about ((w-1)/2, (h-1)/2); because
    ``cv2.getRotationMatrix2D`` maps +angle to the same counter-clockwise image
    rotation (the y axis points down in both), the angle is passed through
    unchanged.  Using the centre ``w/2`` instead of ``(w-1)/2`` costs ~9 grey
    levels; the residual ~3.9 grey levels come from PIL's transform kernel being a
    normalised, weight-renormalised convolution rather than ``cv2.INTER_CUBIC``.
    """
    h, w = img.shape
    m = cv2.getRotationMatrix2D(((w - 1) / 2.0, (h - 1) / 2.0), angle, 1.0)
    return cv2.warpAffine(img, m, (w, h), flags=cv2.INTER_CUBIC,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=fill)


def rotate_csharp_bicubic(src: np.ndarray, degrees: float,
                          fill: int = 128) -> np.ndarray:
    """Port of the App's current RotateBicubic (differs from PIL by ~4 grey levels)."""
    h, w = src.shape
    rad = np.deg2rad(degrees)
    cos, sin = np.cos(rad), np.sin(rad)
    ys, xs = np.mgrid[0:h, 0:w]
    dx = xs + 0.5 - w / 2.0
    dy = ys + 0.5 - h / 2.0
    sx = cos * dx + sin * dy + w / 2.0 - 0.5
    sy = -sin * dx + cos * dy + h / 2.0 - 0.5
    x0 = np.floor(sx).astype(int)
    y0 = np.floor(sy).astype(int)
    acc = np.zeros((h, w))
    wsum = np.zeros((h, w))
    for j in (-1, 0, 1, 2):
        syy = y0 + j
        wy = np.array([_bicubic_filter(v) for v in (sy - syy).ravel()]).reshape(h, w)
        for i in (-1, 0, 1, 2):
            sxx = x0 + i
            wx = np.array([_bicubic_filter(v) for v in (sx - sxx).ravel()]).reshape(h, w)
            wgt = wx * wy
            ok = (sxx >= 0) & (sxx < w) & (syy >= 0) & (syy < h) & (wgt != 0)
            vals = src[np.clip(syy, 0, h - 1), np.clip(sxx, 0, w - 1)]
            acc += np.where(ok, wgt * vals, 0.0)
            wsum += np.where(ok, wgt, 0.0)
    res = np.where(wsum == 0, fill,
                   np.clip(acc / np.where(wsum == 0, 1, wsum) + 0.5, 0, 255))
    return res.astype(np.uint8)


# --------------------------------------------------------------------------- #
# full pipelines
# --------------------------------------------------------------------------- #
def enhance(bgr: np.ndarray, clahe: str = "cv2") -> np.ndarray:
    """Steps (1)-(4): grayscale -> CLAHE -> 3x3 sharpen -> gamma LUT.

    clahe="cv2"          bit-identical with the training chain
    clahe="opencv_port"  the ported OpenCV CLAHE (== cv2, kept for reference)
    clahe="app"          the App's current ApplyCLAHE (differs, see report)
    """
    gray = ref.to_gray(bgr)
    if clahe == "app":
        cl = ref.clahe_csharp_port(gray)
    elif clahe == "opencv_port":
        cl = clahe_opencv(gray)
    else:
        cl = cv2.createCLAHE(clipLimit=ref.CLIP_LIMIT,
                             tileGridSize=ref.TILE_GRID).apply(gray)
    return cv2.LUT(cv2.filter2D(cl, -1, SHARPEN_KERNEL), GAMMA_LUT)


def preprocess(bgr: np.ndarray, tta: int = 5, cache_px: int = CACHE_PX,
               img_size: int = IMG_SIZE, clahe: str = "app",
               rotate: str = "pil") -> np.ndarray:
    """[V, 3, img_size, img_size] float32 0..255, as the App must build it."""
    cl = enhance(bgr, clahe)
    sq = ref.pad_to_square(cl)
    cache = resample_pil_exact(sq, cache_px, cache_px)
    views = [resample_pil_exact(cache, img_size, img_size)]
    if tta > 1:
        big = resample_pil_exact(cache, int(img_size * 1.10), int(img_size * 1.10))
        views.append(big[:img_size, :img_size].copy())
        views.append(big[big.shape[0] - img_size:, big.shape[1] - img_size:].copy())
    if tta >= 5:
        rfn = rotate_pil_like if rotate == "pil" else rotate_csharp_bicubic
        for ang in (-12, 12):
            views.append(resample_pil_exact(rfn(cache, ang, 128), img_size, img_size))
    arr = np.stack(views[:max(tta, 1)]).astype(np.float32)
    return np.ascontiguousarray(np.repeat(arr[:, None], 3, axis=1))
