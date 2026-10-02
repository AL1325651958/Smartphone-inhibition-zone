"""Mild augmentation transforms: rotation, contrast/brightness, gentle distortion.

Kept deliberately subtle - the goal is more views of the same disk, not new
shapes. Disk-text recognition breaks if the glyphs are distorted or cropped away.
"""
from __future__ import annotations

import math
import random

import numpy as np
from PIL import Image


def rotate(im: Image.Image, deg: float, fill=128) -> Image.Image:
    return im.rotate(deg, resample=Image.BICUBIC, fillcolor=fill)


def contrast_brightness(im: Image.Image, contrast: float, brightness: float) -> Image.Image:
    """contrast: 1.0 = unchanged. brightness: additive offset in 0-255 units."""
    g = np.asarray(im, dtype=np.float32)
    g = (g - 128.0) * contrast + 128.0 + brightness
    return Image.fromarray(np.clip(g, 0, 255).astype(np.uint8))


def perspective(im: Image.Image, mag: float, fill=0) -> Image.Image:
    """Mild keystone: shift each corner by at most `mag` of the width."""
    w, h = im.size
    d = mag * min(w, h)

    def corner():
        return random.uniform(-d, d)

    # PIL maps the OUTPUT quad to the SOURCE rectangle, so jitter the source quad
    src = [(0, 0), (w, 0), (w, h), (0, h)]
    jit = [(max(0, min(w, x + corner())), max(0, min(h, y + corner())))
           for x, y in src]
    coeffs = _find_coeffs(jit, src)
    return im.transform((w, h), Image.PERSPECTIVE, coeffs, Image.BICUBIC,
                        fillcolor=fill)


def _find_coeffs(pa, pb):
    """Coefficients mapping the quad pa onto the rectangle pb."""
    A = []
    B = []
    for (x, y), (X, Y) in zip(pa, pb):
        A.append([x, y, 1, 0, 0, 0, -X * x, -X * y])
        B.append(X)
        A.append([0, 0, 0, x, y, 1, -Y * x, -Y * y])
        B.append(Y)
    res = np.linalg.solve(np.array(A, dtype=np.float64), np.array(B, dtype=np.float64))
    return res.tolist()


def barrel(im: Image.Image, k: float, fill=0) -> Image.Image:
    """Radial distortion: k>0 pincushion, k<0 barrel. Applied via inverse map."""
    arr = np.asarray(im)
    while arr.ndim > 2:                       # collapse any trailing singleton axes
        arr = arr[..., 0]
    arr = arr.astype(np.uint8)
    h, w = arr.shape
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    nx = (xx - cx) / cx
    ny = (yy - cy) / cy
    r2 = nx * nx + ny * ny
    f = 1.0 + k * r2                          # mild, single-term
    sx = np.clip(nx * f * cx + cx, 0, w - 1)
    sy = np.clip(ny * f * cy + cy, 0, h - 1)
    x0 = np.floor(sx).astype(np.int32)
    y0 = np.floor(sy).astype(np.int32)
    x1 = np.clip(x0 + 1, 0, w - 1)
    y1 = np.clip(y0 + 1, 0, h - 1)
    wx = (sx - x0).astype(np.float32)
    wy = (sy - y0).astype(np.float32)
    top = arr[y0, x0].astype(np.float32) * (1 - wx) + arr[y0, x1].astype(np.float32) * wx
    bot = arr[y1, x0].astype(np.float32) * (1 - wx) + arr[y1, x1].astype(np.float32) * wx
    out = top * (1 - wy) + bot * wy
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8), mode="L")


def mild_combo(im: Image.Image, rng: random.Random, strength: str = "mild",
               allow_distort: bool = True) -> tuple[Image.Image, dict]:
    """One random mild augmentation. Returns (image, parameters used)."""
    if strength == "mild":
        rot_range, con_range, bri_range = 15.0, 0.12, 10.0
        keystone, barrel_k = 0.025, 0.06
    else:                                   # "gentle"
        rot_range, con_range, bri_range = 8.0, 0.07, 6.0
        keystone, barrel_k = 0.015, 0.035

    p = {}
    out = im
    p["rotate"] = rng.uniform(-rot_range, rot_range)
    out = rotate(out, p["rotate"], fill=128)

    p["contrast"] = 1.0 + rng.uniform(-con_range, con_range)
    p["brightness"] = rng.uniform(-bri_range, bri_range)
    out = contrast_brightness(out, p["contrast"], p["brightness"])

    if allow_distort and rng.random() < 0.6:
        if rng.random() < 0.5:
            p["keystone"] = keystone
            out = perspective(out, keystone, fill=0)
        else:
            p["barrel"] = rng.choice([-barrel_k, barrel_k])
            out = barrel(out, p["barrel"], fill=0)
    return out, p
