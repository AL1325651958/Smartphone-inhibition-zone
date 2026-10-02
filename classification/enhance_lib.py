"""Enhancement pipeline (imagepross.py) made Unicode-path safe.

cv2.imread fails on paths containing Chinese characters on Windows, so images are
read through numpy and decoded with cv2.imdecode. The processing steps are
otherwise identical to the supplied script:
  grayscale -> CLAHE(clipLimit=3.0, tile=8x8) -> Laplacian sharpen -> gamma 1.5
"""
from __future__ import annotations

import cv2
import numpy as np
from pathlib import Path

SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}


def imread_unicode(path) -> np.ndarray:
    """Read an image from a path that may contain non-ASCII characters."""
    buf = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"cannot decode {path}")
    return img


def imwrite_unicode(path, img, quality: int = 95) -> None:
    ext = Path(path).suffix.lower() or ".jpg"
    params = [cv2.IMWRITE_JPEG_QUALITY, quality] if ext in (".jpg", ".jpeg") else []
    ok, buf = cv2.imencode(ext, img, params)
    if not ok:
        raise ValueError(f"cannot encode {path}")
    buf.tofile(str(path))


def enhance_pill_image(img_path) -> np.ndarray:
    img = imread_unicode(img_path)
    # 1. grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 2. CLAHE
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    clahe_img = clahe.apply(gray)
    # 3. Laplacian sharpen
    kernel_sharpen = np.array([[-1, -1, -1],
                               [-1,  9, -1],
                               [-1, -1, -1]])
    sharpened = cv2.filter2D(clahe_img, -1, kernel_sharpen)
    # 4. gamma correction
    gamma = 1.5
    lut = np.array([((i / 255.0) ** (1.0 / gamma)) * 255
                    for i in range(256)], dtype=np.uint8)
    return cv2.LUT(sharpened, lut)


def iter_images(root: Path):
    for p in sorted(root.rglob("*")):
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS:
            yield p
