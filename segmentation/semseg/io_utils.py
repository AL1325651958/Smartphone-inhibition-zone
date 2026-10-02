"""
semseg.io_utils — 兼容非 ASCII 路径的图片读写

Windows 上 cv2.imread / cv2.imwrite 遇到中文路径会**静默失败**（返回 None / False，
不抛异常）。本项目路径含中文，所以统一走这里。
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def imread(path: str | Path, flags: int = cv2.IMREAD_COLOR) -> np.ndarray | None:
    p = Path(path)
    if not p.is_file():
        return None
    try:
        buf = np.fromfile(str(p), dtype=np.uint8)
    except OSError:
        return None
    if buf.size == 0:
        return None
    return cv2.imdecode(buf, flags)


def imwrite(path: str | Path, img: np.ndarray, quality: int | None = None) -> bool:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    ext = p.suffix.lower() or ".jpg"
    params: list[int] = []
    if quality is not None and ext in (".jpg", ".jpeg"):
        params = [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)]
    ok, buf = cv2.imencode(ext, img, params)
    if not ok:
        return False
    try:
        buf.tofile(str(p))
    except OSError:
        return False
    return True
