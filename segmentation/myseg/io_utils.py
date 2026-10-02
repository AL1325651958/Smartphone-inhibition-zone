"""
myseg/io_utils.py — 兼容中文（非 ASCII）路径的图片读写

为什么需要这个
--------------
Windows 上 OpenCV 的 cv2.imread / cv2.imwrite 走的是 ANSI 路径，遇到非 ASCII
字符会**静默失败**：imread 返回 None，imwrite 返回 False，都不抛异常。

本项目路径就含有中文（`...\\抑菌区域分析_返修\\Antibacterial zone mask\\`），
实测：
    cv2.imread("...抑菌区域分析_返修/.../x.jpg")            -> None
    cv2.imread(Path("...抑菌区域分析_返修/.../x.jpg"))      -> None   (pathlib 也没用)
    cv2.imdecode(np.fromfile(p, np.uint8), IMREAD_COLOR)     -> (4032, 3024, 3)  ✅

所以本包内一律使用下面两个包装函数，不再直接调用 cv2.imread / cv2.imwrite。
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def imread(path: str | Path, flags: int = cv2.IMREAD_COLOR) -> np.ndarray | None:
    """读图片，支持含中文/非 ASCII 的路径。读不了返回 None。"""
    p = Path(path)
    if not p.is_file():
        return None
    try:
        buf = np.fromfile(str(p), dtype=np.uint8)
    except OSError:
        return None
    if buf.size == 0:
        return None
    img = cv2.imdecode(buf, flags)
    return img


def imwrite(path: str | Path, img: np.ndarray, quality: int | None = None) -> bool:
    """
    写图片，支持含中文/非 ASCII 的路径。成功返回 True。
    quality 仅对 jpg 生效（默认 OpenCV 的 95）。
    """
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


def self_test(folder: str | Path) -> tuple[int, int]:
    """
    对目录下所有图片做一次读写自检。
    返回 (可读张数, 不可读张数)。用于确认中文路径问题已解决。
    """
    files = sorted(f for f in Path(folder).iterdir()
                   if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"})
    good = bad = 0
    for f in files:
        (good := good + 1) if imread(f) is not None else (bad := bad + 1)
    return good, bad
