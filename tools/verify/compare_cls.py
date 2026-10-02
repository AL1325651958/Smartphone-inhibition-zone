"""独立核验：C# 端（PillClassifier.cs）与「训练链参考实现」的 top-1 一致率。

参考实现 = 训练时的真实管线（cv2 增强 + PIL 缩放 + 5 视图 TTA + logits 平均 + softmax），
直接用 ONNX Runtime 跑 `_new_models/class.onnx`，与 C# 的 `cls_csharp*.json` 对比。

用法：
    python compare_cls.py <csharp_json> [<csharp_json> ...]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
CROPS = ROOT / "Antibacterial zone" / "_verify" / "cls_test"
ONNX = ROOT / "Antibacterial zone" / "_new_models" / "class.onnx"
CLASSES = ["CRO", "DA", "E", "LEV", "LZD", "P", "VA"]
MEAN, STD = 0.5, 0.25
SIZE, CACHE = 160, 192


def imread_u(p: Path) -> np.ndarray:
    return cv2.imdecode(np.fromfile(str(p), np.uint8), cv2.IMREAD_COLOR)


def enhance(bgr: np.ndarray) -> np.ndarray:
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    g = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(g)
    g = cv2.filter2D(g, -1, np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]], np.float32))
    lut = np.array([((i / 255.0) ** (1.0 / 1.5)) * 255 for i in range(256)], np.uint8)
    return cv2.LUT(g, lut)


def pad_square(g: np.ndarray) -> Image.Image:
    im = Image.fromarray(g, mode="L")
    w, h = im.size
    n = max(w, h)
    if (w, h) != (n, n):
        sq = Image.new("L", (n, n), int(g.mean()))
        sq.paste(im, ((n - w) // 2, (n - h) // 2))
        im = sq
    return im.resize((CACHE, CACHE), Image.BICUBIC)


def views(im: Image.Image, tta: int = 5):
    s = SIZE
    out = [im.resize((s, s), Image.BICUBIC)]
    if tta > 1:
        big = im.resize((int(s * 1.10), int(s * 1.10)), Image.BICUBIC)
        out.append(big.crop((0, 0, s, s)))
        out.append(big.crop((big.width - s, big.height - s, big.width, big.height)))
    if tta >= 5:
        for ang in (-12, 12):
            out.append(im.rotate(ang, resample=Image.BICUBIC, fillcolor=128)
                       .resize((s, s), Image.BICUBIC))
    return out[:max(tta, 1)]


def main():
    sess = ort.InferenceSession(str(ONNX), providers=["CPUExecutionProvider"])
    iname = sess.get_inputs()[0].name
    onames = [o.name for o in sess.get_outputs()]
    li = onames.index("logits") if "logits" in onames else 0

    files = sorted(p for p in CROPS.glob("*")
                   if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".bmp"))
    # 参考实现（批量 5 视图一次前向）
    ref = {}
    for f in files:
        bgr = imread_u(f)
        im = pad_square(enhance(bgr))
        xs = []
        for v in views(im, 5):
            a = np.asarray(v, np.float32)                 # 0..255
            a = np.repeat(a[None], 3, 0)                  # 3 通道
            xs.append(a)
        x = np.stack(xs, 0).astype(np.float32)            # [5,3,160,160]
        logits = sess.run(None, {iname: x})[li]           # [5,7]
        m = logits.mean(0)
        e = np.exp(m - m.max())
        p = e / e.sum()
        ref[f.name] = (CLASSES[int(p.argmax())], float(p.max()), p)

    for arg in sys.argv[1:]:
        js = json.loads(Path(arg).read_text(encoding="utf-8"))
        rows = {r["file"]: r for r in js}
        agree = 0
        dps, disagree = [], []
        for name, (lab, conf, _) in ref.items():
            r = rows.get(name)
            if not r:
                continue
            if r["label"] == lab:
                agree += 1
                dps.append(abs(r["confidence"] - conf))
            else:
                disagree.append((name, r["label"], r["confidence"], lab, conf))
        n = len(ref)
        print(f"\n=== {Path(arg).name} vs 训练链参考（{n} 张）===")
        print(f"top-1 一致率 = {agree}/{n} = {agree/n*100:.1f}%")
        print(f"同类别置信度差：平均 {np.mean(dps):.4f} / 最大 {np.max(dps):.4f}" if dps else "无")
        if disagree:
            print("不一致清单：")
            for name, cl, cc, pl, pc in disagree[:12]:
                print(f"  {name:<52} C#={cl} {cc:.3f}   ref={pl} {pc:.3f}")


if __name__ == "__main__":
    main()
