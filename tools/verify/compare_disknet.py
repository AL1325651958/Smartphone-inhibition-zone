"""核验 DiskNet-small@224 在 App 里跑出来的结果。

三件事：
 1) C#（Verify.exe cls）与「训练/评估管线的 Python 参考实现」是否等价
    —— 参考实现 = 灰度 → PIL BICUBIC 224 → 5 视图（identity/hflip/vflip/±10°，fill 128）
       → logits 平均 → softmax，与 药片CNN/ensemble_eval.py 的 predict() 一致；
 2) 准确率：数据集自带的 224 测试图（28 张，batch-disjoint 划分）应复现 0.9286；
 3) App 真实输入路径：同一批的**原始裁剪**（未增强）走同一管线，看准确率掉多少。

用法：
    python compare_disknet.py
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
VER = ROOT / "Antibacterial zone" / "_verify"
ONNX = ROOT / "Antibacterial zone" / "_new_models" / "class_disknet.onnx"
CLASSES = ["CRO", "DA", "E", "LEV", "LZD", "P", "VA"]
SIZE = 224


def imread_u(p: Path) -> np.ndarray:
    return cv2.imdecode(np.fromfile(str(p), np.uint8), cv2.IMREAD_UNCHANGED)


def to_gray(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return img.astype(np.uint8)
    if img.shape[2] == 4:
        img = img[:, :, :3]
    return cv2.cvtColor(img[:, :, ::-1], cv2.COLOR_RGB2GRAY)   # 与 BT.601 一致


def views(im: Image.Image):
    return [im,
            im.transpose(Image.FLIP_LEFT_RIGHT),
            im.transpose(Image.FLIP_TOP_BOTTOM),
            im.rotate(-10, resample=Image.BICUBIC, fillcolor=128),
            im.rotate(+10, resample=Image.BICUBIC, fillcolor=128)]


def main() -> None:
    sess = ort.InferenceSession(str(ONNX), providers=["CPUExecutionProvider"])
    iname = sess.get_inputs()[0].name
    onames = [o.name for o in sess.get_outputs()]
    li = onames.index("logits") if "logits" in onames else 0

    def ref_predict(path: Path):
        g = to_gray(imread_u(path))
        im = Image.fromarray(g).resize((SIZE, SIZE), Image.BICUBIC)
        xs = [np.asarray(v.resize((SIZE, SIZE), Image.BICUBIC), np.float32) for v in views(im)]
        X = np.repeat(np.stack(xs)[:, None], 3, 1).astype(np.float32)      # [5,3,224,224]
        logits = sess.run(None, {iname: X})[li]                            # [5,7]
        m = logits.mean(0)
        e = np.exp(m - m.max())
        p = e / e.sum()
        return CLASSES[int(p.argmax())], float(p.max()), p

    for name, folder in (("dataset-224", VER / "disknet_test"), ("raw-crops", VER / "disknet_raw")):
        js = json.loads((VER / "out_disknet" / ("test224.json" if name == "dataset-224" else "raw.json")
                         ).read_text(encoding="utf-8"))
        csharp = {r["file"]: r for r in js}
        files = sorted(p for p in folder.glob("*") if p.suffix.lower() in (".png", ".jpg", ".jpeg"))
        n_ok_cs = n_ok_ref = n_agree = 0
        dps, bad = [], []
        for p in files:
            truth = p.name.split("__", 1)[0]
            lab, conf, _ = ref_predict(p)
            cs = csharp.get(p.name)
            if cs is None:
                continue
            if cs["label"] == truth: n_ok_cs += 1
            if lab == truth: n_ok_ref += 1
            if cs["label"] == lab:
                n_agree += 1
                dps.append(abs(cs["confidence"] - conf))
            else:
                bad.append((p.name, truth, cs["label"], round(cs["confidence"], 3), lab, round(conf, 3)))
        n = len(files)
        print(f"\n=== {name}（{n} 张）===")
        print(f"  C#       准确率 {n_ok_cs}/{n} = {n_ok_cs/n:.4f}")
        print(f"  参考实现 准确率 {n_ok_ref}/{n} = {n_ok_ref/n:.4f}")
        print(f"  C# ↔ 参考 top-1 一致率 {n_agree}/{n} = {n_agree/n:.1%}；"
              f"同类别置信度差 均值 {np.mean(dps):.4f} 最大 {np.max(dps):.4f}" if dps else "  无共同项")
        for row in bad[:8]:
            print(f"    不一致: {row[0][:44]:<46} 真值={row[1]:<4} C#={row[2]}({row[3]})  参考={row[4]}({row[5]})")


if __name__ == "__main__":
    main()
