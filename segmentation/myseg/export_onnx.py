"""
myseg/export_onnx.py — 导出 ONNX 并校验与 PyTorch 的一致性

    python -m myseg.export_onnx --weights runs_unet/<run>/best.pt
    python -m myseg.export_onnx --weights ... --check-image data/test/images/xxx.jpg

产出
----
export/<权重名>.onnx          固定 batch=1，opset 17
export/export_info.json       输入输出形状、参数量、一致性校验结果

为什么要做一致性校验
--------------------
ONNX 导出可能因为算子映射、精度、动态轴设置而出错。这里用同一张图分别跑
PyTorch 与 onnxruntime，比较输出的最大绝对差与 IoU；只有两者一致，才能说
「手机端 ONNX 的结果和论文里报的指标是同一个模型」。
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from myseg import config as C
    from myseg.model import build_model
    from myseg.io_utils import imread
else:
    from myseg import config as C
    from myseg.model import build_model
    from myseg.io_utils import imread


def main() -> None:
    ap = argparse.ArgumentParser(description="导出 ONNX 并校验一致性")
    ap.add_argument("--weights", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=C.EXPORT)
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--check-image", type=Path, default=None,
                    help="用来做一致性校验的图片；不指定就随机生成输入")
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()

    if not args.weights.is_file():
        raise SystemExit(f"找不到权重：{args.weights}")

    device = torch.device("cpu") if args.cpu or not torch.cuda.is_available() \
        else torch.device("cuda")

    ckpt = torch.load(args.weights, map_location="cpu", weights_only=False)
    encoder = ckpt.get("encoder", C.ENCODER)
    img_size = ckpt.get("img_size", C.IMG_SIZE)
    model = build_model(encoder, pretrained=False)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    args.out.mkdir(parents=True, exist_ok=True)
    onnx_path = args.out / f"{args.weights.stem}_{encoder}.onnx"

    dummy = torch.randn(1, 3, img_size, img_size)
    print(f"导出 {encoder} → {onnx_path}")
    t0 = time.time()
    torch.onnx.export(
        model, dummy, str(onnx_path),
        input_names=["images"], output_names=["logits"],
        opset_version=args.opset,
        dynamic_axes=None,            # 固定 batch=1，手机端部署更稳
        do_constant_folding=True,
    )
    print(f"  导出耗时 {time.time() - t0:.1f}s，"
          f"文件 {onnx_path.stat().st_size / 1e6:.2f} MB")

    # ── 结构校验 ──
    import onnx
    m = onnx.load(str(onnx_path))
    onnx.checker.check_model(m)
    print("  结构校验：通过")
    n_param = sum(int(np.prod(t.dims)) for t in m.graph.initializer)
    print(f"  ONNX 权重参数量：{n_param/1e6:.2f} M")

    # ── 一致性校验 ──
    import onnxruntime as ort
    providers = (["CUDAExecutionProvider", "CPUExecutionProvider"]
                 if "CUDAExecutionProvider" in ort.get_available_providers()
                 else ["CPUExecutionProvider"])
    sess = ort.InferenceSession(str(onnx_path), providers=providers)
    print(f"  推理 Provider：{providers[0]}")

    if args.check_image and args.check_image.is_file():
        import cv2
        img = imread(args.check_image)
        img = cv2.resize(img, (img_size, img_size))
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        mean = np.asarray(C.IMAGENET_MEAN, dtype=np.float32).reshape(3, 1, 1)
        std = np.asarray(C.IMAGENET_STD, dtype=np.float32).reshape(3, 1, 1)
        x = (np.transpose(rgb, (2, 0, 1)) - mean) / std
        x = x[None].astype(np.float32)
        src = f"真实图片 {args.check_image.name}"
    else:
        x = dummy.numpy().astype(np.float32)
        src = "随机输入"

    with torch.no_grad():
        pt_out = torch.sigmoid(model(torch.from_numpy(x))).numpy()
    ort_out = sess.run(None, {"images": x})[0]
    ort_out = 1.0 / (1.0 + np.exp(-ort_out))          # sigmoid

    max_abs = float(np.max(np.abs(pt_out - ort_out)))
    pt_bin = pt_out > C.MASK_THRESH
    ort_bin = ort_out > C.MASK_THRESH
    inter = float(np.logical_and(pt_bin, ort_bin).sum())
    union = float(np.logical_or(pt_bin, ort_bin).sum())
    iou = inter / union if union else 1.0

    print(f"\n  一致性校验（{src}）")
    print(f"    概率最大绝对差 : {max_abs:.3e}")
    print(f"    二值 mask IoU  : {iou:.6f}")
    ok = max_abs < 1e-4 and iou > 0.999
    print(f"    判定           : {'通过' if ok else '注意——差异偏大，需排查'}")

    # ── 速度 ──
    for _ in range(3):
        sess.run(None, {"images": x})
    t0 = time.time()
    for _ in range(10):
        sess.run(None, {"images": x})
    ms = (time.time() - t0) / 10 * 1000
    print(f"    onnxruntime 平均耗时：{ms:.1f} ms/张")

    info = {
        "weights": str(args.weights),
        "encoder": encoder,
        "img_size": img_size,
        "opset": args.opset,
        "onnx_file": str(onnx_path),
        "onnx_size_mb": round(onnx_path.stat().st_size / 1e6, 2),
        "params_m": round(n_param / 1e6, 2),
        "check_source": src,
        "prob_max_abs_diff": max_abs,
        "binary_mask_iou": iou,
        "consistent": bool(ok),
        "onnxruntime_ms": round(ms, 1),
        "providers": providers,
    }
    (args.out / "export_info.json").write_text(
        json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n信息已写入：{args.out / 'export_info.json'}")


if __name__ == "__main__":
    main()
