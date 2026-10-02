"""
semseg.export_onnx — 导出部署用 ONNX（预处理 + sigmoid 全部烘焙进计算图）

    cd "Antibacterial zone mask"
    python -m semseg.export_onnx --weights runs_semseg/20260918_051406_resnet18/best.pt \
                                 --out "_new_models/best.onnx" --check

为什么要把预处理烘焙进去
------------------------
App 端（.NET MAUI + Microsoft.ML.OnnxRuntime）不想再重复实现一遍 ImageNet 归一化，
所以导出的图直接吃 **0..255 的 RGB uint8 语义张量**（float32 存储），
内部完成 `(x/255 - mean) / std` 与 `sigmoid`。C# 端只要：

    var input  = new DenseTensor<float>(new[] {1, 3, 640, 640});   // R,G,B 平面，0..255
    var output = session.Run(...)[0].AsTensor<float>();            // [1,2,640,640]，已是概率
    // 通道 0 = Area(抑菌圈)，通道 1 = Yaoping(药片)

接口约定（与 C# 端逐条一致）
---------------------------
输入  : 单输入，float32，[1, 3, 640, 640]，数值范围 0..255，RGB（非 BGR），
        由原图**直接双线性 resize 到 640×640**（不做 letterbox 灰边）得到。
        —— 与 semseg/dataset.py 的预处理严格等价（后者是 /255 后再减均值除标准差）。
输出  : 单输出，float32，[1, 2, 640, 640]，**已经过逐通道 sigmoid 的概率**（0..1）。
固定形状 batch=1；opset 17；不使用 onnxruntime 不支持的算子。

产出
----
best.onnx               模型
best.onnx.json          旁挂元数据（infer.Predictor 会读取其中的 img_size）
export_info.json        导出与一致性核验信息

一致性核验
----------
同一输入分别跑 PyTorch（sigmoid 概率）与 onnxruntime，比较最大绝对差；
默认要求 < 1e-4，同时给出二值 mask 的 IoU。
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from semseg import config as C
    from semseg.io_utils import imread
    from semseg.model import build_model
else:
    from semseg import config as C
    from semseg.io_utils import imread
    from semseg.model import build_model

INPUT_NAME = "input"
OUTPUT_NAME = "probs"


# ────────────────────────────────────────────────
#  部署用包装：吃 0..255 RGB，吐 0..1 概率
# ────────────────────────────────────────────────
class BakedSegNet(nn.Module):
    """
    forward(x_0_255) -> probs(B,2,H,W)

        x = x_0_255 / 255                     # 与 dataset.py 的 /255.0 等价
        x = (x - mean) / std                  # ImageNet 归一化
        probs = sigmoid(net(x))               # 逐通道 sigmoid

    mean/std 由 C.IMAGENET_MEAN / C.IMAGENET_STD 生成，不写死字面量。
    """

    def __init__(self, net: nn.Module, mean=C.IMAGENET_MEAN, std=C.IMAGENET_STD):
        super().__init__()
        self.net = net
        m = torch.tensor(mean, dtype=torch.float32).view(1, 3, 1, 1)
        s = torch.tensor(std, dtype=torch.float32).view(1, 3, 1, 1)
        self.register_buffer("_mean", m, persistent=False)
        self.register_buffer("_std", s, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.float() / 255.0
        x = (x - self._mean) / self._std
        return torch.sigmoid(self.net(x)).float()


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(chunk), b""):
            h.update(blk)
    return h.hexdigest()


def load_model(weights: Path) -> tuple[nn.Module, str, int, dict]:
    if not weights.is_file():
        raise SystemExit(f"找不到权重：{weights}")
    ckpt = torch.load(weights, map_location="cpu", weights_only=False)
    encoder = ckpt.get("encoder", C.ENCODER)
    img_size = int(ckpt.get("img_size", C.IMG_SIZE))
    net = build_model(encoder, pretrained=False)
    net.load_state_dict(ckpt["model_state"])
    net.eval()
    return net, encoder, img_size, ckpt


def build_dummy(img_size: int, kind: str, check_image: Path | None,
                seed: int = 0) -> tuple[np.ndarray, str]:
    """返回 (NCHW float32 的 0..255 RGB 输入, 来源描述)。"""
    if check_image is not None and Path(check_image).is_file():
        import cv2
        img = imread(check_image)
        if img is not None:
            img = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
            return rgb.transpose(2, 0, 1)[None], f"真实图片 {Path(check_image).name}"
    rng = np.random.default_rng(seed)
    if kind == "photo":
        # 更接近真实动态范围：低频结构 + 噪声，避免全随机噪声的退化数值路径
        low = rng.random((1, 3, 8, 8), dtype=np.float32)
        import cv2
        big = cv2.resize(low[0].transpose(1, 2, 0), (img_size, img_size),
                         interpolation=cv2.INTER_CUBIC)
        x = np.clip(big * 255.0 + rng.normal(0, 6, (img_size, img_size, 3)), 0, 255)
        return x.transpose(2, 0, 1)[None].astype(np.float32), "合成彩色图(低频+噪声)"
    x = rng.integers(0, 256, size=(1, 3, img_size, img_size)).astype(np.float32)
    return x, "随机 0..255 输入"


def export(weights: Path, out: Path, opset: int, check_image: Path | None,
           cpu: bool, verify: bool) -> dict:
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)

    net, encoder, img_size, ckpt = load_model(weights)
    model = BakedSegNet(net).eval()

    x, src = build_dummy(img_size, "random", check_image)
    tx = torch.from_numpy(np.ascontiguousarray(x))

    print("=" * 78)
    print(f"权重      : {weights}")
    print(f"编码器    : {encoder}   img_size={img_size}  epoch={ckpt.get('epoch', '?')}")
    print(f"输出      : {out}")
    print(f"校验输入  : {src}")
    print("=" * 78)

    t0 = time.perf_counter()
    buf = io.BytesIO()
    torch.onnx.export(
        model, tx, buf,
        input_names=[INPUT_NAME], output_names=[OUTPUT_NAME],
        opset_version=opset,
        dynamic_axes=None,              # 固定形状 batch=1
        do_constant_folding=True,
        training=torch.onnx.TrainingMode.EVAL,
        export_params=True,
        dynamo=False,
    )
    onnx_bytes = buf.getvalue()
    # 中文路径安全：绕过 torch 直接写文件（torch 内部按文件名写，Windows 中文路径
    # 在部分版本上会失败；这里统一用 bytes 写盘）。
    with open(out, "wb") as f:
        f.write(onnx_bytes)
    dt = time.perf_counter() - t0
    print(f"导出完成：{dt:.1f}s，{len(onnx_bytes)/1e6:.2f} MB")

    # 旁挂元数据：infer.Predictor 会读 <onnx>.json 里的 img_size
    Path(str(out) + ".json").write_text(json.dumps(
        {"img_size": img_size, "encoder": encoder, "num_classes": C.NUM_CLASSES,
         "class_names": C.CLASS_NAMES, "input": INPUT_NAME, "output": OUTPUT_NAME,
         "input_range": "0..255 RGB float32", "output_semantics": "sigmoid probs",
         "thresh": C.THRESH, "opset": opset},
        indent=2, ensure_ascii=False), encoding="utf-8")

    # ── 结构校验 ──
    import onnx
    m = onnx.load_model_from_string(onnx_bytes)
    onnx.checker.check_model(m)
    ops = sorted({n.op_type for n in m.graph.node})
    print(f"结构校验  : 通过（opset {opset}，{len(m.graph.node)} 个节点，"
          f"算子 {len(ops)} 种）")
    print(f"算子集合  : {', '.join(ops)}")
    n_param = sum(int(np.prod(t.dims)) for t in m.graph.initializer)
    print(f"参数量    : {n_param/1e6:.2f} M（{len(m.graph.initializer)} 个 initializer）")

    def shape_of(vs):
        return [[d.dim_value if d.HasField("dim_value") else (d.dim_param or "?")
                 for d in v.type.tensor_type.shape.dim] for v in vs]
    in_shapes = shape_of(m.graph.input)
    out_shapes = shape_of(m.graph.output)
    in_names = [v.name for v in m.graph.input]
    out_names = [v.name for v in m.graph.output]
    print(f"输入      : {in_names} {in_shapes}")
    print(f"输出      : {out_names} {out_shapes}")

    info: dict = {
        "weights": str(weights),
        "weights_sha256": sha256_file(weights),
        "encoder": encoder,
        "img_size": img_size,
        "opset": opset,
        "onnx_file": str(out),
        "onnx_size_mb": round(len(onnx_bytes) / 1e6, 3),
        "onnx_sha256": hashlib.sha256(onnx_bytes).hexdigest(),
        "params_m": round(n_param / 1e6, 3),
        "input_names": in_names, "input_shapes": in_shapes,
        "output_names": out_names, "output_shapes": out_shapes,
        "ops": ops,
        "operators": len(m.graph.node),
        "check_source": src,
        "export_sec": round(dt, 2),
    }

    if verify:
        info.update(run_verify(out, model, x, src, img_size, cpu))
    return info


def run_verify(onnx_path: Path, model, x: np.ndarray, src: str, img_size: int,
               cpu: bool, n_time: int = 10) -> dict:
    import onnxruntime as ort
    avail = ort.get_available_providers()
    prov = ["CPUExecutionProvider"] if cpu else (
        ["CUDAExecutionProvider", "CPUExecutionProvider"]
        if "CUDAExecutionProvider" in avail else ["CPUExecutionProvider"])
    sess = ort.InferenceSession(str(onnx_path), providers=prov)
    in_name = sess.get_inputs()[0].name
    out_name = sess.get_outputs()[0].name

    tx = torch.from_numpy(np.ascontiguousarray(x))
    with torch.no_grad():
        pt = model(tx).numpy().astype(np.float32)
    ort_out = sess.run(None, {in_name: x})[0].astype(np.float32)

    assert pt.shape == ort_out.shape, f"形状不一致 {pt.shape} vs {ort_out.shape}"
    max_abs = float(np.max(np.abs(pt - ort_out)))
    mean_abs = float(np.mean(np.abs(pt - ort_out)))
    pt_bin, ort_bin = pt > C.THRESH, ort_out > C.THRESH
    inter = float(np.logical_and(pt_bin, ort_bin).sum())
    union = float(np.logical_or(pt_bin, ort_bin).sum())
    iou = inter / union if union else 1.0
    ok = max_abs < 1e-4

    print("\n一致性核验（PyTorch sigmoid 概率 vs onnxruntime 概率）")
    print(f"  提供方         : {sess.get_providers()}")
    print(f"  输入名/形状    : {in_name} {[d for d in sess.get_inputs()[0].shape]}")
    print(f"  输出名/形状    : {out_name} {[d for d in sess.get_outputs()[0].shape]}")
    print(f"  概率最大绝对差 : {max_abs:.3e}")
    print(f"  概率平均绝对差 : {mean_abs:.3e}")
    print(f"  二值 mask IoU  : {iou:.6f}  (阈值 {C.THRESH})")
    print(f"  判定           : {'通过 (<1e-4)' if ok else '不通过 —— 需排查'}")

    for _ in range(3):                                   # warmup
        sess.run(None, {in_name: x})
    ts = []
    for _ in range(n_time):
        t0 = time.perf_counter()
        sess.run(None, {in_name: x})
        ts.append((time.perf_counter() - t0) * 1000)
    ms = float(np.median(ts))
    print(f"  ORT 单张耗时   : 中位数 {ms:.1f} ms（{prov[0]}，warmup 后 {n_time} 次）")

    return {
        "prob_max_abs_diff": max_abs,
        "prob_mean_abs_diff": mean_abs,
        "binary_iou_at_thresh": iou,
        "consistent": bool(ok),
        "ort_providers": sess.get_providers(),
        "ort_ms_median": round(ms, 2),
        "ort_ms_all": [round(v, 2) for v in ts],
        "ort_input_name": in_name, "ort_output_name": out_name,
        "ort_input_shape": [d for d in sess.get_inputs()[0].shape],
        "ort_output_shape": [d for d in sess.get_outputs()[0].shape],
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description="导出部署用 ONNX（0..255 RGB 输入 / sigmoid 概率输出）")
    ap.add_argument("--weights", type=Path,
                    default=C.RUNS / "20260918_051406_resnet18" / "best.pt")
    ap.add_argument("--out", type=Path,
                    default=C.ROOT.parent / "Antibacterial zone" / "_new_models" / "best.onnx",
                    help="输出 onnx 文件路径（默认 Antibacterial zone/_new_models/best.onnx）")
    ap.add_argument("--check", type=Path, nargs="?", const="__AUTO__", default=None,
                    help="用真实图片做一致性核验；不带值时自动取 test 集第一张")
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--cpu", action="store_true", help="核验时只用 CPUExecutionProvider")
    ap.add_argument("--no-verify", dest="verify", action="store_false", default=True)
    args = ap.parse_args()

    check = args.check
    if check == "__AUTO__":
        tif = sorted((C.DATA / "test" / "images").glob("*.jpg"))
        check = tif[0] if tif else None
        if check is not None:
            print(f"自动选取核验图片：{check.name}")
    elif check is not None:
        check = Path(check)

    info = export(args.weights, args.out, args.opset, check, args.cpu, args.verify)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    info_path = out.parent / "export_info.json"
    info_path.write_text(json.dumps(info, indent=2, ensure_ascii=False),
                         encoding="utf-8")
    print(f"\n信息已写入：{info_path}")
    print(f"权重 sha256：{info['weights_sha256']}")
    print(f"ONNX sha256：{info['onnx_sha256']}")


if __name__ == "__main__":
    main()
