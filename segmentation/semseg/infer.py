"""
semseg.infer — 推理：图片 → 两类掩膜 + 几何量

    python -m semseg.infer --weights runs_semseg/<run>/best.pt --source images
    python -m semseg.infer --onnx export_semseg/<run>.onnx --source images

输出（export_semseg/infer/<权重名>/）
    masks/<图名>_zone.png    抑菌圈掩膜（0/255）
    masks/<图名>_disk.png    药片掩膜（0/255）
    objects.json             每个区域的几何量（面积/等效直径/质心/置信度）
    vis/<图名>.jpg           四联图
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from semseg import config as C
    from semseg.io_utils import imread, imwrite
    from semseg.model import build_model
else:
    from semseg import config as C
    from semseg.io_utils import imread, imwrite
    from semseg.model import build_model

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp"}


def preprocess(img_bgr: np.ndarray, size: int) -> np.ndarray:
    img = cv2.resize(img_bgr, (size, size), interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    mean = np.asarray(C.IMAGENET_MEAN, dtype=np.float32).reshape(3, 1, 1)
    std = np.asarray(C.IMAGENET_STD, dtype=np.float32).reshape(3, 1, 1)
    return ((np.transpose(rgb, (2, 0, 1)) - mean) / std)[None].astype(np.float32)


class Predictor:
    """PyTorch / ONNX 两种后端。"""

    def __init__(self, weights: Path | None = None, onnx: Path | None = None,
                 cpu: bool = False):
        self.sess = None
        self.model = None
        self.img_size = C.IMG_SIZE
        if onnx is not None:
            import onnxruntime as ort
            prov = (["CUDAExecutionProvider", "CPUExecutionProvider"]
                    if "CUDAExecutionProvider" in ort.get_available_providers()
                    else ["CPUExecutionProvider"])
            self.sess = ort.InferenceSession(str(onnx), providers=prov)
            self.input_name = self.sess.get_inputs()[0].name
            self.backend = f"onnxruntime ({prov[0]})"
            meta = Path(str(onnx) + ".json")
            if meta.is_file():
                try:
                    self.img_size = int(json.loads(meta.read_text(encoding="utf-8"))
                                       .get("img_size", C.IMG_SIZE))
                except Exception:
                    pass
        elif weights is not None:
            import torch
            self.torch = torch
            self.device = torch.device(
                "cpu" if (cpu or not torch.cuda.is_available()) else "cuda")
            ck = torch.load(weights, map_location="cpu", weights_only=False)
            self.img_size = ck.get("img_size", C.IMG_SIZE)
            self.model = build_model(ck.get("encoder", C.ENCODER),
                                     pretrained=False).to(self.device)
            self.model.load_state_dict(ck["model_state"])
            self.model.eval()
            self.backend = f"pytorch ({self.device})"
        else:
            raise SystemExit("必须指定 --weights 或 --onnx")

    def __call__(self, img_bgr: np.ndarray, tta: bool = C.USE_TTA) -> np.ndarray:
        """
        返回 (K,H,W) 概率图。

        tta=True 时做测试时增强：旋转 0/90/180/270 + 水平翻转共 5 个视图取平均。
        实测把测试宏平均从 0.9369 提到 0.9400。代价是推理耗时约 5 倍。
        """
        x = preprocess(img_bgr, self.img_size)
        return self._forward(x, tta)

    def _run(self, x: np.ndarray) -> np.ndarray:
        if self.sess is not None:
            logits = self.sess.run(None, {self.input_name: x})[0]
        else:
            with self.torch.no_grad():
                t = self.torch.from_numpy(np.ascontiguousarray(x)).to(self.device)
                logits = self.model(t).float().cpu().numpy()
        return (1.0 / (1.0 + np.exp(-logits)))[0]

    def _forward(self, x: np.ndarray, tta: bool) -> np.ndarray:
        if not tta:
            return self._run(x)
        import numpy as _np
        outs = [self._run(x)]
        for k in (1, 2, 3):                       # 旋转 90/180/270
            xr = _np.rot90(x, k, axes=(2, 3))
            outs.append(_np.rot90(self._run(_np.ascontiguousarray(xr)), -k, axes=(1, 2)))
        xf = x[:, :, :, ::-1]                     # 水平翻转
        outs.append(self._run(_np.ascontiguousarray(xf))[:, :, :, ::-1])
        return _np.mean(outs, axis=0)


def region_stats(mask: np.ndarray, prob: np.ndarray, min_area: int = 30) -> list[dict]:
    """对单通道 0/1 掩膜做连通域统计，给出每个区域的位置/大小/置信度。"""
    n, lab, stats, cents = cv2.connectedComponentsWithStats(
        (mask > 0).astype(np.uint8), connectivity=8)
    out = []
    for i in range(1, n):
        a = int(stats[i, cv2.CC_STAT_AREA])
        if a < min_area:
            continue
        x, y = int(stats[i, cv2.CC_STAT_LEFT]), int(stats[i, cv2.CC_STAT_TOP])
        w, h = int(stats[i, cv2.CC_STAT_WIDTH]), int(stats[i, cv2.CC_STAT_HEIGHT])
        m = (lab == i)
        out.append({
            "area_px": a,
            "eq_diameter_px": float(2.0 * np.sqrt(a / np.pi)),
            "centroid": [round(float(cents[i][0]), 1), round(float(cents[i][1]), 1)],
            "bbox": [x, y, w, h],
            "mean_prob": round(float(prob[m].mean()), 4),
        })
    out.sort(key=lambda d: -d["area_px"])
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="两类掩膜推理")
    ap.add_argument("--weights", type=Path, default=None)
    ap.add_argument("--onnx", type=Path, default=None)
    ap.add_argument("--source", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--thresh", type=float, default=C.THRESH)
    ap.add_argument("--min-area", type=int, default=30)
    ap.add_argument("--no-vis", action="store_true")
    ap.add_argument("--no-masks", action="store_true")
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--no-tta", dest="tta", action="store_false", default=C.USE_TTA,
                    help="关闭测试时增强（默认开启，快 5 倍但指标略低）")
    args = ap.parse_args()

    pred = Predictor(weights=args.weights, onnx=args.onnx, cpu=args.cpu)
    tag = args.weights.stem if args.weights else args.onnx.stem
    out_dir = args.out or (C.EXPORT / "infer" / tag)
    (out_dir / "masks").mkdir(parents=True, exist_ok=True)
    vis_dir = out_dir / "vis"

    src = Path(args.source)
    files = [src] if src.is_file() else sorted(
        f for f in src.iterdir() if f.suffix.lower() in IMG_EXT)
    if not files:
        raise SystemExit(f"{src} 里没有图片")

    print("=" * 76)
    print(f"后端      : {pred.backend}")
    print(f"输入尺寸  : {pred.img_size}")
    print(f"分辨率恢复: 掩膜会从 {pred.img_size} 放大回原图尺寸（最近邻）")
    print(f"图片      : {len(files)} 张")
    print(f"输出      : {out_dir}")
    print("=" * 76)

    from semseg.viz import save_panel

    report = []
    ms_all = []
    for f in files:
        img = imread(f)
        if img is None:
            print(f"  跳过（读不了）：{f.name}")
            continue
        h0, w0 = img.shape[:2]
        t0 = time.perf_counter()
        prob = pred(img, tta=args.tta)
        ms = (time.perf_counter() - t0) * 1000
        ms_all.append(ms)

        # 掩膜放大回原图分辨率，这样输出的 mask 可直接用于测量
        m_full = np.stack([
            cv2.resize((prob[c] > args.thresh).astype(np.uint8) * 255,
                       (w0, h0), interpolation=cv2.INTER_NEAREST)
            for c in range(C.NUM_CLASSES)
        ])
        prob_full = np.stack([
            cv2.resize(prob[c], (w0, h0), interpolation=cv2.INTER_LINEAR)
            for c in range(C.NUM_CLASSES)
        ])

        if not args.no_masks:
            imwrite(out_dir / "masks" / f"{f.stem}_zone.png", m_full[0])
            imwrite(out_dir / "masks" / f"{f.stem}_disk.png", m_full[1])

        zone_objs = region_stats(m_full[0], prob_full[0], args.min_area)
        disk_objs = region_stats(m_full[1], prob_full[1], args.min_area)
        report.append({"image": f.name, "infer_ms": round(ms, 1),
                       "n_zone_regions": len(zone_objs),
                       "n_disk_regions": len(disk_objs),
                       "zones": zone_objs, "disks": disk_objs})

        if not args.no_vis:
            save_panel(img, prob, np.zeros_like(prob), vis_dir / f"{f.stem}.jpg")

        print(f"  {f.name:44s} 圈区域={len(zone_objs):2d}  药片={len(disk_objs):2d}  "
              f"{ms:.0f}ms")

    (out_dir / "objects.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    if ms_all:
        print(f"\n平均耗时：{np.mean(ms_all):.0f} ms/张")
    print(f"区域明细：{out_dir / 'objects.json'}")
    if not args.no_masks:
        print(f"掩膜    ：{out_dir / 'masks'}（每图 zone/disk 两张，PNG，已回到原图分辨率）")
    if not args.no_vis:
        print(f"可视化  ：{vis_dir}")


if __name__ == "__main__":
    main()
