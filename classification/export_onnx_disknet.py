"""Export the DiskNet-small @224 disk-abbreviation classifier to ONNX for the app.

Why DiskNet-small @224 (and not PillCNN @160):
  the revised manuscript describes the deployed classifier as DiskNet-small with
  $224\\times224$ inputs and 3.89 M parameters (Methods + Additional file Table S5),
  and reports the batch-stratified test performance of exactly this checkpoint:
  0.857 without test-time augmentation, 0.929 with 5-view TTA
  (`runs/disknet224_small/results.json`, `runs/ensemble_1.json` with `--tta 5`).

ONNX contract (consumed by `PillClassifier.cs` in the MAUI app):
  input   image  : float32 [N, 3, 224, 224]  greyscale replicated to 3 channels, values 0..255
                                              ((x/255 - 0.5) / 0.25 baked into the graph)
  outputs logits : float32 [N, 7]            raw logits (average the 5 TTA views, then softmax)
          probs  : float32 [N, 7]            softmax probabilities
  classes : CRO, DA, E, LEV, LZD, P, VA

Usage
-----
python export_onnx_disknet.py                        # default checkpoint, writes to the app staging dir
python export_onnx_disknet.py --checkpoint runs/ens_small_s1/best.pt --out class.onnx
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from disknet_model import DiskNet          # noqa: E402
from train_augmented import CLASSES, MEAN, STD  # noqa: E402

DEFAULT_CKPT = HERE / "runs" / "disknet224_small" / "best.pt"
DEFAULT_OUT = (HERE.parent / "Antibacterial zone" / "_new_models" / "class_disknet.onnx")
IMG_SIZE = 224


class Wrapped(torch.nn.Module):
    """Bake the 0..255 -> (x/255-0.5)/0.25 normalisation into the graph."""

    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor):
        x = (x / 255.0 - MEAN) / STD
        logits = self.model(x)
        return logits, torch.softmax(logits, 1)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def make_exportable(model: torch.nn.Module) -> None:
    """Re-express HighPass so the graph has static convolution kernels.

    `HighPass.forward` builds its depthwise DoG kernels on the fly with
    `.repeat(c, 1, 1, 1)`, which leaves the ONNX exporter with "kernel of unknown
    shape".  Registering the kernels as buffers and using them in a patched
    forward is numerically identical (asserted below) and exports cleanly.
    """
    import types

    import torch.nn.functional as F

    hp = model.highpass
    with torch.no_grad():
        ks = hp.k_small.squeeze().view(1, 1, 1, -1).repeat(3, 1, 1, 1).contiguous()
        kl = hp.k_large.squeeze().view(1, 1, 1, -1).repeat(3, 1, 1, 1).contiguous()
    hp.register_buffer("ks_static", ks, persistent=False)
    hp.register_buffer("kl_static", kl, persistent=False)

    def forward(self, x):
        c = x.shape[1]
        blur_s = F.conv2d(F.pad(x, (2, 2, 0, 0), mode="replicate"),
                          self.ks_static, groups=c)
        blur_l = F.conv2d(F.pad(x, (4, 4, 0, 0), mode="replicate"),
                          self.kl_static, groups=c)
        return self.proj(torch.cat([x - blur_s, blur_s - blur_l], dim=1))

    hp.forward = types.MethodType(forward, hp)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, default=DEFAULT_CKPT)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--check", type=int, default=6, help="number of random inputs for verification")
    args = ap.parse_args()

    ckpt = args.checkpoint.resolve()
    sd = torch.load(ckpt, map_location="cpu", weights_only=False)
    variant = sd.get("variant", "small")
    model = DiskNet(len(CLASSES), variant)
    model.load_state_dict(sd["state"])
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"checkpoint {ckpt}")
    print(f"variant {variant} | params {n_params:,} | classes {CLASSES}")

    wrapped = Wrapped(model).eval()

    # the patch must not change the model's output
    with torch.no_grad():
        probe = torch.rand(2, 3, IMG_SIZE, IMG_SIZE) * 255.0
        before = wrapped(probe)[0].clone()
        make_exportable(model)
        after = wrapped(probe)[0].clone()
        drift = float((before - after).abs().max())
    print(f"HighPass re-expression drift: {drift:.3e}")
    if drift > 1e-5:
        raise SystemExit("patched HighPass changed the model output")

    args.out.parent.mkdir(parents=True, exist_ok=True)

    dummy = torch.rand(1, 3, IMG_SIZE, IMG_SIZE) * 255.0
    torch.onnx.export(
        wrapped, dummy, str(args.out),
        input_names=["image"], output_names=["logits", "probs"],
        dynamic_axes={"image": {0: "N"}, "logits": {0: "N"}, "probs": {0: "N"}},
        opset_version=args.opset, do_constant_folding=True,
        dynamo=False,          # 经典 TorchScript 导出器：纯 opset-17 图，无需 onnxscript
    )
    print(f"wrote {args.out}  ({args.out.stat().st_size/1e6:.2f} MB)")

    # ---- verification: PyTorch (training pipeline) vs ONNX Runtime ----
    import onnxruntime as ort
    sess = ort.InferenceSession(str(args.out), providers=["CPUExecutionProvider"])
    iname = sess.get_inputs()[0].name
    max_dl = max_dp = 0.0
    disagree = 0
    for k in range(args.check):
        n = 1 if k % 2 == 0 else 5
        raw = torch.rand(n, 3, IMG_SIZE, IMG_SIZE) * 255.0
        with torch.no_grad():
            t_logits, t_probs = wrapped(raw)
        o_logits, o_probs = sess.run(None, {iname: raw.numpy().astype(np.float32)})
        max_dl = max(max_dl, float(np.abs(t_logits.numpy() - o_logits).max()))
        max_dp = max(max_dp, float(np.abs(t_probs.numpy() - o_probs).max()))
        disagree += int((t_probs.numpy().argmax(1) != o_probs.argmax(1)).sum())
    print(f"verification: max |dlogits| {max_dl:.3e}  max |dprobs| {max_dp:.3e}  "
          f"top-1 disagreements {disagree}")

    report = {
        "checkpoint": str(ckpt),
        "checkpoint_sha256": sha256(ckpt),
        "variant": variant,
        "params": n_params,
        "onnx": str(args.out),
        "onnx_sha256": sha256(args.out),
        "onnx_bytes": args.out.stat().st_size,
        "opset": args.opset,
        "input": {"name": "image", "dtype": "float32", "shape": ["N", 3, IMG_SIZE, IMG_SIZE],
                  "range": "0..255 greyscale replicated to 3 channels"},
        "outputs": [{"name": "logits", "shape": ["N", 7]}, {"name": "probs", "shape": ["N", 7]}],
        "classes": CLASSES,
        "mean_std": [MEAN, STD],
        "max_abs_diff_logits": max_dl,
        "max_abs_diff_probs": max_dp,
        "top1_disagreements": disagree,
    }
    rp = args.out.with_name(args.out.stem + "_export_report.json")
    rp.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(f"wrote {rp}")
    if disagree or max_dp > 1e-4:
        raise SystemExit("ONNX verification FAILED")


if __name__ == "__main__":
    main()
