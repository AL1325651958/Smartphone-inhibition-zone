"""Export the project's own PillCNN disk classifier to ONNX for the .NET MAUI app.

Interface contract (fixed with the app side)
--------------------------------------------
input   ``image``  float32 ``[N, 3, 160, 160]``, value range **0..255**
        (the grayscale crop replicated to R=G=B; the training normalisation
        ``(x/255 - 0.5)/0.25`` is baked into the graph, so the caller feeds raw
        8-bit intensities).  ``N`` is dynamic, H/W are frozen at 160.
outputs ``logits`` float32 ``[N, 7]``  raw PillCNN output (NO softmax)
        ``probs``  float32 ``[N, 7]``  ``softmax(logits, axis=1)``
        Class order is ``train.CLASSES`` = CRO, DA, E, LEV, LZD, P, VA.

Two outputs are provided because the reference TTA in ``药片CNN/infer.py``
averages **logits** over the 5 views and softmaxes afterwards; averaging the
softmaxed probabilities is *not* equivalent, so the C# side needs the raw
logits to reproduce the reference pipeline exactly.

The script re-runs a numerical consistency check after export:
PyTorch logits/probs vs onnxruntime logits/probs on random and real inputs.

Examples
--------
python export_onnx.py
python export_onnx.py --checkpoint runs/orig/best.pt \
    --out "..\\Antibacterial zone\\_new_models\\class.onnx"
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from train import CLASSES, MEAN, STD, PillCNN  # noqa: E402

OPSET = 17
IMG_SIZE = 160
NCLS = len(CLASSES)
INPUT_NAME = "image"
OUTPUT_NAMES = ["logits", "probs"]
DEFAULT_CKPT = HERE / "runs" / "orig" / "best.pt"
DEFAULT_OUT = HERE.parent / "Antibacterial zone" / "_new_models" / "class.onnx"


class PillCNNWrapper(nn.Module):
    """PillCNN + baked normalisation, returning raw logits *and* probabilities.

    forward(images_0_255) -> (logits [N,7], probs [N,7])

    ``images_0_255`` is the grayscale crop (0..255) replicated to 3 channels by
    the caller.  Normalisation here is bit-identical to
    ``(to_tensor(x) - MEAN) / STD`` in ``train.EvalDataset``:
    ``to_tensor`` divides by 255, so ``(x/255 - 0.5) / 0.25 == (x - 127.5)/63.75``.
    We spell it as ``x/255`` first to keep the exact same float rounding order.
    """

    def __init__(self, model: nn.Module, mean: float = MEAN, std: float = STD):
        super().__init__()
        self.model = model
        self.mean = float(mean)
        self.std = float(std)

    def forward(self, images: torch.Tensor):
        x = (images / 255.0 - self.mean) / self.std
        logits = self.model(x)
        return logits, torch.softmax(logits, dim=1)


def load_model(ckpt: Path, device: torch.device):
    """Load PillCNN from a checkpoint written by train.py (or a bare state_dict)."""
    if not ckpt.exists():
        raise SystemExit(f"checkpoint not found: {ckpt}")
    obj = torch.load(ckpt, map_location="cpu", weights_only=False)
    if isinstance(obj, dict):
        if "state" in obj:
            sd = obj["state"]
        elif "model" in obj:
            sd = obj["model"]
        else:
            sd = obj
    else:
        sd = obj
    sd = {k: v for k, v in sd.items()}
    model = PillCNN()
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        raise SystemExit(f"checkpoint is missing weights: {missing[:5]} "
                         f"({len(missing)} keys)")
    if unexpected:
        raise SystemExit(f"checkpoint has unexpected keys: {unexpected[:5]} "
                         f"({len(unexpected)} keys)")
    model.eval()
    metrics = obj.get("metrics", {}) if isinstance(obj, dict) else {}
    return model.to(device), sd, metrics


def sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def md5(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def export(ckpt: Path, out: Path, device: torch.device, opset: int = OPSET,
           img_size: int = IMG_SIZE, verify_batch: int = 5, seed: int = 0):
    model, sd, metrics = load_model(ckpt, device)
    wrapper = PillCNNWrapper(model).to(device).eval()
    nparam = sum(p.numel() for p in model.parameters())

    dummy = torch.rand(1, 3, img_size, img_size, device=device) * 255.0
    out.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    # dynamo=False keeps the classic TorchScript-based exporter, which produces a
    # plain opset-17 graph without extra ORT contrib ops.
    torch.onnx.export(
        wrapper,
        (dummy,),
        str(out),
        export_params=True,
        opset_version=opset,
        do_constant_folding=True,
        input_names=[INPUT_NAME],
        output_names=OUTPUT_NAMES,
        dynamic_axes={INPUT_NAME: {0: "N"}, "logits": {0: "N"}, "probs": {0: "N"}},
        dynamo=False,
    )
    secs = time.time() - t0
    print(f"[export] wrote {out} in {secs:.1f}s  ({out.stat().st_size/1e6:.2f} MB)")
    return wrapper, sd, metrics, nparam, secs


def describe_onnx(out: Path):
    import onnx

    m = onnx.load(str(out))
    onnx.checker.check_model(m)
    opset = max(o.version for o in m.opset_import)
    ins, outs = [], []
    for vi in m.graph.input:
        tt = vi.type.tensor_type
        ins.append(dict(name=vi.name, dtype=tt.elem_type,
                        shape=[(d.dim_param or d.dim_value) for d in tt.shape.dim]))
    for vi in m.graph.output:
        tt = vi.type.tensor_type
        outs.append(dict(name=vi.name, dtype=tt.elem_type,
                         shape=[(d.dim_param or d.dim_value) for d in tt.shape.dim]))
    ops = sorted({n.op_type for n in m.graph.node})
    return dict(opset=opset, inputs=ins, outputs=outs, ops=ops,
                n_nodes=len(m.graph.node), size_bytes=out.stat().st_size)


def verify(out: Path, wrapper: nn.Module, device: torch.device, img_size: int = IMG_SIZE,
           batch: int = 5, n_random: int = 8, seed: int = 0):
    """PyTorch (CPU, float32 and float64) vs onnxruntime (CPU) on 0..255 inputs.

    The comparison deliberately runs PyTorch on **CPU**: the ONNX graph is executed
    by onnxruntime on the CPU execution provider, and comparing against a CUDA
    forward pass would fold GPU kernel non-determinism into the "export error".
    float64 is included so that float32 accumulation can be told apart from a
    wrong graph (where float64 would not help).
    """
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.intra_op_num_threads = 1
    sess = ort.InferenceSession(str(out), sess_options=so,
                                providers=["CPUExecutionProvider"])
    in_name = sess.get_inputs()[0].name
    out_names = [o.name for o in sess.get_outputs()]
    if out_names != OUTPUT_NAMES:
        raise SystemExit(f"unexpected ONNX output names {out_names}, "
                         f"expected {OUTPUT_NAMES}")

    wrapper_cpu = wrapper.to("cpu").eval()
    # NOTE: .double() mutates in place, so the float64 twin needs its own copy of
    # the network (otherwise the float32 reference pass would run in float64).
    model_f64 = copy.deepcopy(wrapper_cpu.model).double()
    wrapper_f64 = PillCNNWrapper(model_f64, wrapper_cpu.mean, wrapper_cpu.std).eval()
    rng = np.random.default_rng(seed)
    report = dict(by_shape=[], n_random_inputs=0)
    for b in ([1, batch] if batch != 1 else [1]):
        # random uniform, a realistic mid-range sample, and a constant image
        cases = [rng.uniform(0, 255, size=(b, 3, img_size, img_size)).astype(np.float32)]
        cases.append(rng.normal(170, 40, size=(b, 3, img_size, img_size))
                     .clip(0, 255).astype(np.float32))
        cases.append(np.full((b, 3, img_size, img_size), 127.5, dtype=np.float32))
        max_l = max_p = max_rel = max_l64 = 0.0
        n_top1_diff = 0
        for x in cases:
            xt = torch.from_numpy(x)
            with torch.no_grad():
                tl, tp = wrapper_cpu(xt.clone())
                tl64, _ = wrapper_f64(xt.clone().double())
                tl, tp = tl.numpy(), tp.numpy()
                tl64 = tl64.numpy()
            rl, rp = sess.run(OUTPUT_NAMES, {in_name: x})
            max_l = max(max_l, float(np.abs(tl - rl).max()))
            max_p = max(max_p, float(np.abs(tp - rp).max()))
            max_rel = max(max_rel, float((np.abs(tl - rl) /
                                          (np.abs(rl) + 1e-6)).max()))
            max_l64 = max(max_l64, float(np.abs(tl64 - rl).max()))
            n_top1_diff += int((tl.argmax(1) != rl.argmax(1)).sum())
            report["n_random_inputs"] += b
        report["by_shape"].append(dict(
            batch=b, max_abs_diff_logits=max_l, max_abs_diff_probs=max_p,
            max_rel_diff_logits=max_rel, max_abs_diff_logits_f64_vs_ort=max_l64,
            n_top1_disagreements=n_top1_diff, n_inputs=3 * b))
        print(f"[verify] N={b}: max|Δlogits|={max_l:.3e} (rel {max_rel:.2e})  "
              f"max|Δprobs|={max_p:.3e}  f64-vs-ORT {max_l64:.3e}  "
              f"top-1 mismatches {n_top1_diff}")
    report["max_abs_diff_logits"] = max(r["max_abs_diff_logits"] for r in report["by_shape"])
    report["max_abs_diff_probs"] = max(r["max_abs_diff_probs"] for r in report["by_shape"])
    report["max_rel_diff_logits"] = max(r["max_rel_diff_logits"] for r in report["by_shape"])
    report["max_abs_diff_logits_f64_vs_ort"] = max(
        r["max_abs_diff_logits_f64_vs_ort"] for r in report["by_shape"])
    report["n_top1_disagreements"] = sum(r["n_top1_disagreements"]
                                         for r in report["by_shape"])
    report["ok"] = (report["max_abs_diff_logits"] < 1e-4
                    and report["max_abs_diff_probs"] < 1e-4
                    and report["n_top1_disagreements"] == 0)
    wrapper.to(device)
    return report


def main():
    ap = argparse.ArgumentParser(description="Export PillCNN to ONNX for the MAUI app")
    ap.add_argument("--checkpoint", default=str(DEFAULT_CKPT))
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--opset", type=int, default=OPSET)
    ap.add_argument("--img-size", type=int, default=IMG_SIZE)
    ap.add_argument("--verify-batch", type=int, default=5,
                    help="second dynamic-batch size to test (1 and this)")
    ap.add_argument("--json", default=None, help="also dump the report as JSON")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt, out = Path(args.checkpoint).resolve(), Path(args.out).resolve()
    print(f"device {device} | checkpoint {ckpt}")

    wrapper, sd, metrics, nparam, secs = export(ckpt, out, device, args.opset,
                                                args.img_size, args.verify_batch)
    info = describe_onnx(out)

    print("\n--- ONNX graph ---")
    print(f"  opset           : {info['opset']}")
    print(f"  input           : {info['inputs'][0]['name']} "
          f"float32 {info['inputs'][0]['shape']}")
    print(f"  outputs         : " + ", ".join(
        f"{o['name']} float32 {o['shape']}" for o in info["outputs"]))
    print(f"  nodes/ops       : {info['n_nodes']} | {','.join(info['ops'])}")
    print(f"  file size       : {info['size_bytes']/1e6:.2f} MB  "
          f"({info['size_bytes']} bytes)")
    print(f"  checkpoint sha256: {sha256(ckpt)}")
    print(f"  onnx sha256     : {sha256(out)}")

    # shape contract
    ins = info["inputs"]
    assert len(ins) == 1, "model must have exactly one input"
    shp = ins[0]["shape"]
    assert shp[0] == "N" and [int(v) for v in shp[1:]] == [3, args.img_size, args.img_size], \
        f"bad input shape {shp}"
    outs = {o["name"]: o["shape"] for o in info["outputs"]}
    assert set(outs) == set(OUTPUT_NAMES), outs
    for n in OUTPUT_NAMES:
        assert outs[n][0] == "N" and int(outs[n][1]) == NCLS, (n, outs[n])
    print("  shape contract  : OK  (input [N,3,%d,%d]; logits/probs [N,%d])"
          % (args.img_size, args.img_size, NCLS))

    vr = verify(out, wrapper, device, args.img_size, args.verify_batch)
    verdict = "PASS" if vr["ok"] else "FAIL"
    print(f"\n[verify] PyTorch(CPU) vs ORT(CPU) over {vr['n_random_inputs']} inputs -> "
          f"{verdict}")
    print(f"  max|Δlogits| = {vr['max_abs_diff_logits']:.3e} "
          f"(relative {vr['max_rel_diff_logits']:.3e}; float64 PyTorch vs ORT "
          f"{vr['max_abs_diff_logits_f64_vs_ort']:.3e})")
    print(f"  max|Δprobs|  = {vr['max_abs_diff_probs']:.3e}")
    print(f"  top-1 mismatches = {vr['n_top1_disagreements']}   (threshold 1e-4)")

    report = dict(
        checkpoint=str(ckpt), checkpoint_sha256=sha256(ckpt),
        checkpoint_md5=md5(ckpt), checkpoint_bytes=ckpt.stat().st_size,
        checkpoint_metrics={k: v for k, v in metrics.items() if k != "per_class"},
        onnx=str(out), onnx_sha256=sha256(out), onnx_bytes=out.stat().st_size,
        opset=info["opset"], input=info["inputs"][0], outputs=info["outputs"],
        ops=info["ops"], n_nodes=info["n_nodes"], n_params=int(nparam),
        classes=CLASSES, export_secs=secs, verification=vr,
        torch=torch.__version__, img_size=args.img_size,
    )
    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=1, default=float),
                                   encoding="utf-8")
        print(f"[export] report -> {args.json}")
    if not vr["ok"]:
        raise SystemExit("numerical consistency check FAILED (see above)")
    return report


if __name__ == "__main__":
    main()
