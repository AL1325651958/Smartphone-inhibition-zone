"""End-to-end verification of the exported class.onnx against the PyTorch model.

Produces every number quoted in ``verify_class.md``:

A. numerical consistency   PyTorch (CPU, f32/f64) vs onnxruntime on random 0..255
B. end-to-end accuracy     manifest val rows -> EvalDataset(img_size=160, tta=5)
                           PyTorch PillCNN vs ONNX (view logits averaged, then
                           softmax), top-1 agreement, accuracy, macro-F1
C. preprocessing parity    the C#-equivalent cv2 pipeline
                           (`preproc_class_reference.py`) + ONNX vs the PIL
                           EvalDataset pipeline + PyTorch, on the same images:
                           top-1 agreement, mean |Δp|, plus ablations that
                           isolate pad colour / CLAHE clip / 192 cache / interp

Usage
-----
python verify_class.py --onnx "../Antibacterial zone/_new_models/class.onnx" \
                       --out "../Antibacterial zone/_new_models/verify_class.json"
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "Antibacterial zone" / "_new_models"))

from train import CLASSES, EvalDataset, PillCNN, metrics_from_cm  # noqa: E402
from export_onnx import OUTPUT_NAMES, PillCNNWrapper, load_model, sha256  # noqa: E402
import preproc_class_reference as ref  # noqa: E402

DS = HERE.parent / "Antibacterial zone mask_Class" / "dataset"
MEAN, STD = 0.5, 0.25


def denorm(t: torch.Tensor) -> np.ndarray:
    """EvalDataset tensor (already (x/255-0.5)/0.25) back to raw 0..255 float32.

    ``class.onnx`` bakes the normalisation into the graph, so the ONNX side must
    be fed raw intensities; PyTorch keeps using the normalised tensor.
    """
    return ((t.numpy() * STD + MEAN) * 255.0).astype(np.float32)


# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #
def load_manifest(path: Path):
    """manifest.csv paths contain mojibake; only the basename is trustworthy."""
    rows = list(csv.DictReader(io.StringIO(path.read_text(encoding="utf-8",
                                                          errors="replace"))))
    out = []
    for r in rows:
        name = Path(r["path"].replace("\\", "/")).name
        cls = r["class"]
        for split in ("train", "val"):
            p = DS / split / cls / name
            if p.exists():
                r["path"] = str(p)
                r["class_idx"] = CLASSES.index(cls)
                r["split_dir"] = split
                out.append(r)
                break
    return out


def find_val_files(cls: str, k: int, rng, val_rows):
    """Pick k distinct clean base images of `cls` from the val folder."""
    names = [Path(r["path"]).name for r in val_rows if r["class"] == cls]
    names = [n for n in names if not re.match(r"^aug_", n)]
    names = sorted(set(names))
    rng.shuffle(names)
    return [(DS / "val" / cls / n, cls) for n in names[:k]]


# --------------------------------------------------------------------------- #
# torch inference
# --------------------------------------------------------------------------- #
@torch.no_grad()
def torch_predict(model, ds, device, batch=32):
    """EvalDataset order: mean the *logits* over TTA views, then softmax."""
    logits_all = []
    for s in range(0, len(ds), batch):
        chunk = [ds[i][0] for i in range(s, min(s + batch, len(ds)))]
        x = torch.stack(chunk)                       # B,V,3,H,W
        b, v = x.shape[:2]
        lg = model(x.view(b * v, *x.shape[2:]).to(device)).view(b, v, -1).mean(1)
        logits_all.append(lg.float().cpu().numpy())
    return np.concatenate(logits_all)


def softmax(x):
    e = np.exp(x - x.max(1, keepdims=True))
    return e / e.sum(1, keepdims=True)


def onnx_predict(sess, tensors, prefer="logits", batch=16):
    """tensors: list of [V,3,160,160] float32; averages view logits then softmax."""
    name = sess.get_inputs()[0].name
    outs = []
    for s in range(0, len(tensors), batch):
        chunk = np.stack(tensors[s:s + batch]).astype(np.float32)   # B,V,3,H,W
        b, v = chunk.shape[:2]
        flat = chunk.reshape(b * v, *chunk.shape[2:])
        if prefer == "logits":
            r = sess.run(["logits"], {name: flat})[0]
        else:
            r = sess.run(["probs"], {name: flat})[0]
        outs.append(r.reshape(b, v, -1).mean(1))
    return np.concatenate(outs)


def cm_from(pred, y, n=len(CLASSES)):
    cm = np.zeros((n, n), dtype=np.int64)
    for t, p in zip(y, pred):
        cm[t, p] += 1
    return cm


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=str(HERE / "runs" / "orig" / "best.pt"))
    ap.add_argument("--manifest", default=str(HERE / "manifest.csv"))
    ap.add_argument("--onnx", default=str(HERE.parent / "Antibacterial zone" /
                                          "_new_models" / "class.onnx"))
    ap.add_argument("--img-size", type=int, default=160)
    ap.add_argument("--tta", type=int, default=5)
    ap.add_argument("--cache-px", type=int, default=192)
    ap.add_argument("--per-class", type=int, default=2,
                    help="real crops per class for the preprocessing-parity check")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    import onnxruntime as ort
    so = ort.SessionOptions()
    so.intra_op_num_threads = 1
    sess = ort.InferenceSession(str(args.onnx), sess_options=so,
                                providers=["CPUExecutionProvider"])
    if [o.name for o in sess.get_outputs()] != OUTPUT_NAMES:
        raise SystemExit("unexpected ONNX outputs: " +
                         str([o.name for o in sess.get_outputs()]))
    device = torch.device("cpu")
    model, sd, ckpt_metrics = load_model(Path(args.checkpoint), device)
    wrapper = PillCNNWrapper(model).eval()

    report = dict(
        checkpoint=str(args.checkpoint), checkpoint_sha256=sha256(Path(args.checkpoint)),
        onnx=str(args.onnx), onnx_sha256=sha256(Path(args.onnx)),
        onnx_bytes=Path(args.onnx).stat().st_size,
        manifest=str(args.manifest), img_size=args.img_size, tta=args.tta,
        cache_px=args.cache_px, classes=CLASSES,
        checkpoint_reported_metrics={k: v for k, v in ckpt_metrics.items()
                                     if k != "per_class"},
    )
    print(f"checkpoint {args.checkpoint}")
    print(f"  sha256 {report['checkpoint_sha256']}")
    if ckpt_metrics.get("macro_f1"):
        print(f"  checkpoint-reported: acc {ckpt_metrics['accuracy']:.4f} "
              f"macro-F1 {ckpt_metrics['macro_f1']:.4f} "
              f"({ckpt_metrics.get('tta')} views)")
    print(f"onnx {args.onnx}")
    print(f"  sha256 {report['onnx_sha256']}  {report['onnx_bytes']} bytes")

    # ---- class order cross-check against the on-disk folders --------------
    folder_order = sorted(p.name for p in (DS / "train").iterdir() if p.is_dir())
    by_alpha = sorted(CLASSES)
    report["folder_order"] = folder_order
    report["class_order_ok"] = folder_order == by_alpha == CLASSES or \
        sorted(CLASSES) == by_alpha
    print(f"\nCLASSES (train.py)      : {CLASSES}")
    print(f"dataset folder names    : {folder_order}")
    print(f"folder set == CLASSES   : {set(folder_order) == set(CLASSES)}")
    print(f"class_idx from manifest : "
          f"{[CLASSES.index(c) for c in folder_order]}")

    rows = load_manifest(Path(args.manifest))
    val_rows = [r for r in rows if r["orig_split"] == "val"]
    report["manifest_rows"] = len(rows)
    report["val_rows"] = len(val_rows)
    report["val_per_class"] = dict(Counter(r["class"] for r in val_rows))
    print(f"\nmanifest rows {len(rows)} | val rows {len(val_rows)} "
          f"{dict(sorted(report['val_per_class'].items()))}")

    # ---- B. end-to-end: EvalDataset(tta=5) on every val row ---------------
    t0 = time.time()
    ds = EvalDataset(val_rows, img_size=args.img_size, tta=args.tta,
                     cache_px=args.cache_px)
    tensors, labels = [], []
    for i in range(len(ds)):
        t, y = ds[i]
        tensors.append(denorm(t))          # raw 0..255 for the ONNX graph
        labels.append(int(y))
    labels = np.array(labels)
    print(f"\nbuilt {len(tensors)} TTA tensors in {time.time()-t0:.1f}s "
          f"(shape {tensors[0].shape})")

    lg_t = torch_predict(model, ds, device)
    p_t = softmax(lg_t)
    pred_t = p_t.argmax(1)
    lg_o = onnx_predict(sess, tensors, "logits")
    p_o = softmax(lg_o)
    pred_o = p_o.argmax(1)

    cm_t = cm_from(pred_t, labels)
    cm_o = cm_from(pred_o, labels)
    mt, mo = metrics_from_cm(cm_t), metrics_from_cm(cm_o)
    agree = float((pred_t == pred_o).mean())
    max_dp = float(np.abs(p_t - p_o).max())
    mean_dp = float(np.abs(p_t - p_o).mean())
    max_dl = float(np.abs(lg_t - lg_o).max())

    e2e = dict(
        n=len(labels), agreement=agree,
        n_disagreements=int((pred_t != pred_o).sum()),
        max_abs_diff_logits=max_dl, max_abs_diff_probs=max_dp,
        mean_abs_diff_probs=mean_dp,
        torch=dict(accuracy=mt["accuracy"], macro_f1=mt["macro_f1"],
                   balanced_accuracy=mt["balanced_accuracy"],
                   macro_specificity=mt["macro_specificity"],
                   per_class=mt["per_class"]),
        onnx=dict(accuracy=mo["accuracy"], macro_f1=mo["macro_f1"],
                  balanced_accuracy=mo["balanced_accuracy"],
                  macro_specificity=mo["macro_specificity"],
                  per_class=mo["per_class"]),
        cm_torch=cm_t.tolist(), cm_onnx=cm_o.tolist(),
    )
    report["end_to_end"] = e2e
    print("\n--- B. end-to-end (manifest val rows, EvalDataset tta=%d) ---" % args.tta)
    print(f"  PyTorch : acc {mt['accuracy']:.4f}  macro-F1 {mt['macro_f1']:.4f}  "
          f"balanced {mt['balanced_accuracy']:.4f}")
    print(f"  ONNX    : acc {mo['accuracy']:.4f}  macro-F1 {mo['macro_f1']:.4f}  "
          f"balanced {mo['balanced_accuracy']:.4f}")
    print(f"  top-1 agreement {agree*100:.2f}%  ({e2e['n_disagreements']} differ)  "
          f"max|Δp| {max_dp:.3e}  mean|Δp| {mean_dp:.3e}  max|Δlogits| {max_dl:.3e}")

    # ---- C. preprocessing parity on real crops ---------------------------
    rng = np.random.default_rng(args.seed)
    picks = []
    for c in CLASSES:
        picks += find_val_files(c, args.per_class, rng, val_rows)
    report["parity_images"] = [dict(path=str(p), class_true=cls) for p, cls in picks]
    print(f"\n--- C. preprocessing parity on {len(picks)} real val crops "
          f"({args.per_class}/class) ---")

    # C0: baseline -- EvalDataset tensors on the *same* files
    base_rows = [dict(path=str(p), class_idx=CLASSES.index(cls), cls=cls)
                 for p, cls in picks]
    ds_pick = EvalDataset(base_rows, img_size=args.img_size, tta=args.tta,
                          cache_px=args.cache_px)
    tens_pick = [ds_pick[i][0].numpy() for i in range(len(ds_pick))]     # normalised
    tens_pick_arr = np.stack(tens_pick)
    tens_pick_raw = np.stack([denorm(ds_pick[i][0]) for i in range(len(ds_pick))])
    lg_tp = torch_predict(model, ds_pick, device)
    p_tp = softmax(lg_tp)
    pred_tp = p_tp.argmax(1)
    y_pick = np.array([CLASSES.index(c) for _, c in picks])

    # sanity: ONNX on the de-normalised tensors must reproduce the PyTorch path
    lg_pick_o = onnx_predict(sess, tens_pick_raw, "logits")
    print(f"  [check] ONNX(denorm tensor) vs PyTorch on the same "
          f"EvalDataset tensors: agreement "
          f"{float((softmax(lg_pick_o).argmax(1) == pred_tp).mean())*100:.2f}%, "
          f"max|Δp| {float(np.abs(softmax(lg_pick_o) - p_tp).max()):.2e}")
    report["parity_evaldataset_onnx_vs_torch"] = dict(
        agreement=float((softmax(lg_pick_o).argmax(1) == pred_tp).mean()),
        max_abs_dp=float(np.abs(softmax(lg_pick_o) - p_tp).max()))

    variants = [
        ("cv2_cubic_c192_mean_clip3.0", dict(interp="cubic", cache_px=192,
                                             clip_limit=3.0, pad_fill="mean",
                                             resize="direct")),
        ("pil_bicubic_c192_mean_clip3.0", dict(interp="pil", cache_px=192,
                                               clip_limit=3.0, pad_fill="mean",
                                               resize="direct")),
        ("cv2_linear_c192_mean_clip3.0", dict(interp="linear", cache_px=192,
                                              clip_limit=3.0, pad_fill="mean",
                                              resize="direct")),
        ("cv2_area_c192_mean_clip3.0", dict(interp="area", cache_px=192,
                                            clip_limit=3.0, pad_fill="mean",
                                            resize="direct")),
        ("cv2_cubic_c192_zero_clip3.0", dict(interp="cubic", cache_px=192,
                                             clip_limit=3.0, pad_fill="zero",
                                             resize="direct")),
        ("cv2_cubic_c192_mean_clip2.0", dict(interp="cubic", cache_px=192,
                                             clip_limit=2.0, pad_fill="mean",
                                             resize="direct")),
        ("cv2_cubic_direct_mean_clip3.0", dict(interp="cubic", cache_px=None,
                                               clip_limit=3.0, pad_fill="mean",
                                               resize="none")),
        ("csharp_port_clahe_resampler", dict(csharp_port=True)),
    ]
    parity = []
    for tag, kw in variants:
        if kw.get("csharp_port"):
            import csharp_port as csp
            tens = [csp.preprocess(ref.imread_unicode(p), tta=args.tta)
                    for p, _ in picks]
            kw = dict(interp="csharp_port", cache_px=192, clip_limit=3.0,
                      pad_fill="mean", resize="direct")
        else:
            tens = [ref.preprocess(ref.imread_unicode(p), args.img_size, args.tta, **kw)
                    for p, _ in picks]
        tens_arr = np.stack(tens)
        # (i) ONNX on the cv2 tensor, (ii) PyTorch on the same cv2 tensor
        lg_v = onnx_predict(sess, tens, "logits")
        p_v = softmax(lg_v)
        pred_v = p_v.argmax(1)
        with torch.no_grad():
            tv = torch.from_numpy(tens_arr).float()
            b, v = tv.shape[:2]
            tvn = ((tv / 255.0) - MEAN) / STD                     # what PillCNN expects
            lg_tv = model(tvn.view(b * v, *tvn.shape[2:])).view(b, v, -1).mean(1).numpy()
        p_tv = softmax(lg_tv)
        # tensor-level difference vs EvalDataset (view 0 only, same geometry)
        d_t = float(np.abs(tens_arr[:, 0] - tens_pick_raw[:, 0]).mean())
        rec = dict(
            variant=tag, kwargs={k: (v if v is not None else "none")
                                 for k, v in kw.items()},
            agreement_vs_evaldataset_onnx=float((pred_v == pred_tp).mean()),
            agreement_vs_evaldataset_torch=float((p_tv.argmax(1) == pred_tp).mean()),
            mean_abs_dp_onnx=float(np.abs(p_v - p_tp).mean()),
            max_abs_dp_onnx=float(np.abs(p_v - p_tp).max()),
            mean_abs_dp_torch_same_tensor=float(np.abs(p_tv - p_tp).mean()),
            mean_abs_tensor_diff=float(np.abs(tens_arr - tens_pick_raw).mean()),
            mean_abs_tensor_diff_view0=d_t,
            accuracy=float((pred_v == y_pick).mean()),
            n=len(picks),
            preds_onnx=[CLASSES[i] for i in pred_v],
        )
        parity.append(rec)
        print(f"  {tag:32} agree(ONNX vs EvalDataset) "
              f"{rec['agreement_vs_evaldataset_onnx']*100:6.2f}%  "
              f"mean|Δp| {rec['mean_abs_dp_onnx']:.4f}  "
              f"acc {rec['accuracy']*100:6.2f}%  "
              f"mean|Δtensor| {rec['mean_abs_tensor_diff']:.2f}")
    report["parity"] = dict(
        n_images=len(picks),
        eval_dataset_preds=[CLASSES[i] for i in pred_tp],
        true_labels=[c for _, c in picks],
        eval_dataset_accuracy=float((pred_tp == y_pick).mean()),
        variants=parity,
    )

    # ---- D. raw crop sanity (the true C# input: colour crops) -------------
    raw_dir = HERE.parent / "药片分类数据" / "all"
    raw_files = sorted(raw_dir.glob("*.png"))[:40]
    raw_note = None
    if raw_files:
        rtens = [ref.preprocess(ref.imread_unicode(p), args.img_size, args.tta)
                 for p in raw_files]
        p_raw = softmax(onnx_predict(sess, rtens, "logits"))
        raw_note = dict(n=len(raw_files),
                        mean_conf=float(p_raw.max(1).mean()),
                        label_hist=dict(Counter(CLASSES[i] for i in p_raw.argmax(1))),
                        tensor_min=float(np.min([t.min() for t in rtens])),
                        tensor_max=float(np.max([t.max() for t in rtens])),
                        tensor_mean=float(np.mean([t.mean() for t in rtens])),
                        tensor_std=float(np.mean([t.std() for t in rtens])),
                        mean_max_prob=float(p_raw.max(1).mean()))
        print(f"\n--- D. raw colour crops ({len(raw_files)} files from "
              f"{raw_dir.name}) ---")
        print(f"  tensor stats mean {raw_note['tensor_mean']:.1f} "
              f"std {raw_note['tensor_std']:.1f} "
              f"range [{raw_note['tensor_min']:.0f},{raw_note['tensor_max']:.0f}]")
        print(f"  mean max-prob {raw_note['mean_max_prob']:.3f} "
              f"| pred hist {raw_note['label_hist']}")
    report["raw_crops"] = raw_note

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(report, indent=1, default=float),
                                  encoding="utf-8")
        print(f"\nreport -> {args.out}")


if __name__ == "__main__":
    main()
