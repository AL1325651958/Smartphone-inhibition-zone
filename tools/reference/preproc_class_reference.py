"""Pure-Python reference implementation of the C# classification preprocessing.

This file exists so that the .NET MAUI side (`PillClassifier.cs`) can be aligned
step by step against a known-good float pipeline.  It reproduces the semantics of
`药片CNN/train.py::EvalDataset`:

    BGR crop (any rectangle, 50-300 px)
      (1) grayscale          cv2.cvtColor(img, COLOR_BGR2GRAY)          # BT.601
      (2) CLAHE              clipLimit=3.0, tileGridSize=(8,8)
      (3) sharpen            cv2.filter2D, kernel [[-1,-1,-1],[-1,9,-1],[-1,-1,-1]]
                             (default BORDER_REFLECT_101)
      (4) gamma 1.5 LUT      lut[i] = ((i/255)^(1/1.5))*255  (uint8)
      (5) square pad with the *mean grey value of the enhanced image*, centred,
          side = max(w, h)
      (6) resize to CACHE_PX = 192  (training keeps a 192 px RAM cache)
      (7) TTA views at 160 px:
            v0 = base.resize(160)
            v1 = base.resize(176).crop(0,0,160,160)              # 1.10x, top-left
            v2 = base.resize(176).crop(16,16,176,176)            # 1.10x, bottom-right
            v3 = base.rotate(-12, fillcolor=128).resize(160)
            v4 = base.rotate(+12, fillcolor=128).resize(160)
          (no flips: the printed drug code is always upright)
      (8) replicate to 3 channels, float32, 0..255  ->  [V,3,160,160]

The tensor is fed to `class.onnx` **without** any further normalisation: the
`(x/255 - 0.5)/0.25` transform is baked into the exported graph.

Two resampling backends are offered, because PIL and OpenCV disagree slightly:
  --interp cubic  cv2.INTER_CUBIC  (default; what a C# implementation would port)
  --interp pil    PIL BICUBIC      (bit-exact with train.py::EvalDataset)
The difference between them is quantified in `verify_class.md`.

Examples
--------
python preproc_class_reference.py --image crop.png
python preproc_class_reference.py --dir crops/ --tta 5 --onnx class.onnx --json out.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

CLASSES = ["CRO", "DA", "E", "LEV", "LZD", "P", "VA"]
FULL_NAME = {"CRO": "Ceftriaxone", "DA": "Clindamycin", "E": "Erythromycin",
             "LEV": "Levofloxacin", "LZD": "Linezolid", "P": "Penicillin",
             "VA": "Vancomycin"}

IMG_SIZE = 160          # model input side
CACHE_PX = 192          # train.py cache_px default (max(192, img_size + 32))
MEAN, STD = 0.5, 0.25   # train.py normalisation (baked into class.onnx)
CLIP_LIMIT = 3.0        # imagepross.py / enhance_lib.py (NOT the app's old 2.0)
TILE_GRID = (8, 8)
GAMMA = 1.5
SHARPEN_KERNEL = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]], dtype=np.float32)
SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}

_GAMMA_LUT = np.array([((i / 255.0) ** (1.0 / GAMMA)) * 255 for i in range(256)],
                      dtype=np.uint8)


# --------------------------------------------------------------------------- #
# unicode-safe IO (cv2.imread/imwrite fail silently on Chinese paths)
# --------------------------------------------------------------------------- #
def imread_unicode(path, flags=cv2.IMREAD_COLOR) -> np.ndarray:
    buf = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(buf, flags)
    if img is None:
        raise ValueError(f"cannot decode {path}")
    return img


def imwrite_unicode(path, img) -> None:
    ext = Path(path).suffix.lower() or ".png"
    ok, buf = cv2.imencode(ext, img)
    if not ok:
        raise ValueError(f"cannot encode {path}")
    buf.tofile(str(path))


# --------------------------------------------------------------------------- #
# steps
# --------------------------------------------------------------------------- #
def to_gray(bgr: np.ndarray) -> np.ndarray:
    """(1) BT.601 grayscale. Accepts 1/3/4-channel input."""
    if bgr.ndim == 2:
        return bgr
    if bgr.shape[2] == 4:
        bgr = cv2.cvtColor(bgr, cv2.COLOR_BGRA2BGR)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)


def enhance(gray: np.ndarray, clip_limit: float = CLIP_LIMIT) -> np.ndarray:
    """(2)-(4) CLAHE -> 3x3 sharpen -> gamma 1.5 LUT. Input/output uint8 HxW."""
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=TILE_GRID)
    cl = clahe.apply(gray)
    sharp = cv2.filter2D(cl, -1, SHARPEN_KERNEL)     # BORDER_REFLECT_101 default
    return cv2.LUT(sharp, _GAMMA_LUT)


def pad_to_square(img: np.ndarray, fill: int | None = None) -> np.ndarray:
    """(5) centre-pad to max(w,h) with the image mean (EvalDataset semantics)."""
    h, w = img.shape[:2]
    n = max(w, h)
    if (w, h) == (n, n):
        return img
    if fill is None:
        fill = int(img.mean())
    canvas = np.full((n, n), fill, dtype=img.dtype)
    canvas[(n - h) // 2:(n - h) // 2 + h, (n - w) // 2:(n - w) // 2 + w] = img
    return canvas


_CV2_FLAG = {"cubic": cv2.INTER_CUBIC, "linear": cv2.INTER_LINEAR,
             "area": cv2.INTER_AREA}


def resize_cv2(img: np.ndarray, size: int, interp: str = "cubic") -> np.ndarray:
    return cv2.resize(img, (size, size), interpolation=_CV2_FLAG[interp])


def resize_pil(img: np.ndarray, size: int, interp: str = "cubic") -> np.ndarray:
    """PIL path, kept for bit-exact comparison with train.EvalDataset.

    Always returns a numpy array so that the view list stays homogeneous.
    """
    from PIL import Image
    flag = Image.BICUBIC if interp in ("cubic", "area") else Image.BILINEAR
    return np.asarray(Image.fromarray(np.ascontiguousarray(img)).resize((size, size),
                                                                       flag))


def rotate_pil(img: np.ndarray, angle: float, fill: int = 128) -> np.ndarray:
    from PIL import Image
    return np.asarray(Image.fromarray(np.ascontiguousarray(img))
                      .rotate(angle, resample=Image.BICUBIC, fillcolor=fill))


def clahe_csharp_port(gray: np.ndarray, clip_limit: float = CLIP_LIMIT,
                      tiles=(8, 8)) -> np.ndarray:
    """Line-by-line port of the App's ``PillClassifier.ApplyCLAHE``.

    It is *not* OpenCV's CLAHE: the App uses ``clip = (int)(clipLimit*area/256 +
    0.5)`` and re-distributes the excess with its own formula, and it interpolates
    between tile LUTs at integer tile centres ``tile*  (i+0.5)``.  Measured on the
    App's own 70 test crops this port reproduces the App's exported enhanced grey
    to MAE 0.16 / 84.6 % exact pixels, whereas ``cv2.createCLAHE`` differs by
    MAE 7.8 (~3 % of the range) and 68 % exact pixels.  Included so that both
    chains can be scored; see verify_class.md.
    """
    h, w = gray.shape
    tw = (w + tiles[0] - 1) // tiles[0]
    th = (h + tiles[1] - 1) // tiles[1]
    luts = np.zeros((tiles[1], tiles[0], 256), dtype=np.uint8)
    for ty in range(tiles[1]):
        for tx in range(tiles[0]):
            x0, x1 = tx * tw, min(tx * tw + tw, w)
            y0, y1 = ty * th, min(ty * th + th, h)
            area = (x1 - x0) * (y1 - y0)
            if area <= 0:
                continue
            hist = np.bincount(gray[y0:y1, x0:x1].ravel(), minlength=256).astype(np.int64)
            clip = max(1, int(clip_limit * area / 256.0 + 0.5))
            excess = int(np.maximum(hist - clip, 0).sum())
            hist = np.minimum(hist, clip)
            per_bin = excess // 256
            leftover = excess - per_bin * 256
            hist = hist + per_bin
            start = (256 - leftover) // 2
            hist[start:start + leftover] += 1
            cdf = np.cumsum(hist)
            luts[ty, tx] = np.minimum(
                255, (cdf * (255.0 / area) + 0.5).astype(np.int64)).astype(np.uint8)

    cx = tw * (np.arange(tiles[0]) + 0.5)
    cy = th * (np.arange(tiles[1]) + 0.5)
    out = np.zeros((h, w), dtype=np.uint8)
    for y in range(h):
        if y < cy[0]:
            ty1 = ty2 = 0
            yw = 0.0
        elif y >= cy[-1]:
            ty1 = ty2 = tiles[1] - 1
            yw = 0.0
        else:
            ty1 = int(np.searchsorted(cy, y, "right") - 1)
            ty2 = ty1 + 1
            yw = (y - cy[ty1]) / (cy[ty2] - cy[ty1])
        for x in range(w):
            if x < cx[0]:
                tx1 = tx2 = 0
                xw = 0.0
            elif x >= cx[-1]:
                tx1 = tx2 = tiles[0] - 1
                xw = 0.0
            else:
                tx1 = int(np.searchsorted(cx, x, "right") - 1)
                tx2 = tx1 + 1
                xw = (x - cx[tx1]) / (cx[tx2] - cx[tx1])
            v = int(gray[y, x])
            val = (luts[ty1, tx1, v] * (1 - xw) * (1 - yw)
                   + luts[ty1, tx2, v] * xw * (1 - yw)
                   + luts[ty2, tx1, v] * (1 - xw) * yw
                   + luts[ty2, tx2, v] * xw * yw)
            out[y, x] = min(255, max(0, int(val + 0.5)))
    return out


def enhance_csharp_port(bgr: np.ndarray) -> np.ndarray:
    """Steps (1)-(4) with the App's own CLAHE."""
    return cv2.LUT(cv2.filter2D(clahe_csharp_port(to_gray(bgr)), -1, SHARPEN_KERNEL),
                   _GAMMA_LUT)


def base_view(bgr: np.ndarray, cache_px: int = CACHE_PX, interp: str = "cubic",
              clip_limit: float = CLIP_LIMIT, pad_fill: str = "mean",
              resize: str = "direct"):
    """Steps (1)-(6): enhanced, square, resized to `cache_px`.

    `resize="direct"`  : square-pad at the source size, then resize to cache_px
                         (identical structure to EvalDataset).
    `resize=="none"`   : skip the intermediate 192 px stage (ablation only).
    `pad_fill="zero"`  : ablation of the pad colour.
    """
    gray = enhance(to_gray(bgr), clip_limit)
    fill = None if pad_fill == "mean" else 0
    sq = pad_to_square(gray, fill)
    if resize == "none" or cache_px is None:
        return sq
    if interp == "pil":
        return resize_pil(sq, cache_px)
    return resize_cv2(sq, cache_px)


def tta_views(base, img_size: int = IMG_SIZE, tta: int = 5, interp: str = "cubic"):
    """Step (7): the exact view set of EvalDataset(img_size, tta).

    Views 3/4 use PIL ``Image.rotate`` semantics (counter-clockwise, fill=128,
    no expansion) via `rotate_pil`, or the OpenCV equivalent via `_rotate_cv2`.
    """
    pil = interp == "pil"
    s = img_size
    if tta <= 1:
        return [resize_pil(base, s) if pil else resize_cv2(base, s, interp)]
    views = [resize_pil(base, s) if pil else resize_cv2(base, s, interp)]
    if pil:
        big = resize_pil(base, int(s * 1.10))
        views.append(big[:s, :s].copy())
        views.append(big[big.shape[0] - s:, big.shape[1] - s:].copy())
        for ang in (-12, 12):
            views.append(resize_pil(rotate_pil(base, ang, 128), s))
    else:
        big = resize_cv2(base, int(s * 1.10), interp)
        views.append(big[:s, :s].copy())
        views.append(big[big.shape[0] - s:, big.shape[1] - s:].copy())
        for ang in (-12, 12):
            views.append(resize_cv2(_rotate_cv2(base, ang, 128), s, interp))
    return views[: max(tta, 1)]


def _rotate_cv2(img: np.ndarray, angle: float, fill: int = 128) -> np.ndarray:
    """PIL `Image.rotate` (counter-clockwise, expand=False) equivalent in OpenCV."""
    h, w = img.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2.0 - 0.5, h / 2.0 - 0.5), angle, 1.0)
    return cv2.warpAffine(img, m, (w, h), flags=cv2.INTER_CUBIC,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=fill)


def to_tensor(views, dtype=np.float32) -> np.ndarray:
    """Step (8): stack views into [V, 3, 160, 160] float32, range 0..255."""
    arr = np.stack([np.asarray(v, dtype=dtype) for v in views])       # V,H,W
    arr = np.repeat(arr[:, None, :, :], 3, axis=1)                    # V,3,H,W
    return np.ascontiguousarray(arr.astype(dtype, copy=False))


def preprocess(bgr: np.ndarray, img_size: int = IMG_SIZE, tta: int = 5,
               cache_px: int = CACHE_PX, interp: str = "cubic",
               clip_limit: float = CLIP_LIMIT, pad_fill: str = "mean",
               resize: str = "direct") -> np.ndarray:
    """Full pipeline: BGR crop -> [V, 3, img_size, img_size] float32 0..255."""
    base = base_view(bgr, cache_px, interp, clip_limit, pad_fill, resize)
    return to_tensor(tta_views(base, img_size, tta, interp))


def preprocess_files(paths, **kw) -> np.ndarray:
    return np.stack([preprocess(imread_unicode(p), **kw) for p in paths])


# --------------------------------------------------------------------------- #
# onnx helpers (logits are averaged over views, exactly like infer.py)
# --------------------------------------------------------------------------- #
def make_session(onnx_path, providers=("CPUExecutionProvider",)):
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.intra_op_num_threads = 1
    return ort.InferenceSession(str(onnx_path), sess_options=so, providers=list(providers))


def predict_onnx(sess, tensor, prefer_logits: bool = True):
    """tensor [B,V,3,160,160] -> (label_idx, prob) using mean-over-views logits.

    Mirrors `infer.py::classify`: average the **logits** of the views, then
    softmax once.  (Averaging the probabilities is not equivalent.)
    """
    b, v = tensor.shape[:2]
    flat = tensor.reshape(b * v, *tensor.shape[2:]).astype(np.float32)
    names = {o.name for o in sess.get_outputs()}
    if prefer_logits and "logits" in names:
        logits = sess.run(["logits"], {sess.get_inputs()[0].name: flat})[0]
        logits = logits.reshape(b, v, -1).mean(1)
        e = np.exp(logits - logits.max(1, keepdims=True))
        probs = e / e.sum(1, keepdims=True)
    else:
        probs = sess.run(["probs"], {sess.get_inputs()[0].name: flat})[0]
        probs = probs.reshape(b, v, -1).mean(1)
    return probs.argmax(1), probs


# --------------------------------------------------------------------------- #
# cli
# --------------------------------------------------------------------------- #
def iter_inputs(args):
    if args.image:
        return [Path(p) for p in args.image]
    d = Path(args.dir)
    return sorted(p for p in d.rglob("*") if p.suffix.lower() in SUPPORTED_EXTS)


def main():
    ap = argparse.ArgumentParser(description="C#-equivalent PillCNN preprocessing")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--image", nargs="+", help="one or more image files")
    src.add_argument("--dir", help="folder of images (recursive)")
    ap.add_argument("--onnx", default=None, help="class.onnx; enables classification")
    ap.add_argument("--img-size", type=int, default=IMG_SIZE)
    ap.add_argument("--cache-px", type=int, default=CACHE_PX)
    ap.add_argument("--tta", type=int, default=1, help="1 or 5")
    ap.add_argument("--interp", choices=["cubic", "linear", "area", "pil"],
                    default="cubic",
                    help="cubic/linear/area = OpenCV flags; pil = PIL BICUBIC "
                         "(bit-exact with train.EvalDataset)")
    ap.add_argument("--clip-limit", type=float, default=CLIP_LIMIT)
    ap.add_argument("--pad-fill", choices=["mean", "zero"], default="mean")
    ap.add_argument("--resize", choices=["direct", "none"], default="direct")
    ap.add_argument("--json", default=None)
    ap.add_argument("--save-tensors", default=None,
                    help="directory to dump the [1,3,160,160] base view as .npy")
    args = ap.parse_args()

    paths = iter_inputs(args)
    if not paths:
        raise SystemExit("no input images")
    sess = make_session(args.onnx) if args.onnx else None

    rows = []
    for p in paths:
        bgr = imread_unicode(p)
        t = preprocess(bgr, args.img_size, args.tta, args.cache_px, args.interp,
                       args.clip_limit, args.pad_fill, args.resize)
        stat = dict(min=float(t.min()), max=float(t.max()), mean=float(t.mean()),
                    std=float(t.std()), shape=list(t.shape))
        rec = dict(path=str(p), input_size=[int(v) for v in bgr.shape[:2]],
                   tensor=stat)
        if args.save_tensors:
            d = Path(args.save_tensors)
            d.mkdir(parents=True, exist_ok=True)
            np.save(d / (Path(p).stem + f"_tta{args.tta}.npy"), t)
        if sess is not None:
            idx, probs = predict_onnx(sess, t)
            k = int(idx[0])
            rec.update(label=CLASSES[k], antibiotic=FULL_NAME[CLASSES[k]],
                       confidence=float(probs[0, k]),
                       probabilities={c: float(x) for c, x in zip(CLASSES, probs[0])})
            print(f"{Path(p).name:44} -> {CLASSES[k]:4} "
                  f"({FULL_NAME[CLASSES[k]]:12}) p={probs[0, k]:.4f} | "
                  f"tensor min {stat['min']:.0f} max {stat['max']:.0f} "
                  f"mean {stat['mean']:.1f}")
        else:
            print(f"{Path(p).name:44} -> tensor {stat['shape']} "
                  f"min {stat['min']:.0f} max {stat['max']:.0f} mean {stat['mean']:.1f}")
        rows.append(rec)

    if args.json:
        Path(args.json).write_text(json.dumps(rows, ensure_ascii=False, indent=1),
                                   encoding="utf-8")
        print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
