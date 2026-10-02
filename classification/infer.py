"""Classify antibiotic disks with a trained model (single image, folder, or plate).

The classifier operates on **disk crops**, which in the full pipeline come from
the project's instance-segmentation model (`写作/runs/segment`). Provide crops
with `--image` or `--dir`. The `--plate` mode is only a convenience fallback on
small, clean images (bright disks on a dark background); it is not a substitute
for the segmentation model on real plate photographs.

Examples
--------
python infer.py --checkpoint runs/orig/best.pt --image crop.jpg
python infer.py --checkpoint runs/orig/best.pt --dir some_folder/ --json out.json
python infer.py --checkpoint runs/orig/best.pt --plate segmented.png --out annotated.png
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from train import CLASSES, EvalDataset, PillCNN  # noqa: E402

FULL_NAME = {"CRO": "Ceftriaxone", "DA": "Clindamycin", "E": "Erythromycin",
             "LEV": "Levofloxacin", "LZD": "Linezolid", "P": "Penicillin",
             "VA": "Vancomycin"}


def load_model(ckpt, device):
    m = PillCNN().to(device)
    sd = torch.load(ckpt, map_location=device, weights_only=False)
    m.load_state_dict(sd["state"] if "state" in sd else sd)
    m.eval()
    return m, sd.get("metrics", {})


@torch.no_grad()
def classify(model, paths, device, img_size=160, blur=0.0, tta=5, batch=32):
    rows = [{"path": str(p), "class_idx": 0} for p in paths]
    ds = EvalDataset(rows, img_size=img_size, blur=blur, tta=tta)
    probs = []
    for s in range(0, len(ds), batch):
        chunk = [ds[i] for i in range(s, min(s + batch, len(ds)))]
        x = torch.stack([c[0] for c in chunk])
        b, v = x.shape[:2]
        x = x.view(b * v, *x.shape[2:]).to(device)
        logits = model(x).view(b, v, -1).mean(1)
        probs.append(F.softmax(logits.float(), 1).cpu().numpy())
    return np.concatenate(probs)


@torch.no_grad()
def classify_plate(model, plate_path, device, disk_diameter=None, quadrant=None,
                   img_size=160, tta=5):
    """Detect bright disk-like blobs on a plate and classify each one.

    Uses a simple threshold + connected-component pass (no extra dependencies),
    then falls back to a 7-disk ring layout if detection is ambiguous.
    """
    from PIL import Image, ImageDraw

    im = Image.open(plate_path).convert("L")
    a = np.asarray(im, dtype=np.float32)
    # disks are the brightest compact objects; require both a relative and an
    # absolute margin so that a uniformly bright plate does not select agar
    thr = max(float(np.percentile(a, 99.5)), float(a.mean() + 1.5 * a.std()))
    mask = a >= thr
    lab = np.zeros(a.shape, dtype=np.int32)
    boxes = []
    from collections import deque
    cur = 0
    ys, xs = np.nonzero(mask)
    for y0, x0 in zip(ys, xs):
        if lab[y0, x0]:
            continue
        cur += 1
        q = deque([(y0, x0)])
        lab[y0, x0] = cur
        pix = []
        while q:
            y, x = q.popleft()
            pix.append((y, x))
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    yy, xx = y + dy, x + dx
                    if 0 <= yy < a.shape[0] and 0 <= xx < a.shape[1] and \
                            mask[yy, xx] and not lab[yy, xx]:
                        lab[yy, xx] = cur
                        q.append((yy, xx))
        pix = np.array(pix)
        if len(pix) < 80:
            continue
        h = pix[:, 0].max() - pix[:, 0].min()
        w = pix[:, 1].max() - pix[:, 1].min()
        if h < 8 or w < 8 or h > a.shape[0] * 0.35 or w > a.shape[1] * 0.35:
            continue
        if not (0.55 < h / max(w, 1) < 1.8):
            continue
        boxes.append((pix[:, 1].min(), pix[:, 0].min(), pix[:, 1].max(), pix[:, 0].max()))

    boxes = sorted(boxes, key=lambda b: -((b[2] - b[0]) * (b[3] - b[1])))[:7]
    if len(boxes) < 5:
        raise SystemExit(
            f"only {len(boxes)} disk candidates found in {plate_path}. Real plate "
            "photographs need the segmentation model to locate disks; pass the "
            "cropped disks with --image/--dir instead.")
    crops, centers = [], []
    for (x0, y0, x1, y1) in boxes:
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        r = max(x1 - x0, y1 - y0) / 2 * 1.35
        box = (int(cx - r), int(cy - r), int(cx + r), int(cy + r))
        crops.append(im.crop(box))
        centers.append((cx, cy, r))
    tmp = HERE / "_tmp_crops"
    tmp.mkdir(exist_ok=True)
    paths = []
    for i, c in enumerate(crops):
        p = tmp / f"disk_{i}.png"
        c.save(p)
        paths.append(p)
    p = classify(model, paths, device, img_size, tta=tta)
    pred = p.argmax(1)

    out = im.convert("RGB")
    d = ImageDraw.Draw(out)
    for (cx, cy, r), k, prob in zip(centers, pred, p.max(1)):
        d.rectangle([cx - r, cy - r, cx + r, cy + r], outline=(220, 30, 30), width=3)
        d.text((cx - r + 3, cy - r + 3), f"{CLASSES[k]} {prob:.2f}", fill=(220, 30, 30))
    return out, [(CLASSES[k], float(pr)) for k, pr in zip(pred, p.max(1))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=str(HERE / "runs" / "orig" / "best.pt"))
    ap.add_argument("--image", nargs="*", default=[])
    ap.add_argument("--dir", default=None)
    ap.add_argument("--plate", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--img-size", type=int, default=160)
    ap.add_argument("--tta", type=int, default=5)
    ap.add_argument("--blur", type=float, default=0.0)
    ap.add_argument("--json", default=None)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model, metrics = load_model(args.checkpoint, device)
    print(f"loaded {args.checkpoint} on {device}"
          + (f" | reported macro-F1 {metrics['macro_f1']:.4f}" if metrics else ""))

    if args.plate:
        out, res = classify_plate(model, args.plate, device, img_size=args.img_size,
                                  tta=args.tta)
        if args.out:
            out.save(args.out)
            print(f"saved annotated plate -> {args.out}")
        for name, pr in res:
            print(f"  {name:4} ({FULL_NAME[name]:12}) p={pr:.4f}")
        return

    paths = [Path(p) for p in args.image]
    if args.dir:
        d = Path(args.dir)
        paths += sorted(p for p in d.rglob("*")
                        if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    if not paths:
        ap.error("give --image, --dir or --plate")
    probs = classify(model, paths, device, args.img_size, args.blur, args.tta)
    out = []
    for p, pr in zip(paths, probs):
        k = int(pr.argmax())
        out.append(dict(path=str(p), label=CLASSES[k], antibiotic=FULL_NAME[CLASSES[k]],
                        confidence=float(pr[k]),
                        probabilities={c: float(v) for c, v in zip(CLASSES, pr)}))
        print(f"{p.name:44} -> {CLASSES[k]:4} ({FULL_NAME[CLASSES[k]]:12}) p={pr[k]:.4f}")
    if args.json:
        Path(args.json).write_text(json.dumps(out, indent=1), encoding="utf-8")
        print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
