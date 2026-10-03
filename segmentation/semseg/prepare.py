"""
semseg.prepare — 准备数据：按物理平板划分 train/val/test，并对 train/val 做增强

    python -m semseg.prepare                 # 划分 + 增强
    python -m semseg.prepare --reuse         # 复用已有的 data/（跳过重建）
    python -m semseg.prepare --preview       # 只出一张增强对齐检查图

关键约束：**必须按物理平板整体划分**。
78 张照片来自 31 块物理平板，026–033 是多角度/多高度重复拍摄（单块最多 8 张）。
若按单张随机划分，同一块皿的照片会同时进训练集和测试集，指标虚高。
"""
from __future__ import annotations

import argparse
import random
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from semseg import config as C
    from semseg.dataset import read_polygon_seg, write_polygon_seg
    from semseg.io_utils import imread, imwrite
else:
    from semseg import config as C
    from semseg.dataset import read_polygon_seg, write_polygon_seg
    from semseg.io_utils import imread, imwrite


def plate_of(stem: str) -> str:
    """'49c67615-026_4' -> '026'"""
    rest = stem.split("-", 1)[1] if "-" in stem else stem
    return rest.split("_")[0]


def scan() -> dict[str, list[str]]:
    plates: dict[str, list[str]] = defaultdict(list)
    imgs = {p.stem for p in C.SRC_IMAGES.iterdir()
            if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}}
    lbls = {p.stem for p in C.SRC_LABELS.iterdir() if p.suffix.lower() == ".txt"}
    missing, orphan = sorted(imgs - lbls), sorted(lbls - imgs)
    if missing:
        raise SystemExit("以下图片没有标签：\n  " + "\n  ".join(missing))
    if orphan:
        raise SystemExit("以下标签没有图片：\n  " + "\n  ".join(orphan))
    for s in sorted(imgs):
        plates[plate_of(s)].append(s)
    for v in plates.values():
        v.sort()
    return dict(sorted(plates.items()))


def assign(plates: dict[str, list[str]], strategy: str = "eval-diverse"
           ) -> dict[str, list[str]]:
    """
    按平板整体分配，精确命中 N_TRAIN/N_VAL/N_TEST。

    strategy
    --------
    "eval-diverse"（默认，原始策略）
        val/test 的平板数尽量多 —— 评估面铺得宽。代价是 5 块 8 张平板合计 40 张
        超过 train 需要的 38 张，于是大平板几乎全被挤到 val/test，
        **train 只剩 5 块平板**，模型见的场景太少、几百步就过拟合。

    "balanced"
        让 **min(train 平板数, test 平板数) 最大** —— 训练多样性与测试覆盖面兼顾。
        实测可把 train 从 5 块提到 13 块（图片数不变），test 仍有 13 块独立平板。

    两种策略的图片数完全相同（38 / 20 / 20），只是平板怎么分不同，
    因此可以直接对比「多样性」带来的差异。
    """
    want = {"train": C.N_TRAIN, "val": C.N_VAL, "test": C.N_TEST}
    total = sum(len(v) for v in plates.values())
    if sum(want.values()) != total:
        raise SystemExit(f"目标 {want} 合计 {sum(want.values())} 张，实际 {total} 张")

    sizes = {p: len(v) for p, v in plates.items()}
    group_sizes = sorted({sizes[p] for p in plates}, reverse=True)
    groups = {s: sorted(p for p in plates if sizes[p] == s) for s in group_sizes}

    MIN_EVAL_PLATES = 5 if strategy == "balanced" else 6
    sols: list[tuple] = []
    alloc: dict[int, dict[str, int]] = {}

    def rec(gi, img, cnt):
        if gi == len(group_sizes):
            if not (all(img[k] == want[k] for k in want)
                    and cnt["val"] >= MIN_EVAL_PLATES
                    and cnt["test"] >= MIN_EVAL_PLATES
                    and cnt["train"] >= MIN_EVAL_PLATES):
                return
            if strategy == "balanced":
                # 主目标：抬高 min(train平板, test平板)；再看 val 平板数
                score = (-min(cnt["train"], cnt["test"]), -cnt["val"], -cnt["train"])
            else:
                # 原策略：test 平板数优先，其次 val，其次 train
                score = (-3 * cnt["test"], -2 * cnt["val"], -cnt["train"])
            sols.append((score, {s: dict(alloc[s]) for s in group_sizes}))
            return
        s = group_sizes[gi]
        n = len(groups[s])
        for kt in range(n + 1):
            if img["train"] + s * kt > want["train"]:
                break
            for kv in range(n - kt + 1):
                ke = n - kt - kv
                a = {"train": kt, "val": kv, "test": ke}
                for k, m in a.items():
                    img[k] += s * m
                    cnt[k] += m
                if all(img[k] <= want[k] for k in want):
                    alloc[s] = a
                    rec(gi + 1, img, cnt)
                for k, m in a.items():
                    img[k] -= s * m
                    cnt[k] -= m

    rec(0, {k: 0 for k in want}, {k: 0 for k in want})
    if not sols:
        raise SystemExit("在「平板不跨 split」约束下无法精确划分，请调整 config 中的数量")
    sols.sort(key=lambda x: x[0])
    best = sols[0][1]
    rng = random.Random(C.SPLIT_SEED)
    chosen: dict[str, list[str]] = {k: [] for k in want}
    for s in group_sizes:
        pool = groups[s][:]
        rng.shuffle(pool)
        n = best[s]
        chosen["train"] += pool[:n["train"]]
        chosen["val"] += pool[n["train"]:n["train"] + n["val"]]
        chosen["test"] += pool[n["train"] + n["val"]:n["train"] + n["val"] + n["test"]]

    result = {k: sorted(st for p in sorted(chosen[k]) for st in plates[p])
              for k in ("train", "val", "test")}

    # 防泄露自检：同一块平板只能出现在一个 split
    seen: dict[str, str] = {}
    for k in ("train", "val", "test"):
        for p in {plate_of(st) for st in result[k]}:
            if p in seen:
                raise SystemExit(f"数据泄露！平板 {p} 同时在 {seen[p]} 和 {k}")
            seen[p] = k
    if set(seen) != set(plates):
        raise SystemExit(f"未分配的平板：{sorted(set(plates) - set(seen))}")

    result["_plates"] = {k: sorted({plate_of(st) for st in result[k]})
                         for k in ("train", "val", "test")}
    return result


def materialize(result: dict) -> None:
    if C.DATA.exists():
        shutil.rmtree(C.DATA)
    for split in ("train", "val", "test"):
        (C.DATA / split / "images").mkdir(parents=True)
        (C.DATA / split / "labels").mkdir(parents=True)
        for stem in result[split]:
            src_img = next(C.SRC_IMAGES.glob(stem + ".*"))
            shutil.copy2(src_img, C.DATA / split / "images" / src_img.name)
            shutil.copy2(C.SRC_LABELS / f"{stem}.txt",
                         C.DATA / split / "labels" / f"{stem}.txt")
    C.SPLIT_JSON.write_text(
        __import__("json").dumps(result, indent=2, ensure_ascii=False),
        encoding="utf-8")


# ────────────────────────────────────────────────
#  增强（纯 OpenCV，多边形与像素用同一矩阵变换）
# ────────────────────────────────────────────────
def _augment_once(img, instances, rng):
    h, w = img.shape[:2]
    cls_list = [c for c, _ in instances]
    kpts = [p.astype(np.float32).copy() for _, p in instances]

    if rng.random() < 0.5:
        img = cv2.flip(img, 1)
        for p in kpts:
            p[:, 0] = (w - 1) - p[:, 0]
    if rng.random() < 0.5:
        img = cv2.flip(img, 0)
        for p in kpts:
            p[:, 1] = (h - 1) - p[:, 1]
    if rng.random() < 0.5:
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        for p in kpts:
            x, y = p[:, 0].copy(), p[:, 1].copy()
            p[:, 0] = (h - 1) - y
            p[:, 1] = x
        h, w = img.shape[:2]

    if rng.random() < 0.85:
        m = cv2.getRotationMatrix2D((w / 2.0, h / 2.0),
                                    rng.uniform(-180, 180), rng.uniform(0.9, 1.1))
        m[0, 2] += rng.uniform(-0.05, 0.05) * w
        m[1, 2] += rng.uniform(-0.05, 0.05) * h
        img = cv2.warpAffine(img, m, (w, h), flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_REPLICATE)
        kpts = [cv2.transform(p.reshape(-1, 1, 2), m).reshape(-1, 2) for p in kpts]

    if rng.random() < 0.30:
        s = 0.04 * min(w, h)
        src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
        dst = src + rng.uniform(-s, s, size=(4, 2)).astype(np.float32)
        m = cv2.getPerspectiveTransform(src, dst)
        img = cv2.warpPerspective(img, m, (w, h), flags=cv2.INTER_LINEAR,
                                  borderMode=cv2.BORDER_REPLICATE)
        kpts = [cv2.perspectiveTransform(p.reshape(-1, 1, 2), m).reshape(-1, 2)
                for p in kpts]

    out = img.astype(np.float32)
    if rng.random() < 0.8:
        out = out * (1 + rng.uniform(-0.2, 0.2)) + rng.uniform(-25, 25)
    if rng.random() < 0.5:
        hsv = cv2.cvtColor(np.clip(out, 0, 255).astype(np.uint8), cv2.COLOR_BGR2HSV)
        hsv = hsv.astype(np.int16)
        hsv[..., 0] = (hsv[..., 0] + int(rng.integers(-5, 6))) % 180
        hsv[..., 1] = np.clip(hsv[..., 1] * (1 + rng.uniform(-0.3, 0.3)), 0, 255)
        hsv[..., 2] = np.clip(hsv[..., 2] * (1 + rng.uniform(-0.2, 0.2)), 0, 255)
        out = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR).astype(np.float32)
    if rng.random() < 0.2:
        k = int(rng.choice([3, 5]))
        out = cv2.GaussianBlur(out, (k, k), 0)
    if rng.random() < 0.2:
        out = out + rng.normal(0, rng.uniform(2, 8), out.shape).astype(np.float32)
    img = np.clip(out, 0, 255).astype(np.uint8)

    ha, wa = img.shape[:2]
    res = []
    for cls, p in zip(cls_list, kpts):
        if len(p) < 3:
            continue
        q = np.clip(p, [0.0, 0.0], [float(wa - 1), float(ha - 1)]).astype(np.float32)
        area = float(cv2.contourArea(q.reshape(-1, 1, 2)))
        if area <= 1.0 or area / float(wa * ha) > 0.98:
            return None
        res.append((cls, q))
    return (img, res) if res else None


def augment_split(split: str, per_image: int, seed: int) -> dict:
    src_i = C.DATA / split / "images"
    src_l = C.DATA / split / "labels"
    dst_i = C.DATA / f"{split}_aug" / "images"
    dst_l = C.DATA / f"{split}_aug" / "labels"
    dst_i.mkdir(parents=True, exist_ok=True)
    dst_l.mkdir(parents=True, exist_ok=True)
    for d in (dst_i, dst_l):
        for f in d.iterdir():
            f.unlink()

    rng = np.random.default_rng(seed)
    stats = {"copied": 0, "augmented": 0}
    for stem in sorted(p.stem for p in src_i.glob("*.jpg")):
        img = imread(src_i / f"{stem}.jpg")
        if img is None:
            continue
        h, w = img.shape[:2]
        inst = read_polygon_seg(src_l / f"{stem}.txt", w, h)
        imwrite(dst_i / f"{stem}.jpg", img, quality=95)
        write_polygon_seg(dst_l / f"{stem}.txt", inst, w, h)
        stats["copied"] += 1

        ok, tries = 0, 0
        while ok < per_image and tries < per_image * 8:
            tries += 1
            got = _augment_once(img, inst, rng)
            if got is None:
                continue
            aimg, ainst = got
            ah, aw = aimg.shape[:2]
            name = f"{stem}_aug{ok:03d}"
            imwrite(dst_i / f"{name}.jpg", aimg, quality=95)
            write_polygon_seg(dst_l / f"{name}.txt", ainst, aw, ah)
            ok += 1
            stats["augmented"] += 1
    return stats


def main() -> None:
    ap = argparse.ArgumentParser(description="semseg 数据准备：划分 + 增强")
    ap.add_argument("--reuse", action="store_true", help="复用已有 data/，跳过重建")
    ap.add_argument("--strategy", default="balanced",
                    choices=["eval-diverse", "balanced"],
                    help="划分策略：eval-diverse=测试面铺最宽（原策略，train仅5块平板）；balanced=最大化min(train平板,test平板)，train可到13块")
    ap.add_argument("--preview", action="store_true", help="只出一张增强对齐检查图")
    ap.add_argument("--per-image", type=int, default=C.AUG_PER_IMAGE)
    args = ap.parse_args()

    if args.reuse:
        for s in ("train", "val", "test"):
            n = len(list((C.DATA / s / "images").glob("*.jpg")))
            na = len(list((C.DATA / f"{s}_aug" / "images").glob("*.jpg")))
            print(f"  {s:6s} 原图 {n:4d} 张，增强后目录 {na:4d} 张")
        print("已复用。")
        return

    plates = scan()
    result = assign(plates, args.strategy)
    print("=" * 70)
    print(f"原图 {sum(len(v) for v in plates.values())} 张，物理平板 {len(plates)} 块")
    print("=" * 70)
    for split in ("train", "val", "test"):
        pl = result["_plates"][split]
        print(f"[{split.upper():5s}] {len(result[split]):3d} 张 / {len(pl)} 块平板   "
              f"{', '.join(pl)}")
    print("\n防泄露自检：通过（无平板跨 split）")

    materialize(result)
    print(f"\n已写入：{C.DATA}")

    print(f"\n开始增强（train + val，test 保持原图）…")
    for split in ("train", "val"):
        st = augment_split(split, args.per_image, C.AUG_SEED)
        n = len(list((C.DATA / f"{split}_aug" / "images").glob("*.jpg")))
        print(f"  {split}: 原图 {st['copied']} + 增强 {st['augmented']} = {n} 张")
    print(f"  test: {len(list((C.DATA / 'test' / 'images').glob('*.jpg')))} 张（不增强）")

    if args.preview:
        _preview()


def _preview() -> None:
    src_i = C.DATA / "train" / "images"
    src_l = C.DATA / "train" / "labels"
    stems = sorted(p.stem for p in src_i.glob("*.jpg"))
    rng = np.random.default_rng(C.AUG_SEED)
    tiles = []
    for k in range(4):
        stem = stems[int(rng.integers(len(stems)))]
        img = imread(src_i / f"{stem}.jpg")
        h, w = img.shape[:2]
        inst = read_polygon_seg(src_l / f"{stem}.txt", w, h)
        if k > 0:
            got = None
            for _ in range(30):
                got = _augment_once(img, inst, rng)
                if got is not None:
                    break
            if got is None:
                continue
            img, inst = got
        vis = img.copy()
        for c, pts in inst:
            col = (117, 158, 29) if c == C.ZONE else (165, 95, 24)
            cv2.polylines(vis, [np.round(pts).astype(np.int32)], True, col, 4)
        s = 520 / vis.shape[0]
        vis = cv2.resize(vis, (int(vis.shape[1] * s), 520))
        cv2.putText(vis, "original" if k == 0 else f"aug{k}", (10, 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
        tiles.append(vis)
    if len(tiles) >= 4:
        grid = np.vstack([np.hstack(tiles[:2]), np.hstack(tiles[2:4])])
        out = C.RUNS / "augment_preview.jpg"
        out.parent.mkdir(parents=True, exist_ok=True)
        imwrite(out, grid, quality=92)
        print(f"\n增强对齐检查图：{out}")


if __name__ == "__main__":
    main()
