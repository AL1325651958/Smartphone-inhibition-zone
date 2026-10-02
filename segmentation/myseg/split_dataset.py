"""
myseg/split_dataset.py — 把 78 张图按「物理平板」划分成 train/val/test

为什么必须按平板分组
--------------------
文件名形如  <uuid>-<平板号>[_<拍摄序号}>.jpg
  703b896a-002.jpg        → 平板 002，第 1 次拍摄
  49c67615-026_4.jpg      → 平板 026，第 4 次拍摄

026–033 是多角度/多高度的重复拍摄：同一块平板有 2–8 张照片。
若按单张图片随机划分，同一块平板的照片会同时落进 train 和 test，
测试指标会因「模型见过同一块皿」而虚高。因此以平板号为单位整体分配。

划分策略
--------
按平板「整体」分配，任何平板不得跨 split（防止同皿照片同时进 train 与 test）。

平板规模直方图：单张 23 块、4 张 1 块、5 张 1 块、6 张 1 块、8 张 5 块。

关键约束（这也是为什么不能简单按数量切）：
    5 块 8 张平板合计 40 张，**已经超过 train 需要的 38 张**。
    所以 8 张的大平板不可能全部进 train，必须有一部分进 val/test。

求解方式：枚举「每类平板各给哪个 split」，用 0/1 背包保证每个 split 精确命中
目标张数，再在全部可行解里按下面优先级挑最好的：

    1. test 的平板数最多（测试单位多，指标更稳）
    2. val  的平板数最多
    3. 从 train 抽走的张数最少（尽量保住训练量）

并强制 val / test 至少各含 MIN_EVAL_PLATES 块平板，避免出现「只有 4 张图的验证集」。

结果连同平板去向写入 split.json，可完整复现。

用法
----
    python -m myseg.split_dataset            # 执行划分
    python -m myseg.split_dataset --dry-run  # 只看方案，不落盘
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from collections import defaultdict
from pathlib import Path

if __package__ in (None, ""):                     # 允许 python split_dataset.py 直跑
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from myseg import config as C
else:
    from myseg import config as C


# ────────────────────────────────────────────────
#  1. 扫描 + 按平板分组
# ────────────────────────────────────────────────
def plate_of(stem: str) -> str:
    """'49c67615-026_4' -> '026'"""
    rest = stem.split("-", 1)[1] if "-" in stem else stem
    return rest.split("_")[0]


def scan() -> dict[str, list[str]]:
    """返回 {平板号: [stem, ...]}，全部按名称排序。"""
    if not C.SRC_IMAGES.is_dir():
        raise SystemExit(f"找不到图片目录：{C.SRC_IMAGES}")
    if not C.SRC_LABELS.is_dir():
        raise SystemExit(f"找不到标签目录：{C.SRC_LABELS}")

    plates: dict[str, list[str]] = defaultdict(list)
    img_stems = {p.stem for p in C.SRC_IMAGES.iterdir()
                 if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}}
    lbl_stems = {p.stem for p in C.SRC_LABELS.iterdir() if p.suffix.lower() == ".txt"}

    missing = sorted(img_stems - lbl_stems)
    orphan = sorted(lbl_stems - img_stems)
    if missing:
        raise SystemExit(f"以下图片没有标签，先补齐再划分：\n  " + "\n  ".join(missing))
    if orphan:
        raise SystemExit(f"以下标签没有图片，先删除再划分：\n  " + "\n  ".join(orphan))

    for stem in sorted(img_stems):
        plates[plate_of(stem)].append(stem)
    for v in plates.values():
        v.sort()
    return dict(sorted(plates.items()))


# ────────────────────────────────────────────────
#  2. 分配
# ────────────────────────────────────────────────
def _subset_sum(items: list[str], sizes: dict[str, int], target: int) -> list[str] | None:
    """
    从 items 中选出一个子集，使 sizes 之和恰好等于 target。
    按「用了 k 块」分层推进的 0/1 背包，保证每块只用一次；
    在可达的前提下偏好 k 大（平板数多、规模更均衡）。失败返回 None。
    """
    if target == 0:
        return []
    if sum(sizes[i] for i in items) < target:
        return None

    # dp[s] = 达成和 s 时「最后加入的 item 的下标」；-1 表示从 0 起步
    NONE = -1
    START = -2
    dp: list[int] = [NONE] * (target + 1)
    dp[0] = START

    for idx, it in enumerate(items):
        sz = sizes[it]
        if sz > target:
            continue
        for s in range(target - sz, -1, -1):
            if dp[s] == NONE:                          # 该和不可达（含被本块覆盖前）
                continue
            ns = s + sz
            if dp[ns] == NONE:                         # 已有解不覆盖，块数自然递增
                dp[ns] = idx

    if dp[target] == NONE:
        return None

    out, s = [], target
    while dp[s] != START:
        idx = dp[s]
        if idx == NONE:
            return None                                # 回溯断链，视为失败
        out.append(items[idx])
        s -= sizes[items[idx]]
    return sorted(out)


def assign(plates: dict[str, list[str]]) -> dict[str, list[str]]:
    """
    按平板整体分配，精确命中 N_TRAIN/N_VAL/N_TEST，并在可行解中取最优。
    返回 {'train':[...], 'val':[...], 'test':[...], '_plates':{...}}。
    """
    want = {"train": C.N_TRAIN, "val": C.N_VAL, "test": C.N_TEST}
    total = sum(len(v) for v in plates.values())
    if sum(want.values()) != total:
        raise SystemExit(
            f"目标 {want} 合计 {sum(want.values())} 张，但实际有 {total} 张，"
            f"请调整 config 中的划分数量。"
        )

    sizes = {p: len(v) for p, v in plates.items()}
    group_sizes = sorted({sizes[p] for p in plates}, reverse=True)
    groups: dict[int, list[str]] = {s: sorted(p for p in plates if sizes[p] == s)
                                    for s in group_sizes}

    MIN_EVAL_PLATES = 6                     # val/test 至少这么多块平板

    # ── 枚举「每种规模的平板各分多少块给 train/val/test」 ──
    solutions: list[tuple] = []
    alloc: dict[int, dict[str, int]] = {}

    def rec(gi: int, img: dict[str, int], cnt: dict[str, int]) -> None:
        if gi == len(group_sizes):
            if all(img[k] == want[k] for k in want) and \
               cnt["val"] >= MIN_EVAL_PLATES and cnt["test"] >= MIN_EVAL_PLATES:
                # 评分（越小越好）：
                #   test 平板数最多  —— 测试单位多，指标更稳，也避免全押在少数皿上
                #   val  平板数最多  —— 早停/选模更稳
                #   train 平板数最多 —— 训练皿多样性好
                score = (-3 * cnt["test"]) + (-2 * cnt["val"]) + (-1 * cnt["train"])
                solutions.append((score, -cnt["test"], -cnt["val"], -cnt["train"],
                                  {s: dict(alloc[s]) for s in group_sizes}))
            return

        s = group_sizes[gi]
        n_avail = len(groups[s])
        for kt in range(n_avail + 1):
            if img["train"] + s * kt > want["train"]:
                break                               # kt 再大只会更超，剪枝
            for kv in range(n_avail - kt + 1):
                ke = n_avail - kt - kv
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

    if not solutions:
        hist = {s: len(groups[s]) for s in group_sizes}
        big = max(group_sizes) if group_sizes else 0
        raise SystemExit(
            f"在「平板不跨 split」约束下无法精确划分 "
            f"{want['train']}/{want['val']}/{want['test']}。\n"
            f"平板规模直方图：{hist}\n"
            f"提示：最大的平板每块 {big} 张，共 {len(groups.get(big, []))} 块；"
            f"若其合计已超过 train 目标张数，就必须有大平板进 val/test。\n"
            f"请调整 config 中的划分数量。"
        )

    solutions.sort(key=lambda x: x[:4])
    best_alloc = solutions[0][4]

    # ── 按 best_alloc 把具体平板号分配下去（同规模内用固定种子打乱） ──
    rng = random.Random(C.SPLIT_SEED)
    chosen: dict[str, list[str]] = {k: [] for k in want}
    for s in group_sizes:
        pool = groups[s][:]
        rng.shuffle(pool)
        n = best_alloc[s]
        chosen["train"] += pool[:n["train"]]
        chosen["val"] += pool[n["train"]:n["train"] + n["val"]]
        chosen["test"] += pool[n["train"] + n["val"]:n["train"] + n["val"] + n["test"]]

    result = {k: sorted(st for p in sorted(chosen[k]) for st in plates[p])
              for k in ("train", "val", "test")}

    for k, exp in want.items():
        if len(result[k]) != exp:
            raise SystemExit(f"{k} 得到 {len(result[k])} 张，期望 {exp} 张（内部逻辑错误）")

    # ── 防泄露自检 ──
    # 注意：同一个 split 内平板号当然会重复（030 有 8 次拍摄就有 8 个同号 stem），
    #       这是正常的。真正要禁止的是「同一块平板跨 split」。
    seen: dict[str, str] = {}
    for k in ("train", "val", "test"):
        for p in {plate_of(st) for st in result[k]}:        # 该 split 覆盖的平板集合
            if p in seen:
                raise SystemExit(
                    f"数据泄露！平板 {p} 同时出现在 {seen[p]} 和 {k}（分配器内部错误）"
                )
            seen[p] = k

    if set(seen) != set(plates):
        miss = sorted(set(plates) - set(seen))
        raise SystemExit(f"以下平板未被分配到任何 split：{miss}（分配器内部错误）")

    result["_plates"] = {k: sorted({plate_of(st) for st in result[k]}) for k in ("train", "val", "test")}
    return result


# ────────────────────────────────────────────────
#  3. 落盘
# ────────────────────────────────────────────────
def materialize(result: dict, dry: bool) -> None:
    if dry:
        return
    if C.DATA.exists():
        shutil.rmtree(C.DATA)
    for split in ("train", "val", "test"):
        (C.DATA / split / "images").mkdir(parents=True)
        (C.DATA / split / "labels").mkdir(parents=True)

    for split in ("train", "val", "test"):
        for stem in result[split]:
            img_src = next(C.SRC_IMAGES.glob(stem + ".*"))
            shutil.copy2(img_src, C.DATA / split / "images" / img_src.name)
            shutil.copy2(C.SRC_LABELS / f"{stem}.txt",
                         C.DATA / split / "labels" / f"{stem}.txt")

    C.SPLIT_JSON.parent.mkdir(parents=True, exist_ok=True)
    C.SPLIT_JSON.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")


# ────────────────────────────────────────────────
#  4. 报告
# ────────────────────────────────────────────────
def report(plates: dict[str, list[str]], result: dict) -> None:
    print("=" * 72)
    print(f"原图总数 {sum(len(v) for v in plates.values())} 张，"
          f"物理平板 {len(plates)} 块")
    print(f"  多拍摄平板 {sum(1 for v in plates.values() if len(v) > 1)} 块"
          f"（{sum(len(v) for v in plates.values() if len(v) > 1)} 张）")
    print(f"  单张平板   {sum(1 for v in plates.values() if len(v) == 1)} 块")
    print("=" * 72)

    for split in ("train", "val", "test"):
        pl = result["_plates"][split]
        stems = result[split]
        print(f"\n[{split.upper()}]  {len(stems)} 张 / {len(pl)} 块平板")
        by_plate: dict[str, list[str]] = defaultdict(list)
        for s in stems:
            by_plate[plate_of(s)].append(s)
        for p in pl:
            tag = "  ← 多拍摄" if len(by_plate[p]) > 1 else ""
            print(f"   {p:<5} {len(by_plate[p])} 张  {', '.join(by_plate[p])}{tag}")

    print("\n" + "=" * 72)
    print("防泄露自检：通过（无平板跨 split）")
    all_stems = [s for k in ("train", "val", "test") for s in result[k]]
    print(f"覆盖自检：{len(all_stems)} 张，去重后 {len(set(all_stems))} 张"
          f" → {'通过' if len(all_stems) == len(set(all_stems)) else '有重复！'}")
    print("=" * 72)


def main() -> None:
    ap = argparse.ArgumentParser(description="按物理平板划分 train/val/test")
    ap.add_argument("--dry-run", action="store_true", help="只打印方案，不写文件")
    args = ap.parse_args()

    plates = scan()
    result = assign(plates)
    report(plates, result)
    materialize(result, args.dry_run)
    if args.dry_run:
        print("\n[dry-run] 未写入任何文件。")
    else:
        print(f"\n已写入：{C.DATA}")
        print(f"划分清单：{C.SPLIT_JSON}")


if __name__ == "__main__":
    main()
