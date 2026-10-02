"""Plain accuracy report + error inventory (no protocol jargon)."""
import csv
from collections import Counter
from pathlib import Path

HERE = Path(r"F:\Code\SCI\Nanjing_Children_Hospital\抑菌区域分析_返修\药片CNN")
CLASSES = ["CRO", "DA", "E", "LEV", "LZD", "P", "VA"]
rows = list(csv.DictReader(open(HERE / "error_report" / "predictions_lopo.csv",
                                encoding="utf-8")))
allr = list(csv.DictReader(open(HERE / "manifest_plate.csv", encoding="utf-8")))

n_all = len(allr)
n_wrong = len(rows)
acc_all = 1 - n_wrong / n_all

BAD = "20260505_063258"
n_bad = sum(1 for r in allr if r["plate"] == BAD)
n_good = n_all - n_bad
wrong_bad = sum(1 for r in rows if r["plate"] == BAD)
wrong_good = n_wrong - wrong_bad

print("=" * 66)
print("抗生素缩写识别 —— 准确率汇总")
print("=" * 66)
print(f"评估方式：留一批次交叉验证，每张图只由没见过其批次的模型预测一次")
print(f"总图像数 {n_all}   错误 {n_wrong}   正确 {n_all-n_wrong}")
print()
print(f"  全体 13 个批次           准确率 {acc_all:.4f}  ({n_all-n_wrong}/{n_all})")
print(f"  排除 063258 后的 12 个批次 准确率 {1-wrong_good/n_good:.4f}  "
      f"({n_good-wrong_good}/{n_good})")
print()
print(f"  批次 063258 单独         准确率 {1-wrong_bad/n_bad:.4f}  "
      f"({n_bad-wrong_bad}/{n_bad})  ← 占全部错误的 "
      f"{wrong_bad/max(n_wrong,1):.0%}")

print("\n各批次明细：")
print(f"{'批次':22} {'图数':>5} {'错误':>5} {'准确率':>8}")
per = Counter(r["plate"] for r in rows)
for p in sorted({r["plate"] for r in allr},
                key=lambda p: -per.get(p, 0) / max(sum(1 for r in allr if r["plate"] == p), 1)):
    tot = sum(1 for r in allr if r["plate"] == p)
    w = per.get(p, 0)
    print(f"{p:22} {tot:>5} {w:>5} {1-w/tot:>8.4f}")

print("\n最常混淆的组合（真实 -> 预测）：")
pairs = Counter((r["class"], r["predicted"]) for r in rows)
tot_cls = Counter(r["class"] for r in allr)
for (a, b), n in pairs.most_common(10):
    print(f"  {a:4} -> {b:4} {n:5d}  占该类 {n/tot_cls[a]:5.1%}")

print("\n各类召回率：")
print(f"{'类别':6} {'总数':>5} {'错误':>5} {'召回':>7}")
err_cls = Counter(r["class"] for r in rows)
for c in CLASSES:
    t = tot_cls[c]
    e = err_cls.get(c, 0)
    print(f"{c:6} {t:>5} {e:>5} {1-e/t:>7.3f}")

print("\n" + "=" * 66)
print("错误图片清单")
print("=" * 66)
out = HERE / "error_report"
for f in sorted(out.glob("*.png")):
    print(f"  {f.name}")
print(f"\n逐张预测明细：{out/'predictions_lopo.csv'}")
