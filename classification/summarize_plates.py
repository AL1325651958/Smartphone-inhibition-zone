"""Summarise plate (batch) composition of the 60 clean base disks."""
import json
from collections import Counter, defaultdict
from pathlib import Path

OUT = Path(r"F:\Code\SCI\Nanjing_Children_Hospital\抑菌区域分析_返修\药片CNN\reports")
d = json.load(open(OUT / "base_to_plate.json", encoding="utf-8"))

by_plate = defaultdict(list)
for core, v in d.items():
    by_plate[v["plate"]].append((v["pill"], v["cls"], core, v["corr"], v["second"]))

CLASSES = ["CRO", "DA", "E", "LEV", "LZD", "P", "VA"]
print(f"{len(d)} base disks recovered across {len(by_plate)} plates\n")
print(f"{'plate':22} {'n':>3}  class counts (CRO DA E LEV LZD P VA)")
for plate in sorted(by_plate):
    items = by_plate[plate]
    cc = Counter(c for _, c, *_ in items)
    row = " ".join(f"{cc.get(k, 0):>3}" for k in CLASSES)
    print(f"{plate:22} {len(items):>3}  {row}")

print("\nplate -> per class, with match strength (corr = similarity to raw crop)")
tot = Counter()
for plate in sorted(by_plate):
    for pill, cls, core, corr, second in sorted(by_plate[plate]):
        tot[cls] += 1
print("\nbase disks per class:", dict(tot))

# which plates supply which classes
print("\nclasses contributed by each plate:")
for plate in sorted(by_plate):
    cs = sorted({c for _, c, *_ in by_plate[plate]})
    print(f"  {plate:22} {cs}")

# plate purity: does a single plate cover all 7 classes?
full = [p for p in by_plate if len({c for _, c, *_ in by_plate[p]}) == 7]
print(f"\nplates covering all 7 classes: {len(full)} -> {sorted(full)}")

# how many classes would remain if a plate were held out entirely?
print("\nif each plate alone were the test set, how many classes appear in it?")
for plate in sorted(by_plate, key=lambda p: -len({c for _, c, *_ in by_plate[p]})):
    cs = {c for _, c, *_ in by_plate[plate]}
    print(f"  {plate:22} {len(cs)}/7 classes, n={len(by_plate[plate])}")

# per-class availability: can we hold out disks and still train each class?
print("\nper-class base-disk counts (for grouped hold-out sizing):")
for c in CLASSES:
    n = sum(1 for v in d.values() if v["cls"] == c)
    plates = sorted({v["plate"] for v in d.values() if v["cls"] == c})
    print(f"  {c:4} disks {n}  from plates {plates}")
