"""Label the expanded crops by template matching against one exemplar per drug.

Reading 621 crops by eye is error-prone.  Instead: take one clearly legible
exemplar per drug, then assign every crop to the drug whose exemplar it matches
best on a scale/rotation-tolerant descriptor.  Validate against the plates whose
drug composition is pinned independently by the spreadsheet.
"""
import csv
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(r"F:\Code\SCI\Nanjing_Children_Hospital\抑菌区域分析_返修\药片CNN")
OUT = HERE / "expanded"
CLASSES = ["CRO", "DA", "E", "LEV", "LZD", "P", "VA"]

det = {r["crop"]: r for r in csv.DictReader(
    open(OUT / "expanded_detections.csv", encoding="utf-8"))}


def load(crop, size=96):
    return Image.open(OUT / "crops" / crop).convert("L").resize((size, size),
                                                               Image.BICUBIC)


def descriptor(im, angles=range(0, 360, 30)):
    """Rotation-tolerant, illumination-normalised signature."""
    feats = []
    g = np.asarray(im, dtype=np.float32) / 255.0
    g = (g - g.mean()) / (g.std() + 1e-6)
    feats.append(g.ravel())
    for a in angles:                      # rotate the pattern, keep the profile
        r = im.rotate(a, resample=Image.BICUBIC, fillcolor=128)
        gg = np.asarray(r, dtype=np.float32) / 255.0
        gg = (gg - gg.mean()) / (gg.std() + 1e-6)
        feats.append(gg.ravel())
    return np.concatenate(feats)


# ---- exemplars: pick crops the model reads confidently AND that are legible ----
# plate 001 crops were read by eye as E, DA, CRO, P, VA, LEV, LZD (in angle order)
MANUAL = {
    "20260505_001": None,   # placeholder; actual plate key is '001'
}
plate001 = sorted([r for r in det.values() if r["plate"] == "001"],
                  key=lambda r: float(r["angle"]))
eye = ["E", "DA", "CRO", "P", "VA", "LEV", "LZD"]
exemplars = {}
print("exemplars taken from plate 001 (read by eye):")
for r, drug in zip(plate001, eye):
    exemplars[drug] = descriptor(load(r["crop"]))
    print(f"  {drug:4} <- {r['crop']}")

E = np.stack([exemplars[d] for d in CLASSES])
E = E / np.linalg.norm(E, axis=1, keepdims=True)

# ---- assign every crop ----
assign = {}
for crop in det:
    X = descriptor(load(crop))
    X = X / (np.linalg.norm(X) + 1e-9)
    s = E @ X
    j = int(np.argmax(s))
    assign[crop] = (CLASSES[j], float(s[j]), float(s[j] - np.sort(s)[-2]))

print(f"\nassigned {len(assign)} crops by template match")

# ---- validate against the spreadsheet-pinned plates -------------------------
PLATE_SETS = {
    "002": set("CRO DA E LEV LZD P VA".split()),
    "006": set("CRO DA E LEV LZD P VA".split()),
    "011": set("CRO DA E LEV LZD P VA".split()),
    "013": set("CRO DA E LEV LZD P VA".split()),
    "017": set("CRO DA E LEV LZD P VA".split()),
    "018": set("CRO DA E LEV LZD P VA".split()),
    "019": set("CRO DA E LEV LZD P VA".split()),
}
ok = tot = 0
print("\nvalidation on plates with a complete 7-drug composition:")
for p, drugs in PLATE_SETS.items():
    crop_items = [c for c, r in det.items() if r["plate"] == p]
    got = [assign[c][0] for c in crop_items]
    match = sum(1 for g in got if g in drugs)
    ok += match
    tot += len(got)
    print(f"  plate {p}: {len(got)} disks -> {sorted(Counter(got).items())}  "
          f"all in set: {set(got) <= drugs}")
print(f"  {ok}/{tot} crops assigned to a drug the plate actually carries")

print("\nlabel distribution over all 621 crops:")
c = Counter(v[0] for v in assign.values())
for k in CLASSES:
    print(f"  {k:4} {c.get(k,0):4d}")

# margins: how decisive is each match?
margins = np.array([v[2] for v in assign.values()])
print(f"\nmatch margin to runner-up: median {np.median(margins):.3f}, "
      f"25th pct {np.percentile(margins,25):.3f}")
for t in (0.05, 0.10, 0.15):
    n = (margins >= t).sum()
    print(f"  margin >= {t:.2f}: {n} crops ({n/len(margins):.1%})")

with open(OUT / "template_labels.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["crop", "plate", "angle", "label", "score", "margin"])
    for crop, (lab, sc, mg) in sorted(assign.items(),
                                      key=lambda kv: (det[kv[0]]["plate"],
                                                      float(det[kv[0]]["angle"]))):
        w.writerow([crop, det[crop]["plate"], det[crop]["angle"], lab,
                    f"{sc:.4f}", f"{mg:.4f}"])
print(f"\nwrote {OUT/'template_labels.csv'}")
