"""Set up a Label Studio project for annotating the 621 expanded disk crops.

Creates:
  labelstudio/files/<plate>/<crop>.png   images served from Label Studio
  labelstudio/config.xml                 labelling interface (7 drug classes)
  labelstudio/tasks.json                 one task per crop, pre-sorted by plate
  labelstudio/start.cmd / start.ps1      launcher

Run:  python setup_labelstudio.py
Then: labelstudio\\start.cmd   and open http://127.0.0.1:8081
"""
from __future__ import annotations

import csv
import json
import shutil
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "expanded"
LS = HERE / "labelstudio"
FILES = LS / "files"
CLASSES = ["CRO", "DA", "E", "LEV", "LZD", "P", "VA"]
FULL = {"CRO": "Ceftriaxone", "DA": "Clindamycin", "E": "Erythromycin",
        "LEV": "Levofloxacin", "LZD": "Linezolid", "P": "Penicillin",
        "VA": "Vancomycin"}

rows = list(csv.DictReader(open(OUT / "expanded_detections.csv", encoding="utf-8")))
by_plate = defaultdict(list)
for r in rows:
    by_plate[r["plate"]].append(r)
for p in by_plate:
    by_plate[p].sort(key=lambda r: float(r["angle"]))

# ---------------------------------------------------------------- copy images
FILES.mkdir(parents=True, exist_ok=True)
copied = 0
for p, items in by_plate.items():
    d = FILES / p
    d.mkdir(parents=True, exist_ok=True)
    for r in items:
        src = OUT / "crops" / r["crop"]
        if src.exists():
            shutil.copy2(src, d / r["crop"])
            copied += 1
print(f"copied {copied} crops into {FILES}")

# ---------------------------------------------------------------- config
choices = "\n".join(
    f'    <Choice value="{c}" alias="{c}" background="#{"2166ac" if i % 2 == 0 else "92c5de"}"/>'
    f'  <!-- {FULL[c]} -->' for i, c in enumerate(CLASSES))

config = f"""<View>
  <Header value="抗生素纸片识别标注  -  请选择该磁盘上的药物缩写"/>
  <View style="display:flex;flex-direction:row;gap:20px">
    <View style="width:320px;flex-shrink:0">
      <Image name="disk" value="$image" zoom="true" maxWidth="320px"/>
    </View>
    <View style="flex:1">
      <View style="background:#f5f5f5;padding:8px;border-radius:4px;margin-bottom:10px">
        <Text name="meta_plate" value="皿号: $plate"/>
        <Text name="meta_angle" value="角度: $angle 度"/>
        <Text name="meta_ndisks" value="该皿磁盘数: $n_disks"/>
        <Text name="meta_model" value="模型预测: $model_pred (置信度 $model_conf)"/>
      </View>
      <Choices name="drug" toName="disk" choice="single" showInline="false">
{choices}
      </Choices>
      <View style="margin-top:8px">
        <Text name="hint" value="若字迹无法辨认，请选择 unclear 而不是猜测。"/>
        <Choices name="quality" toName="disk" choice="single" showInline="true">
          <Choice value="legible" alias="legible"/>
          <Choice value="unclear" alias="unclear"/>
        </Choices>
      </View>
    </View>
  </View>
</View>
"""
(LS / "config.xml").write_text(config, encoding="utf-8")
print(f"wrote {LS/'config.xml'}")

# ---------------------------------------------------------------- tasks
# model predictions from the earlier pass (kept only as a hint for the annotator)
pred = {}
pred_file = OUT / "labelled_crops.csv"
if pred_file.exists():
    for r in csv.DictReader(open(pred_file, encoding="utf-8")):
        pred[r["crop"]] = (r["model_pred"], r["model_conf"])

tasks = []
for p in sorted(by_plate):
    for i, r in enumerate(by_plate[p]):
        crop = r["crop"]
        mp, mc = pred.get(crop, ("", ""))
        rel = f"{p}/{crop}"
        tasks.append({
            "data": {
                "image": f"/data/local-files/?d={rel}",
                "crop": crop,
                "plate": p,
                "angle": r["angle"],
                "n_disks": int(r["n_disks"]),
                "gap_next": r["gap_next"],
                "model_pred": mp,
                "model_conf": mc,
                "order_in_plate": i + 1,
            },
            "meta": {"plate": p, "crop": crop},
        })

(LS / "tasks.json").write_text(json.dumps(tasks, ensure_ascii=False, indent=1),
                               encoding="utf-8")
print(f"wrote {LS/'tasks.json'}  ({len(tasks)} tasks)")

# ---------------------------------------------------------------- launcher
data_dir = LS / "data"
data_dir.mkdir(exist_ok=True)
py = Path(r"F:\Code\SCI\Nanjing_Children_Hospital\抑菌区域分析_返修\_tools\ls-venv\Scripts\label-studio.exe")
relpy = r"..\..\_tools\ls-venv\Scripts\label-studio.exe"

cmd = f"""@echo off
REM Launch Label Studio for the antibiotic-disk annotation project.
REM Images are served from labelstudio\\files via LABEL_STUDIO_LOCAL_FILES_*.
setlocal
set "HERE=%~dp0"
set "LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true"
set "LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT=%HERE%"
set "LABEL_STUDIO_DISABLE_SIGNUP_WITHOUT_LINK=true"
set "LABEL_STUDIO_USERNAME=admin@example.com"
set "LABEL_STUDIO_PASSWORD=labelstudio"
set "LABEL_STUDIO_ANALYTICS=false"
set "DATA_UPLOAD_MAX_NUMBER_FILES=2000"
set "LABEL_STUDIO_BASE_DATA_DIR=%HERE%data"
echo Starting Label Studio on http://127.0.0.1:8081
echo   login: admin@example.com / labelstudio
"%HERE%{relpy}" start --no-browser --host 127.0.0.1 --port 8081 --data-dir "%HERE%data"
endlocal
"""
(LS / "start.cmd").write_text(cmd, encoding="ascii", errors="replace")
print(f"wrote {LS/'start.cmd'}")

print(f"""
next steps
  1. run:  {LS / 'start.cmd'}
  2. open: http://127.0.0.1:8081     (login admin@example.com / labelstudio)
  3. Create Project -> 导入 {LS / 'tasks.json'}
  4. Labeling interface -> 粘贴 {LS / 'config.xml'} 的内容
  5. 给每个 task 标注药物缩写

crops: {copied}   plates: {len(by_plate)}   tasks: {len(tasks)}
""")
