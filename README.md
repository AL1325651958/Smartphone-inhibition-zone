# Smartphone-assisted measurement of streptococcal disk-diffusion inhibition zones

Code, trained models and annotated data for the paper

> **Smartphone-Assisted Measurement of Streptococcal Disk Diffusion Inhibition Zones
> Using Convolutional Neural Networks and Internal Disk Calibration**
> (Children's Hospital of Nanjing Medical University, Department of Clinical Laboratory)

The pipeline takes a single smartphone photograph of a Kirby–Bauer plate and returns, for every
antibiotic disk, the inhibition-zone diameter in millimetres — without any ruler or reference
object other than the 6 mm disk itself (internal calibration).

---

## What is in this repository

| Component | Directory | What it does |
|---|---|---|
| **Segmentation** | `segmentation/` | Two-channel semantic segmentation of the inhibition zone (`Area`) and the antibiotic disk (`Yaoping`) from a plate photograph. ResNet18 + U-Net, native PyTorch, 640×640. |
| **Instance derivation** | `segmentation/semseg/` | Disk = 8-connected component; zone = **geodesic partition seeded by disk centroids** (this is how merged neighbouring zones are separated). |
| **Measurement (paper definition)** | `segmentation/myseg/` | The ±15° radial-sector sampling along the plate-centre→disk direction, with `mm_per_px = 6.0 / median(disk equivalent diameter)`. |
| **Disk classifier** | `classification/` | Seven-class CNN (**DiskNet-small**, 3.89 M parameters, 224×224) that reads the antibiotic abbreviation printed on the disk: CRO / DA / E / LEV / LZD / P / VA. |
| **Mobile app** | `app/` | .NET MAUI application (Android / iOS / Windows) that runs both ONNX models on-device and exports the annotated result. |
| **Weights** | `weights/` (`.pt`) and `app/Antibacterial zone/Resources/Raw/` (`.onnx`) | Deployed models. |
| **Data** | `segmentation/images/`, `segmentation/labels/` | 78 plate photographs (31 physical plates) and their YOLO-seg annotations. |

---

## Results

**Segmentation** (20 held-out photographs / 13 physical plates never seen in training, TTA on):

| Channel | Dice | IoU | Precision | Recall |
|---|---|---|---|---|
| Inhibition zone (`Area`) | **0.9561** | 0.9159 | 0.9597 | 0.9525 |
| Antibiotic disk (`Yaoping`) | **0.9257** | 0.8616 | 0.9330 | 0.9185 |
| macro | **0.9409** | | | |

**Instance level in the app** (same 20 photographs, compared against the annotations, IoU ≥ 0.5):

| Class | predicted | ground truth | Precision | Recall | F1 | matched IoU |
|---|---|---|---|---|---|---|
| `Area` | 136 | 110 | 0.802 | 0.991 | 0.886 | 0.757 |
| `Yaoping` | 141 | 143 | 0.993 | 0.979 | 0.986 | 0.858 |

**Disk classification** (batch-disjoint split: 28 test crops from five acquisition batches that
contributed no images to training or validation): single model **0.857**, with 5-view test-time
augmentation **0.929**; the reported five-seed ensemble reaches 0.929 with macro-F1 0.927.
Blurring the printed code drops the earlier baseline to 0.172, i.e. the model reads the printed
label rather than the agar texture.

Verification scripts (and the harness that reproduces every number quoted here) are in `tools/verify/`.

---

## Repository layout

```
.
├── segmentation/            # plate segmentation (+ measurement implementation)
│   ├── images/              # 78 plate photographs (JPEG, 4032×3024 mostly)
│   ├── labels/              # 78 YOLO-seg label files (class 0 = Area, 1 = Yaoping)
│   ├── semseg/              # ResNet18+U-Net: prepare / train / infer / export_onnx
│   │   └── split.json       # train/val/test split, grouped by physical plate
│   ├── myseg/               # first-generation implementation; contains the ±15° sector
│   │                        # measurement and the 6 mm internal calibration
│   └── requirements.txt
├── classification/          # DiskNet / PillCNN training, evaluation and reporting scripts
│   └── manifests/           # dataset index (class + split per image; images on request)
├── app/                     # .NET MAUI application (source + the two ONNX models)
│   ├── Antibacterial zone/  # ZoneSegModel.cs, PillClassifier.cs, ImageProcessor.cs, ...
│   └── ...                  # see app/README.md and app/model_doc.txt
├── weights/                 # semseg_resnet18_best.pt, disknet224_small_best.pt
└── tools/                   # verification harness (C# console + Python comparison scripts)
```

---

## Model interfaces

```
best.onnx   (segmentation, ResNet18+U-Net, 14.4 M params, opset 17)
  input   input : float32 [1,3,640,640]   RGB, 0..255  (ImageNet normalisation baked into the graph)
  output  probs : float32 [1,2,640,640]   sigmoid probabilities; 0 = inhibition zone, 1 = disk
  preprocess : resize the whole image to 640×640 (no letterbox, no grey padding)
  postprocess: threshold 0.45 → disks = 8-connected components (≥30 px);
               zones  = multi-source BFS geodesic partition seeded by the disk centroids

class.onnx  (disk classifier, DiskNet-small, 3.89 M params, opset 17)
  input   image  : float32 [N,3,224,224]  greyscale replicated to 3 channels, 0..255
                                          ((x/255-0.5)/0.25 baked into the graph)
  output  logits : float32 [N,7]          raw logits (average over TTA views, then softmax)
          probs  : float32 [N,7]          softmax probabilities
  classes : CRO, DA, E, LEV, LZD, P, VA
  preprocess : grey (BT.601) → PIL-style bicubic resize to 224
               → 5 TTA views (identity, horizontal flip, vertical flip, ±10° with fill 128)
               → average the logits of the 5 views, then softmax
               (no CLAHE/sharpen/gamma: the DiskNet dataset holds raw greyscale crops)
```

Both models are exported with `export_onnx.py` scripts that verify PyTorch ↔ ONNX-Runtime
agreement automatically (segmentation: max |Δp| 1.4e-7, binarised IoU 1.000;
classification: 0 top-1 disagreements over 18 random inputs).

---

## Reproducing

### Segmentation

```bash
cd segmentation
pip install -r requirements.txt          # torch, torchvision, opencv-python, onnx, onnxruntime
python -m semseg.prepare                 # split by physical plate + augment (writes data/)
python -m semseg.train --encoder resnet18 --epochs 80 --batch 4 --lr 3e-4 --workers 3
python -m semseg.export_onnx --weights runs_semseg/<run>/best.pt --out best.onnx
```

The split is grouped by **physical plate** (all photographs of one plate stay on one side);
`semseg/prepare.py` aborts if a plate appears in two splits.

### Classification

```bash
cd classification
python train_disknet.py --variant small --img-size 224   # the shipped classifier
python ensemble_eval.py --runs disknet224_small --tta 5  # 28-crop held-out split (0.929)
python export_onnx_disknet.py                            # writes class.onnx
```

### App

```bash
cd "app/Antibacterial zone"
dotnet build -t:Run -f net10.0-android
dotnet build -t:Run -f net10.0-windows10.0.19041.0
```

### Verification harness (no MAUI workload required)

`tools/verify/` compiles the app's `ZoneSegModel.cs` / `PillClassifier.cs` / `ImageProcessor.cs`
into a small console project and runs them on batches of images:

```bash
dotnet build "tools/verify/Verify.csproj" -c Release
Verify.exe seg weights/../app/.../best.onnx segmentation/images out_seg
python tools/verify/compare_seg.py out_seg
```

---

## Data availability

* **Shipped here**: the 78 plate photographs (`segmentation/images/`), their YOLO-seg annotations
  (`segmentation/labels/`), the plate-grouped split (`segmentation/semseg/split.json`) and the
  classifier dataset index (`classification/manifests/*.csv`, which carries the class label and
  split of every crop).
* **Not shipped**: the classifier image dataset (9,062 enhanced/augmented disk crops, 60 source
  disks) and the app's raw disk crops, because they were derived from routine clinical laboratory
  work. They are available from the corresponding authors on reasonable request and with
  institutional approval.
* The `path` column of the manifests is an absolute path from the machine that produced them;
  regenerate the manifests with `build_manifest.py` after obtaining the crops.

## Known issues / caveats

1. **EXIF orientation.** `SKBitmap.Decode` (used by the app) does **not** apply EXIF orientation,
   while OpenCV/PIL/browsers do. The app therefore works in the camera's stored pixel frame.
   Every reported quantity (equivalent diameter, inhibition distance, ratios) is rotation
   invariant, so measurements are unaffected — but when comparing app output with the annotations
   you must rotate one of them first (see `tools/verify/compare_seg.py`), otherwise the masks look
   completely disjoint.
2. **Classifier metric.** Quote the batch-disjoint result of the shipped model
   (`weights/disknet224_small_best.pt`): 0.857 without test-time augmentation, **0.929** with
   5-view TTA, on 28 test crops from five held-out acquisition batches. The earlier PillCNN
   checkpoint (`runs/orig/best.pt`) reported `accuracy = 1.000` on the shipped split, where
   augmented copies of the same 60 disks appear on both sides — that number is **not** a
   generalisation estimate.
3. **App build.** Building the MAUI app needs the .NET 10 SDK with the MAUI workload and NuGet
   access. The verification harness under `tools/verify/` reproduces all inference-side numbers
   without them.
4. **Ablation branch.** `segmentation/netmask` (a CenterMask-style instance-segmentation attempt)
   is intentionally not included: it never converged (`diskF1 = 0` throughout).

## Citation

If you use this code or these models, please cite the paper (reference to be added upon
publication). Model checkpoints were produced from:

* `runs_semseg/20260918_051406_resnet18/best.pt` — ResNet18+U-Net, macro Dice 0.9409
* `runs/disknet224_small/best.pt` — DiskNet-small @224, 0.929 with 5-view TTA

## License

Code: MIT (see `LICENSE`). Data and model weights: CC BY-NC 4.0 — non-commercial research use,
please cite the paper.

---

## 中文概要

本仓库是上述论文的代码与模型：`segmentation/` 是培养皿抑菌圈/药片的语义分割（ResNet18+U-Net，
纯 PyTorch，640×640，宏平均 Dice 0.9409），`segmentation/myseg/` 里保留了论文口径的测量实现
（±15° 扇区采样 + 6 mm 药片内标），`classification/` 是药片缩写七分类（DiskNet-small@224，单模型+TTA 准确率 0.929），
`app/` 是 .NET MAUI 端（内含两个 ONNX）。78 张平板原图与 YOLO-seg 标注随仓库提供；
分类数据集因来源限制需向通讯作者申请；推理侧的复现脚本见 `tools/verify/`。
