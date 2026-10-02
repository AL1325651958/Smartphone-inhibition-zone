# Antibacterial Zone Analyzer

基于 .NET MAUI + 自研 CNN 的抑菌圈自动分析移动应用。  
拍摄或上传培养皿图片，自动分割抑菌区域与药片、配对、计算抑菌距离与比值，并对药片进行分类识别。

> **模型已替换为本项目自研模型**（不再使用 YOLOv8）：
> * `best.onnx` — `Antibacterial zone mask/semseg` 的 ResNet18 + U-Net 双通道语义分割（640×640）
> * `class.onnx` — `药片CNN` 的 **DiskNet-small** 七类分类器（224×224，5 视图 TTA）
>
> 两者的导出脚本、接口约定、预处理对齐说明见本文件「模型说明」与「预处理流程」两节，
> 详细技术文档见 `model_doc.txt`。

---

## 目录

- [功能概览](#功能概览)
- [技术架构](#技术架构)
- [项目结构](#项目结构)
- [推理流程](#推理流程)
- [界面说明](#界面说明)
- [测量指标说明](#测量指标说明)
- [模型说明](#模型说明)
- [预处理流程](#预处理流程)
- [Python 训练工具](#python-训练工具)
- [环境要求与依赖](#环境要求与依赖)
- [部署步骤](#部署步骤)
- [常见问题](#常见问题)

---

## 功能概览

| 功能 | 说明 |
|---|---|
| 图片输入 | 支持从相册上传或直接调用摄像头拍摄 |
| 目标分割 | 自研 ResNet18+U-Net 双通道语义分割（`Area` 抑菌圈 / `Yaoping` 药片），掩膜 640×640 |
| 实例化 | 药片 = 8 连通域；抑菌圈 = 以药片质心为种子的测地分区（相邻融合的圈在此切开） |
| 配对匹配 | 直接采用分区归属（每个圈实例天然知道自己属于哪块药片） |
| 几何测量 | 计算等效直径、最小外接圆、椭圆拟合、中心距、抑菌距离、抑菌比值 |
| 药片分类 | 自研 **DiskNet-small** 对药片进行 7 类抗生素识别，输出类别与置信度（224×224，5 视图 TTA） |
| 结果可视化 | 在原图上叠加彩色分割轮廓、测量标注；左侧面板显示药片特写与分类结果 |
| 结果导出 | Export：原图 + 所有裁剪子图垂直拼接为长图，通过系统分享保存 |
| 原图导出 | Save Raw Crops：将药片原始裁剪图打包为 ZIP，用于调试与重新训练 |

---

## 技术架构

```
┌─────────────────────────────────────────────────────┐
│                    .NET MAUI App                     │
│                                                      │
│  MainPage.xaml / MainPage.xaml.cs                    │
│    ├── 图片输入（FilePicker / MediaPicker）           │
│    ├── 异步推理流程（Task.Run + IProgress）           │
│    └── 结果显示与导出（Share API）                   │
│                                                      │
│  ZoneSegModel.cs            PillClassifier.cs        │
│    ├── 整图 resize 640×640      ├── BT.601 灰度化      │
│    ├── ONNX 推理（best.onnx）  ├── PIL 式 BICUBIC 224  │
│    ├── 5 视图 TTA              ├── 5 视图 TTA          │
│    ├── 阈值 0.45 二值化        │   (hflip/vflip/±10°)  │
│    ├── 连通域 + 测地分区实例化  ├── logits 平均 → softmax│
│    └── 轮廓提取（原图坐标）     └── ONNX 推理(class.onnx)│
│                                                      │
│  ImageProcessor.cs                                   │
│    ├── 区域配对（最近邻）                             │
│    ├── Ritter 最小外接圆                              │
│    ├── 椭圆拟合                                       │
│    ├── 测量计算与标注绘制                             │
│    └── 左侧面板合成                                   │
└─────────────────────────────────────────────────────┘

运行时依赖：
  Microsoft.ML.OnnxRuntime  — ONNX 模型推理
  SkiaSharp                 — 图像处理与绘制
  .NET MAUI                 — 跨平台 UI 框架
```

---

## 项目结构

```
Antibacterial_zone/
├── MainPage.xaml              # UI 布局（按钮、图片显示、进度条）
├── MainPage.xaml.cs           # 主页逻辑（输入、推理流程、导出）
├── ZoneSegModel.cs            # 自研语义分割模型封装（+ 实例化与轮廓/测量）
├── PillClassifier.cs          # 自研七类分类模型封装（+ 增强/TTA 预处理）
├── ImageProcessor.cs          # 区域配对、测量计算、可视化合成
│
├── Resources/
│   └── Raw/
│       ├── best.onnx          # 分割模型（semseg：ResNet18+U-Net，2 通道）
│       └── class.onnx         # 分类模型（DiskNet-small，7 类，224×224）
│
└── ../_verify/                # 本机核验工程（不随 App 发布，见「模型说明」末节）
    ├── Verify.csproj          # 轻量控制台工程：直接编译上面两个 .cs 做数值核验
    ├── Program.cs             # seg / cls 两个子命令，导出 JSON + 掩膜 PNG
    └── compare_seg.py         # 与真值标注对比，算实例级/语义级指标
```

---

## 推理流程

### 完整分析流程（5步）

```
用户输入图片
    │
    ▼
Step 1  PreprocessImage()          主线程
        整图双线性 resize 到 640×640（**不做 letterbox、不补灰边**，
        与 semseg 训练时的 dataset.py 一致），NRGB 0..255
    │
    ▼
Step 2  Predict()                  后台线程
        ONNX 推理 best.onnx → [1,2,640,640] sigmoid 概率
        （通道 0 = Area 抑菌圈，通道 1 = Yaoping 药片）
        开启 5 视图 TTA（0/90/180/270 + 水平翻转）取平均
        → 阈值 0.45 二值化
        → 药片：8 连通域（面积 ≥ 30 px）
        → 抑菌圈：以药片质心为种子做测地分区（多源 BFS），相邻融合圈在此切开
    │
    ▼
Step 3  DrawPredictions()          主线程
        掩膜坐标还原到原图（各向异性：x×W0/640, y×H0/640）
        绘制分割轮廓（绿/蓝）、标签、等效直径
        结果图保存为 PNG 临时文件显示
    │
    ▼
Step 4  ProcessDetectedRegions()   后台线程
        每个 Area 按「分区归属」拿到自己的药片（兜底才用最近邻）
        CropAndDraw()：
          ├── 从原图直接像素拷贝裁剪（ExtractSubset，无插值损失）
          ├── 绘制轮廓、最小外接圆（Ritter算法）、椭圆拟合
          ├── 计算 Equiv.D / Enc.Circ / Pill Dia / Ctr Dist
          │   Inh.Dist / Dist/R / Result
          ├── ClassifyWithPreview()：
          │   彩色裁剪图 → ToGray(BT.601) → PIL 式 BICUBIC 224×224
          │   → 5 视图（原图 / 水平翻转 / 垂直翻转 / ±10°，fill=128）
          │   → class.onnx → 对 5 个视图的 logits 求平均再 softmax
          │   → 返回（类别, 置信度, 灰度图）
          └── 合成左侧面板（药片特写 + 测量数据 + 分类结果）
    │
    ▼
Step 5  ShowCurrentCroppedRegion() 主线程
        显示分析结果，启用翻页、Export、Save Raw Crops 按钮
```

---

## 界面说明

```
┌──────────────────────────────────┐
│         原图 / 分析结果图         │  HeightRequest=200, AspectFit
├──────────────────────────────────┤
│         状态栏（Status）          │  实时显示当前步骤
├──────────────────────────────────┤
│         进度条（分析时显示）      │  0% → 10% → 35% → 50% → 100%
├──────────────────────────────────┤
│   Upload    │   Analyze   │  Capture  │
├──────────────────────────────────┤
│         裁剪子图（左+右拼合）      │  HeightRequest=200
│  ┌────────┬──────────────────┐   │
│  │ 药片   │  分析结果大图     │   │
│  │ 特写   │  (轮廓+测量标注) │   │
│  │ Class  │                  │   │
│  │ Conf%  │                  │   │
│  └────────┴──────────────────┘   │
├──────────────────────────────────┤
│   ← Prev   │   N / Total   │  Next →  │
├──────────────────────────────────┤
│    Export   │  Save Raw Crops  │
└──────────────────────────────────┘
```

### 按钮功能

| 按钮 | 功能 | 启用条件 |
|---|---|---|
| Upload | 从系统相册选择图片 | 始终可用 |
| Analyze | 运行完整分析流程 | 已加载图片 + 模型就绪 |
| Capture | 调用系统摄像头拍照 | 始终可用（需摄像头权限） |
| ← Prev / Next → | 在多个抑菌区域之间翻页 | 分析完成后 |
| Export | 将原图+所有裁剪子图拼为长图，通过系统分享保存 | 分析完成后 |
| Save Raw Crops | 将所有药片原始裁剪图打包为 ZIP 输出 | 分析完成后 |

---

## 测量指标说明

每个抑菌区域（Area）与药片（Pill）配对后，在裁剪子图右侧显示以下指标：

| 指标 | 含义 | 计算方式 |
|---|---|---|
| **Equiv.D** | 等效圆直径（px） | `2 × √(轮廓面积 / π)`，基于 Shoelace 公式计算轮廓多边形面积 |
| **Enc.Circ** | 最小外接圆直径（px） | Ritter 近似算法，与 `cv2.minEnclosingCircle` 高度一致 |
| **Pill Dia** | 药片直径（px） | 药片轮廓的最小外接圆直径 |
| **Ctr Dist** | 抑菌圆心到药片圆心的距离（px） | 欧氏距离 |
| **Inh.Dist** | 抑菌距离（px） | `max(0, 抑菌区半径 + 圆心距 - 药片半径)` |
| **Dist/R** | 抑菌比值 | `抑菌距离 / 药片半径` |
| **Result** | 综合结果值 | `(Dist/R × 3 + 3) × 2` |
| **Class** | 药片分类 | 自研 DiskNet-small 输出，7 类抗生素名称 + 置信度（5 视图 TTA） |

### 可视化标注颜色

| 颜色 | 含义 |
|---|---|
| 🟢 绿色轮廓 | 抑菌区域（Area）分割轮廓 |
| 🔵 蓝色轮廓 | 药片（Yaoping）分割轮廓 |
| 🟡 黄色虚线圆 | 抑菌区域最小外接圆 |
| 🔵 蓝色虚线圆 | 药片最小外接圆 |
| 🩷 粉色虚线椭圆 | 椭圆拟合结果 |
| 🔴 红色标尺线 | 抑菌距离（Inh.Dist） |
| 🔵 蓝色标尺线 | 药片半径（r） |
| ⚪ 白色虚线 | 圆心连线 |

---

## 模型说明

### best.onnx（分割模型，本项目自研 semseg）

| 项目 | 参数 |
|---|---|
| 架构 | ResNet18 编码器 + U-Net 解码器，双通道语义分割（14.4 M 参数） |
| 训练代码 | `Antibacterial zone mask/semseg/`（原生 PyTorch，不依赖 YOLO/Ultralytics） |
| 权重来源 | `Antibacterial zone mask/runs_semseg/20260918_051406_resnet18/best.pt` |
| 导出脚本 | `Antibacterial zone mask/semseg/export_onnx.py` |
| 输入 | `input` `[1,3,640,640]` float32，**RGB 0..255**（ImageNet 归一化已烘焙进图） |
| 输出 | `probs` `[1,2,640,640]` float32，逐通道 sigmoid 概率（通道 0 = Area，1 = Yaoping） |
| 阈值 | 0.45（两通道共用，来自 `semseg/config.py`） |
| TTA | 5 视图（0/90/180/270 + 水平翻转）取平均；`ZoneSegModel.UseTta` 可关 |
| 最小面积 | 30 px（640 尺度） |
| 实测 | 测试集 20 张 / 13 块训练未见过的独立平板：Area Dice **0.9561**、Yaoping Dice **0.9257**（含 TTA）；CPU 单视图约 0.33 s |

### class.onnx（分类模型，本项目自研 DiskNet-small）

| 项目 | 参数 |
|---|---|
| 架构 | **DiskNet-small**：固定 DoG 高通分支 + 两层 stem（224→112）+ 四段残差（32/64/128/256 @ 112/56/28/14）+ SPPF + 多尺度 token 门控头（3.89 M 参数） |
| 训练代码 | `药片CNN/train_disknet.py`（`disknet_model.py`） |
| 权重来源 | `药片CNN/runs/disknet224_small/best.pt` |
| 导出脚本 | `药片CNN/export_onnx_disknet.py` |
| 输入 | `image` `[N,3,224,224]` float32，**灰度复制 3 通道、0..255**（`(x/255-0.5)/0.25` 已烘焙进图） |
| 输出 | `logits` `[N,7]`（未过 softmax）与 `probs` `[N,7]`（softmax）；App 取 `logits`，5 视图平均后再 softmax |
| 类别 | CRO / DA / E / LEV / LZD / P / VA |
| TTA | 5 视图：原图、水平翻转、垂直翻转、±10° 旋转（fill=128） |
| 精度 | batch-disjoint 测试划分（28 张、来自 5 个采集批次）：单模型 **0.857**，加 TTA **0.929**；论文报告的 5 seed 集成同为 0.929 且 macro-F1 0.927（本 App 打包的是**单 seed + TTA**，与 0.929 对应） |

> ⚠️ 这两个 ONNX 与 `_original_onnx/` 里备份的旧 YOLOv8 模型**不通用**：
> 输入预处理与输出结构都变了，必须配套使用当前版本的 `ZoneSegModel.cs` / `PillClassifier.cs`。

### 实例化：语义模型怎么给出「一个药片一个圈」

`semseg` 只输出两类语义掩膜，实例化在 `ZoneSegModel.BuildInstances()` 里做：

1. **药片** = 药片通道 `>0.45` 的 8 连通域（面积 ≥ 30 px），一步得到实例与置信度；
2. **抑菌圈** = 以每块药片质心为种子，在抑菌圈掩膜内做**多源 BFS 测地分区**：
   相邻药片的圈在像素上会融合成一块，纯连通域无法区分归属，而测地分区只在
   「相邻圈相接处」切开、不外扩，因此每块药片最多得到一块圈区域；
3. 没分到区域的药片 → `HasZone=false`，不产生圈实例（即「无抑菌圈」的药片）；
4. 该归属关系直接用于 Step 4 的配对（`ImageProcessor` 优先按归属匹配，最近邻仅作兜底）。

实测（20 张 test 图，与本机 C# 端逐图对比真值）：

| 类别 | 预测 | 真值 | Precision | Recall | F1 | 匹配平均 IoU |
|---|---|---|---|---|---|---|
| `Area` 抑菌圈 | 136 | 110 | 0.802 | 0.991 | 0.886 | 0.757 |
| `Yaoping` 药片 | 141 | 143 | 0.993 | 0.979 | 0.986 | 0.858 |

语义（掩膜并集）：Area Dice 0.957 / Yaoping Dice 0.924。
`Area` 的假阳性主要来自「本来没有抑菌圈的药片也被分到了一小块区域」，
这是阈值 0.45 与测地分区的固有行为（与论文口径下的 `Area` 多边形数少于药片数一致）。

## 预处理流程

### 分割模型预处理（与 semseg 训练严格一致）

```
原图（任意尺寸；按存储像素直接解码，不读 EXIF 方向）
  → 双线性 resize 到 640×640（**非等比、无灰边**）
  → 取 RGB，float32 0..255
  → 送入 best.onnx（均值/方差归一化在图内）
```

⚠️ 与旧版 YOLOv8 的 Letterbox（等比缩放 + 灰边 114）完全不同：本模型是按
「整图拉伸到正方形」训练的（`semseg/dataset.py`），再补灰边会引入训练时没见过的输入分布。

### 分类模型预处理（与 药片CNN 的 Dataset / ensemble_eval 一致）

```python
gray   = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)          # BT.601
im     = Image.fromarray(gray).resize((224, 224), Image.BICUBIC)
views  = [im,
          im.transpose(Image.FLIP_LEFT_RIGHT),
          im.transpose(Image.FLIP_TOP_BOTTOM),
          im.rotate(-10, resample=Image.BICUBIC, fillcolor=128),
          im.rotate(+10, resample=Image.BICUBIC, fillcolor=128)]
# 每个视图都给 ONNX（数值 0..255，复制 3 通道；(x/255-0.5)/0.25 在图内）
# logits 先按视图求平均，再 softmax
```

**这里不再有 CLAHE / 锐化 / Gamma**：DiskNet 的数据集是**原始灰度裁剪**（`药片CNN/dataset_augmented224`，
已核验它与原始裁剪的逐像素差只有 0.2–0.5 灰度级，而做增强后会差 23–56 灰度级），
所以 App 只需灰度 + 缩放到 224。旧的 CLAHE(2.0/3.0)+锐化+Gamma 链路属于 PillCNN 时代
（那份数据集确实是增强过的），换模型后一并去掉。

C# 端（`PillClassifier.cs`）复刻这条链路，包括：

* **PIL 风格的可分离 BICUBIC**（向下缩放时按 scale 放大滤波支撑，即抗锯齿）
* **PIL 语义的旋转**（正角度 = 逆时针，中心 (w/2,h/2)，落点超出用 fill=128）
* 5 视图一次前向（ONNX 动态 batch = 5），**对 logits 求平均后再 softmax**

实测（28 张 batch-disjoint 测试裁剪，C# ↔ Python 参考实现）：
准确率 **0.9286 / 0.9286**（与论文的 0.929 一致）；
同一批图的**原始裁剪**（App 真实输入路径）也是 **0.9286 / 0.9286**；
C# ↔ 参考 top-1 一致率 28/28，同类别置信度平均差 0.0036。

## Python 训练工具与模型导出

App 用到的两个模型都由本仓库的 Python 项目训练，App 目录下不再放训练脚本：

| 模型 | 训练/导出位置 | 关键命令 |
|---|---|---|
| 分割 `best.onnx` | `Antibacterial zone mask/semseg/` | `python -m semseg.prepare` → `python -m semseg.train --encoder resnet18` → `python -m semseg.export_onnx --weights runs_semseg/<run>/best.pt --out "../Antibacterial zone/_new_models/best.onnx"` |
| 分类 `class.onnx` | `药片CNN/` | `python train_disknet.py --variant small --img-size 224` → `python export_onnx_disknet.py`（默认导出 `runs/disknet224_small/best.pt`） |

分类数据集 `药片CNN/dataset_augmented224` 是**原始灰度裁剪**（未增强）+ 离线增广，
所以推理端不再需要增强脚本；历史脚本 `imagepross.py` / `药片CNN/enhance_lib.py`
（灰度 → CLAHE → 锐化 → Gamma）属于 PillCNN 时代的数据集，本 App 已不使用。
训练好的模型导出后复制到 `Resources/Raw/` 即可（导出脚本会
自动做 PyTorch↔ONNX 数值一致性核验）。

### 本机核验（不需要 MAUI 工作负载）

`Antibacterial zone/_verify/` 是一个轻量控制台工程，把 `ZoneSegModel.cs` 与
`PillClassifier.cs` **原样**编译进去，可以直接跑推理并与 Python 参考实现/真值对比：

```bash
# 1) 编译：直接引用 App 上次构建产物里的 ONNX Runtime / SkiaSharp，不需要 NuGet 联网还原
dotnet build "Antibacterial zone/_verify/Verify.csproj" -c Release

# 2) 分割：对一批图跑，导出 JSON + 每实例 640×640 掩膜 + 可视化
Verify.exe seg "Antibacterial zone/_new_models/best.onnx" "Antibacterial zone mask/data/test/images" out_seg
Verify.exe seg ... out_seg_notta --no-tta

# 3) 分类：对一批药片裁剪图跑，导出类别/置信度 + 增强后的灰度图
Verify.exe cls "Antibacterial zone/_new_models/class.onnx" cls_test out_cls/cls_csharp.json

# 4) 与真值标注对比（实例级 P/R/F1 + 语义 Dice）
python "Antibacterial zone/_verify/compare_seg.py" out_seg
```

> `compare_seg.py` 会处理 EXIF 方向差异：OpenCV 解码会自动应用 EXIF，而 SkiaSharp
> （App 端）不应用，两者相差一个 90°/180° 旋转。脚本会先把预测掩膜旋到与标注同一
> 坐标系再算指标——否则指标会看似崩到 Dice≈0.05（这是坐标系问题，不是模型问题）。

---

## 环境要求与依赖

### C# / MAUI 应用

| 依赖 | 本项目版本 | 用途 |
|---|---|---|
| .NET | 10.0（net10.0-android / -windows / -ios / -maccatalyst） | MAUI 运行时 |
| .NET MAUI | 10.0.20 | 跨平台 UI |
| Microsoft.ML.OnnxRuntime | 1.24.3 | ONNX 模型推理 |
| SkiaSharp (+ Views.Maui) | 3.119.2 | 图像处理与绘制 |

目标平台：Android 8.0+（API 28+）、iOS 15+、Windows 10.0.17763+

### Python 训练环境

已实测通过：Python 3.10.6 / torch 2.14.0+cu126 / torchvision 0.29.0+cu126 /
opencv-python 5.0.0 / numpy / tqdm / matplotlib / onnx 1.22.0 / onnxruntime 1.23.2
（CUDA 可用，RTX 3070 Laptop 8 GB）。

**不使用 ultralytics**：两个模型都是原生 PyTorch 实现，训练与导出都不依赖检测框架。

---

## 部署步骤

### 1. 准备模型文件

`Resources/Raw/best.onnx` 与 `Resources/Raw/class.onnx` 必须来自本项目自研模型，
并与 `ZoneSegModel.cs` / `PillClassifier.cs` 的接口约定一致（见「模型说明」）。
旧 YOLOv8 版本的备份在 `Antibacterial zone/_original_onnx/`，仅作存档，**不要混用**。

### 2. 构建与运行

```bash
dotnet build -t:Run -f net10.0-android
dotnet build -t:Run -f net10.0-windows10.0.19041.0
```

### 3. Android 权限配置

在 `Platforms/Android/AndroidManifest.xml` 中确保包含：

```xml
<uses-permission android:name="android.permission.CAMERA" />
<uses-permission android:name="android.permission.READ_EXTERNAL_STORAGE" />
```

### 4. 重新训练后替换模型

用上表的两条导出命令生成新的 ONNX，复制到 `Resources/Raw/` 覆盖同名文件即可；
**不需要**再改 C# 代码（只要接口约定不变）。若改了输入尺寸/归一化方式，
必须同步修改 `ZoneSegModel.cs` / `PillClassifier.cs` 里的常量，并重跑 `_verify` 核验。

---

## 常见问题

**Q: 分析后显示 "No targets detected"**  
A: 两个通道的概率都低于阈值 0.45。确保培养皿清晰可见、光线均匀、药片与抑菌圈对比度明显；
若整批图都检不出，先确认 `Resources/Raw/best.onnx` 是自研 semseg 版本（输入 0..255 RGB、
输出 `probs [1,2,640,640]`），而不是备份里的旧 YOLOv8 模型。

**Q: 药片分类置信度很低（< 50%）**  
A: 三个常见原因：① 裁剪图太小/太糊；② 药片上的缩写被完全遮挡或反光；
③ 图像方向异常（见下方 EXIF 说明）。先用 `Save Raw Crops` 导出裁剪图，
再跑 `_verify` 的 `cls` 子命令复现同一输入下的输出，判断是预处理问题还是模型问题。

**Q: 模型加载失败 "Model load failed"**  
A: 检查 `Resources/Raw/` 下是否存在 `best.onnx` 与 `class.onnx`，构建属性为 `MauiAsset`；
另外确认替换模型后**重新构建过**（bin 目录里会残留旧模型的副本）。

**Q: Capture 按钮显示 "Camera not available (emulator?)"**  
A: 模拟器不支持摄像头，请在真机上运行。

**Q: Export 后图片在哪里**  
A: 点击 Export 后会弹出系统分享菜单，选择"保存到相册"或发送至其他应用。

**Q: Save Raw Crops 的用途是什么**  
A: 将 App 实际送入分类器的药片原始裁剪图（未经增强处理）打包为 ZIP 输出。
用这些图可以直接喂给 `_verify` 的 `cls` 子命令或 `药片CNN/preprocess_class_reference.py`，
排查 C# 与 Python 推理结果不一致的问题，也是扩充训练数据的来源。

**Q: 照片在 App 里显示的方向和相册里不一样**  
A: 这是已知行为：SkiaSharp 的 `SKBitmap.Decode` **不应用 EXIF 方向**，而手机相册
按 EXIF 旋转显示，所以本 App 用的是「存储像素方向」。所有测量量（等效直径、
抑菌距离、比值）都是旋转不变的，因此不影响结果；但如果以后要做「显示方向与相册一致」，
需要在解码后按 `SKCodec.EncodedOrigin` 旋转，并注意分类器的训练裁剪图也是按当前
（未旋转）方向采集的。

---

## 数据流转图

```
实验拍照
    │
    ▼
手机摄像头 / 相册
    │ 原始彩色大图（任意分辨率）
    ▼
ZoneSegModel（自研 semseg，best.onnx）
    │ 整图 resize 640 → 2 通道概率 → 阈值 0.45
    │ 药片=8连通域；抑菌圈=以药片质心为种子的测地分区
    │ 输出：每实例 640 掩膜 + 原图坐标包围盒 + 置信度 + 归属
    ▼
ImageProcessor
    │ Area ↔ Pill 配对（优先用分区归属）
    │ 从原图直接像素裁剪（无插值损失）
    │ 几何计算：等效径、外接圆、椭圆、抑菌距离
    │
    ├──→ PillClassifier（自研 DiskNet-small，class.onnx）
    │         灰度化 → BICUBIC 224×224
    │         5 视图（原图 / hflip / vflip / ±10°）→ ONNX
    │         logits 平均 → softmax → 类别（CRO/DA/E/LEV/LZD/P/VA）+ 置信度
    │
    ▼
可视化合成图（左侧面板 + 右侧标注图）
    │
    ├──→ 屏幕显示（CroppedImage）
    ├──→ Export（长图 JPEG）
    └──→ Save Raw Crops（ZIP）
```
