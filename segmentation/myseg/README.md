# myseg — 抑菌圈实例分割（ResNet 编码器 + U-Net 解码器）

原生 PyTorch 实现，**不依赖 YOLO / Ultralytics / 任何检测框架**。

---

## 1. 它做什么

从一张培养皿照片里分割出两类区域，并给出**实例级**结果：
每个药片、每个抑菌圈都是**独立实例**，而且知道**哪个圈属于哪个药片**
（即原来 YOLOv8-seg 那种输出）。

| 类别 id | 名称 | 含义 |
|---|---|---|
| 0 | `Area` | 抑菌圈区域 |
| 1 | `Yaoping` | 药片（6 mm 标准纸片，用作内标标定） |

### 实例化是怎么做的（关键）

模型本体输出的是**语义**掩膜（每类一张图），不是实例掩膜。实测本项目数据：

| 类别 | 人工标注多边形数（=真实实例数） | 语义掩膜连通域数 | 结论 |
|---|---|---|---|
| `Yaoping` 药片 | 140 | **140** | 天然分离，**连通域即实例** |
| `Area` 抑菌圈 | 116 | **45** | 相邻圈融合，**必须拆分** |

所以实例化分两步（见 `myseg/instance.py`）：

1. **药片**：连通域直接就是实例（20 张测试图无一例外）
2. **抑菌圈**：以**药片质心为种子**，在圈区域内做**测地距离划分**（多源 BFS）
   * 每个圈像素归属「沿区域内路径最近的那个药片」
   * **不外扩** —— 圈的真实外缘完全保留，只在相邻圈相接处切开
   * 没有圈的药片自然拿不到区域，如实输出 `has_zone=False`

这比在整幅图上做普通 Voronoi 好：普通 Voronoi 的分界线是直线，会切进圈内部、
且不受圈边界约束。

### 实例级客观指标（epoch 14 权重，训练仍在进行）

以人工标注的 **116 个圈 / 140 个药片实例多边形**为基准，IoU≥0.5 视为匹配
（同 COCO segm AP@0.5 口径）：

| 类别 | 预测 | GT | TP | Precision | Recall | 匹配平均 IoU |
|---|---|---|---|---|---|---|
| `Yaoping` | 137 | 140 | 134 | **0.98** | **0.96** | 0.865 |
| `Area` | 137 | 116 | 111 | **0.81** | **0.96** | 0.760 |

**归属正确率 85.7%**（每个预测圈是否落在「含它那个药片」的圈内）。

直径计算链路：

```
预测 mask
  → 药片连通域 = 药片实例
  → 药片质心做种子，测地划分圈区域 = 圈实例 + 归属
  → 标定：mm_per_px = 6.0 / median(药片等效直径)     ← 6 mm 标准纸片做内标
  → 抑菌圈直径 = 药片等效直径 + 2 × 平均环宽
```

平均环宽由面积差推得（`环面积 / 圈等效周长`），对椭圆或不规则圈比
`minEnclosingCircle` 稳。

---

## 2. 目录结构

```
Antibacterial zone mask/
├─ images/            78 张原始照片          ← 只读，不动
├─ labels/            78 个 YOLO-seg 标签     ← 只读，不动
├─ myseg/             本包
│   ├─ config.py          全局配置（唯一事实来源）
│   ├─ split_dataset.py   按物理平板划分 train/val/test
│   ├─ augment.py         训练/验证集增强（纯 OpenCV）
│   ├─ dataset.py         torch Dataset
│   ├─ model.py           ResNet + U-Net
│   ├─ instance.py        ★ 实例提取：每个药片/圈独立 + 归属关系
│   ├─ instance_viz.py    ★ 实例可视化（每实例一色 + 归属连线 + 编号）
│   ├─ eval_mask.py       ★ mask 质量评估（逐类别 Dice/IoU/P/R）
│   ├─ metrics.py         像素层指标（+ 可选的直径测量）
│   ├─ train.py           训练与测试集评估
│   ├─ predict.py         推理，输出实例 CSV
│   ├─ export_onnx.py     ONNX 导出 + 一致性校验
│   ├─ visualize.py       通用可视化
│   └─ split.json         划分清单（可追溯）
├─ data/              划分与增强产物
│   ├─ train/{images,labels}      38 张原图
│   ├─ train_aug/{images,labels}  38 原图 + 570 增强
│   ├─ val/{images,labels}        20 张原图
│   ├─ val_aug/{images,labels}    20 原图 + 300 增强
│   └─ test/{images,labels}       20 张原图（**不增强**）
├─ runs_unet/         训练输出
└─ export/            ONNX 与推理输出
```

---

## 3. 环境

依赖已装好（写入 base site-packages），验证：

```bash
python -c "import torch, torchvision, cv2, onnxruntime; print('ok')"
```

预期：`torch 2.14.0+cu126`、`torchvision 0.29.0+cu126`、`cv2 5.0.0`、
`onnxruntime 1.23`，CUDA 可用（RTX 3070 Laptop 8 GB）。

**本实现不需要 albumentations** —— 数据增强是纯 OpenCV + NumPy 自己写的（见 `augment.py`）。
这样既没有 C 扩展依赖的下载问题，也避开了第三方库在非 ASCII 路径上的坑。

### ⚠️ 非 ASCII 路径（本项目路径含中文）

Windows 上 `cv2.imread` / `cv2.imwrite` 遇到中文路径会**静默失败**（返回 None / False，
不抛异常）。实测：

```
cv2.imread("...抑菌区域分析_返修/.../x.jpg")        -> None
cv2.imread(Path("...抑菌区域分析_返修/.../x.jpg"))  -> None
cv2.imdecode(np.fromfile(p, np.uint8), ...)         -> (4032, 3024, 3)   ✅
```

因此**本包内一律不直接调用 `cv2.imread/imwrite`**，统一走 `io_utils.imread/imwrite`。

### 本机实测性能（resnet34 @ 640×640）

| 配置 | ms/step | 峰值显存 |
|---|---|---|
| batch 2 | 131 | 1.7 GiB |
| **batch 4（采用）** | **243** | **2.9 GiB** |
| batch 8 | 2930 ⚠️ | 5.6 GiB |

batch 8 会异常降速，故默认用 batch 4。数据加载（8 workers）约 59 图/秒，
训练一轮约 60–100 秒。

---

## 4. 跑一遍完整流程

```bash
cd "Antibacterial zone mask"

# ① 划分（38 / 20 / 20，按物理平板分组，防泄露）
python -m myseg.split_dataset

# ② 增强（train + val；test 保持原图）
python -m myseg.augment
python -m myseg.augment --preview      # 生成对齐检查图，务必人工看一眼

# ③ 训练（默认 resnet34；换 resnet18 加 --encoder resnet18）
python -m myseg.train
#    或直接双击 run_train.cmd（内含论文口径的超参，并把日志写到 runs_unet\train_*.log）

# ④ 导出 ONNX，并校验与 PyTorch 一致
python -m myseg.export_onnx --weights runs_unet/<run>/best.pt

# ⑤ 对新图推理 → 每图一个实例 CSV（药片/圈实例 + 归属）
python -m myseg.predict --weights runs_unet/<run>/best.pt --source images

# ⑥ 实例可视化：每实例一色 + 编号 + 药片↔圈 归属连线
python -m myseg.instance_viz --weights runs_unet/<run>/best.pt --split test

# ⑦ mask 质量评估：逐类别 Dice / IoU / Precision / Recall
python -m myseg.eval_mask --weights runs_unet/<run>/best.pt --split test
```

### 实例输出的三个入口

| 命令 | 给什么 | 输出位置 |
|---|---|---|
| `myseg.predict` | 每图一个实例 CSV：`disk_id / zone_id / 归属 / 直径(mm)` | `export/predict/instances.csv` |
| `myseg.instance_viz` | 四联图：原图 \| 语义 \| **实例+归属连线** \| 差异 | `export/instance/<权重名>_<split>/` |
| `myseg.eval_mask` | 逐类别 mask 指标表（能直接贴进稿子） | `export/mask_eval/<权重名>_<split>/report.md` |

`instance_viz` 出的图就是「像 YOLO mask 那样」：每个实例一种颜色，
药片标 `D1…D7`、抑菌圈标 `Z1…Z7`，`D3→Z5` 表示 3 号药片对应 5 号圈，
中间有连线；无圈药片用灰色标出并显示 `D6(-)`。

### 输入尺寸与超参：对齐论文

`config.py` 里的关键项**取自论文**，不是随便定的：

| 项 | 值 | 依据 |
|---|---|---|
| 输入尺寸 | **640 × 640** | Additional file 1, Table S6「Input」行；Manuscript.tex L175 |
| seed | 0 | 同上 |
| 初始学习率 | 0.01 | 同上 |
| 最大轮数 | 300 | 同上 |
| 早停 patience | 30 | 同上 |
| batch | 4（论文写 100） | **差异项**：论文那版是 YOLOv8n-seg，本实现换成 ResNet+U-Net，参数量更大；本机 8 GB 显存照搬 100 会直接 OOM |

这一段在 `config.py` 里也有逐条注释，写稿时可直接引用。

---

## 5. 测量口径：完全按论文的定义

这一点很关键 —— 直径不是用「等效圆」或「minEnclosingCircle」算的，而是照抄论文
Methods 里 “Calibration, Measurement, and Application Workflow” 一节的原文：

> For each disk–zone pair, a local sector with an angular range of **±15°** was defined
> **along the radial direction between the disk centre and the estimated centre of the
> Petri dish**. Radial intersections with the disk and inhibition-zone masks were sampled
> within this sector. Representative pixel diameters were calculated as the **mean**
> across the sampled directions.

实现要点（`labels.measure_zone_diameters`）：

1. 皿心 = 全部药片质心的中位数（论文：“used to estimate the geometric centre of the Petri dish”）
2. 扇区方向 = **皿心 → 药片**，即以皿外那一侧为中心 ±15° 采样
3. 每条射线取该方向上圈掩膜的**最外缘**距离
4. 直径 = 2 × 扇区内各方向距离的**均值**
5. 标定 `s = 6 mm / D_disk,px`，`D_zone,mm = D_zone,px × s`

> ⚠️ 为什么扇区必须朝皿外：相邻药片的抑菌圈在像素上会融合成一个连通域（实测某皿
> 8 个药片只有 5 个 Area 连通域）。若朝内采样，射线会打到邻片的圈或融合边界上，
> 读数严重偏大。实测对比：朝外采样 98.3% 的读数落在论文量程 12–40 mm 内，
> 全方向取均值只有 80%，取最大值只有 17%。

---

## 6. 划分方案（重要）

78 张照片来自 **31 块物理平板**。`026`–`033` 是多角度/多高度重复拍摄，
单块最多 8 张。**划分必须按平板整体进行**，否则同一块皿的照片会同时出现在
训练集和测试集，指标虚高。

平板规模直方图：单张 23 块、4 张 1 块、5 张 1 块、6 张 1 块、8 张 5 块。

> ⚠️ 一个绕不过的约束：**5 块 8 张平板合计 40 张，已超过 train 需要的 38 张**，
> 所以不可能「大平板全进训练」。`split_dataset.py` 会枚举所有可行方案，
> 按「test 平板数 → val 平板数 → train 平板数」的优先级选最优解。

实际结果（38 / 20 / 20 张精确命中）：

| split | 原图 | 平板数 | 组成 | 增强后 |
|---|---|---|---|---|
| train | 38 | 5 | 027(8) + 028(6) + 029(8) + 030(8) + 031(8) | 608 |
| val | 20 | 6 | 032(8) + 033(5) + 026(4) + 003/015/020 | 136 |
| test | 20 | 20 | 20 块单张平板，每块独立 | **20（不增强）** |

测试集 20 张来自 **20 块互不相同的平板**，训练集未见过其中任何一块。
完整去向见 `myseg/split.json`。

---

## 7. 评估口径

### 主入口：只看 mask 区域识别

```bash
python -m myseg.eval_mask --weights runs_unet/<run>/best.pt --split test
```

产出在 `export/mask_eval/<权重名>_<split>/`：

| 文件 | 内容 |
|---|---|
| `report.md` | 逐类别指标表 + 逐图分布表，可直接贴进稿子 |
| `report.json` | 机器可读版 |
| `metrics.csv` | 每张图一行，便于做统计检验 |
| `vis/<图名>.jpg` | 四联图：原图 \| 人工 \| 预测 \| **差异图** |

逐类别指标：**Dice / IoU / Precision / Recall / F1**，还给出正像素数（判断指标是否稳）、
TP/FP/FN 计数、以及「该类缺失的图数」（人工标注里本来就没这个类的图片数——
这一列必须一起看，否则 Recall 会被误读）。

差异图配色：**绿=命中(TP)，红=误检(FP)，蓝=漏检(FN)**，左上角标出像素数与 IoU，
一眼就能看出错在哪。

指标由**跨图累计**的像素级混淆计数算出（不是逐图平均），更稳。

### 直径测量（已降级为可选）

`myseg.metrics` 里保留了一套按论文 ±15° 扇区定义的直径测量，`train.py` 结束时
也会在 `test_report.txt` 里打印。如果不做直径一致性分析，可以忽略这部分输出，
只看 `eval_mask` 的 mask 指标。

---

## 8. 已知取舍

1. **`val` 也做了增强**（按要求）。代价是验证指标会偏乐观。
   测 `eval_mask` 时默认用 val 的**原图**（不加 `--use-aug`），这样数字是干净的。
2. **训练集只有 5 块平板**，多样性受限 —— 这是 78 张 / 38 张的硬约束，
   不是配置失误。增强（每张 15 个变体）部分补偿了这一点。
3. **2 张竖幅图**（`3ec0e627-008` 是 2928×3329，`7d1068f8-020` 是 1280×1707）
   在非等比缩放到 640×640 时会有形变。其余 76 张都是 4032×3024。
4. **数据集中没有 6 mm 无圈观测**（`Area` 多边形 454 个 < `Yaoping` 555 个，
   差 101 个），也就是说训练集里几乎没有耐药/无圈难例。模型把「没有抑菌圈的药片」
   正确判成无区域的能力，无法从本数据集评估。
