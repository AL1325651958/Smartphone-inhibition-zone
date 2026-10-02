"""
myseg/config.py — 全局配置（单一事实来源）

所有路径都相对项目根目录（Antibacterial zone mask/）解析，脚本从任何位置运行都一致。
"""
from pathlib import Path

# ── 路径 ────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent     # .../Antibacterial zone mask
SRC_IMAGES = ROOT / "images"                      # 原始 78 张
SRC_LABELS = ROOT / "labels"                      # 原始 78 个 YOLO-seg 标签

DATA = ROOT / "data"                              # 划分输出
SPLIT_JSON = ROOT / "myseg" / "split.json"        # 划分清单（可追溯）
RUNS = ROOT / "runs_unet"                         # 训练输出
EXPORT = ROOT / "export"                          # ONNX 输出

# ── 类别 ────────────────────────────────────────────────
CLASS_NAMES = {0: "Area", 1: "Yaoping"}           # 0=抑菌圈区域, 1=药片
NUM_CLASSES = 2

# ── 划分 ────────────────────────────────────────────────
SPLIT_SEED = 20260917
N_TRAIN, N_VAL, N_TEST = 38, 20, 20

# ── 输入尺寸 ────────────────────────────────────────────
# 必须与论文一致。依据：
#   Additional file 1, Table S6 「Input: Plate image resized to 640 × 640 pixels」
#   Manuscript.tex L175 「trained with images resized to 640 × 640 pixels」
IMG_SIZE = 640
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# ── 增强 ────────────────────────────────────────────────
AUG_PER_IMAGE = 15                                # 每张原图生成多少增强样本
AUG_SEED = 20260917

# ── 训练超参 ────────────────────────────────────────────
# 以下「对齐论文」的项取自 Additional file 1 与 Methods：
#   batch size 100 / seed 0 / lr 0.01 / max 300 epochs / patience 30
#   （论文实际训到 145 轮，最佳 epoch 115）
#
# ⚠️ 差异说明：论文那一版是实例分割框架（YOLOv8n-seg），batch=100 是配合该架构
#    与其训练集规模设定的。本实现改为 ResNet+U-Net，参数量更大、激活显存更高，
#    本机 RTX 3070 Laptop 只有 8 GB，照搬 batch=100 会直接 OOM。
#    因此 batch 按本机实际设为 8，其余项均与论文一致，差异在此显式记录。
ENCODER = "resnet34"          # resnet18 | resnet34
ENCODER_WEIGHTS = "imagenet"  # imagenet | None
EPOCHS = 300                  # 对齐论文：maximum 300 epochs
PATIENCE = 30                 # 对齐论文：early-stopping patience 30
LR = 0.01                     # 对齐论文：initial learning rate 0.01
SEED = 0                      # 对齐论文：random seed 0
BATCH_SIZE = 4                # 非论文值（论文 100）。本机实测：resnet34@640 用 batch 8 会掉到 2.9 s/step，batch 4 为 0.24 s/step 且仅占 2.9 GiB
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 8
AMP = True                    # 混合精度，省显存

# ── 评估 ────────────────────────────────────────────────
MASK_THRESH = 0.5             # 概率 → 二值 mask
MIN_BLOB_AREA = 40            # 小于该像素面积的分割块丢弃（640 尺度下；按面积等比于 512 时的 30）
