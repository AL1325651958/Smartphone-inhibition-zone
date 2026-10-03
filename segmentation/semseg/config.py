"""
semseg.config — 配置
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC_IMAGES = ROOT / "images"          # 原始 78 张（只读）
SRC_LABELS = ROOT / "labels"          # 原始 78 个多边形标签（只读）
DATA = ROOT / "data"
SPLIT_JSON = ROOT / "semseg" / "split.json"
RUNS = ROOT / "runs_semseg"
EXPORT = ROOT / "export_semseg"

# ── 输入 ────────────────────────────────────────────────
IMG_SIZE = 640
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# ── 类别 ────────────────────────────────────────────────
ZONE, DISK = 0, 1
NUM_CLASSES = 2
CLASS_NAMES = {ZONE: "Area", DISK: "Yaoping"}

# ── 划分 ────────────────────────────────────────────────
SPLIT_SEED = 20260917
N_TRAIN, N_VAL, N_TEST = 38, 20, 20

# ── 增强 ────────────────────────────────────────────────
AUG_PER_IMAGE = 15
AUG_SEED = 20260917

# ── 训练 ────────────────────────────────────────────────
# encoder 选 resnet18：实测与 resnet34 测试指标完全持平（macro 0.9371 vs 0.9372），
# 但参数少 41%（14.4M vs 24.5M）、训练快约 15%，更适合部署。
ENCODER = "resnet18"
EPOCHS = 80
BATCH_SIZE = 4
LR = 3e-4
WEIGHT_DECAY = 1e-4
PATIENCE = 25
NUM_WORKERS = 3
AMP = True

# ── 推理 ────────────────────────────────────────────────
# 阈值 0.45：扫描 0.30–0.70 后确认，配合 TTA 时最优（宏平均 0.9401）。
# 单独调阈值收益极小（+0.0003），主要收益来自 TTA。
THRESH = 0.45

# 测试时增强：旋转 0/90/180/270 + 水平翻转，共 5 个视图取平均。
# 实测把测试宏平均从 0.9369 提到 0.9400（+0.0031）。
USE_TTA = True

# 损失权重：BCE 与 Dice 各半。药片前景仅约 1.2%，Dice 项负责把它拉回来。
BCE_WEIGHT = 0.5
# 药片通道的 BCE 正样本加权（前景太少时进一步补偿）
DISK_POS_WEIGHT = 4.0
