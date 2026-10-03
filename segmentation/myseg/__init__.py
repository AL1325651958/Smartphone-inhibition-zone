"""myseg — 抑菌圈分割模型（原生 PyTorch）。

模块
----
config          全局配置
labels          多边形分割标签 <-> 多边形 互转、mask 栅格化、几何测量
split_dataset   按物理平板划分 train/val/test
augment         训练/验证集数据增强（几何 + 光度，同步变换多边形）
dataset         torch Dataset
model           ResNet 编码器 + U-Net 解码器
metrics         每类 IoU / Dice / Precision / Recall，实例级直径误差
train           训练 + 验证 + 早停 + 测试
predict         对任意图片推理，输出标注图与直径读数
export_onnx     ONNX 导出 + onnxruntime 一致性校验
"""
