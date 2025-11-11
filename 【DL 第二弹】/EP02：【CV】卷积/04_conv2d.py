import torch
from torch import nn

# 创建一个4维张量，模拟批量图像数据
data = torch.ones(size=(10, 3, 28, 28))
"""
形状为 (batch_size, channels, height, width)
    - batch_size=10: 批次大小为10，表示有10个样本
    - channels=3: 通道数为3，对应RGB彩色图像
    - height=28, width=28: 图像高度和宽度均为28像素
"""

