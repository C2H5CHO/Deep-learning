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

conv1 = nn.Conv2d(
    in_channels=3, # 输入通道数为3，对应RGB彩色图像
    out_channels=16, # 输出通道数为16，表示有16个卷积核
    kernel_size=3 # 卷积核大小为3x3
)
data_conv1 = conv1(data)
print(f"第一层卷积层的形状: {data_conv1.shape}")

conv2 = nn.Conv2d(
    in_channels=16, # 输入通道数为16，对应上一层的输出通道数
    out_channels=32, # 输出通道数为32，表示有32个卷积核
    kernel_size=3 # 卷积核大小为3x3
)
data_conv2 = conv2(data_conv1)
print(f"第二层卷积层的形状: {data_conv2.shape}")

print('--'*50)
data2 = torch.ones(size=(10, 3, 28, 28))
conv1_ = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=3)
conv2_ = nn.Conv2d(in_channels=6, out_channels=10, kernel_size=3)
conv3_ = nn.Conv2d(in_channels=10, out_channels=16, kernel_size=5, stride=2, padding=1)
"""
    - in_channels=10: 输入通道数，来自上一层conv2_的输出
    - out_channels=16: 输出通道数，增加特征图的深度
    - kernel_size=5: 5x5卷积核，比3x3更大的感受野
    - stride=2: 步长为2，实现2倍下采样，减小特征图尺寸
    - padding=1: 填充1像素，保持边界信息完整性

特征图尺寸变化计算：
假设输入特征图尺寸为 H×W
输出尺寸 = (H + 2×padding - kernel_size) / stride + 1
例如：输入28×28 → 输出(28+2×1-5)/2+1 = 13×13
"""
conv4_ = nn.Conv2d(in_channels=16, out_channels=3, kernel_size=5, stride=3, padding=2)
