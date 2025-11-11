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
data2_conv1 = conv1_(data2)
print(f"第一层卷积层的形状: {data2_conv1.shape}")
"""
data2_conv1_参数说明：
    - batch_size=10
    - channels=out_channels=6
    - height=width=(28+2×0-3+1)=26
"""

conv2_ = nn.Conv2d(in_channels=6, out_channels=10, kernel_size=3)
data2_conv2 = conv2_(data2_conv1)
print(f"第二层卷积层的形状: {data2_conv2.shape}")
"""
data2_conv2_参数说明：
    - batch_size=10
    - channels=out_channels=10
    - height=width=(26+2×0-3+1)=24
"""

conv3_ = nn.Conv2d(in_channels=10, out_channels=16, kernel_size=5, stride=2, padding=1)
"""
conv3_参数说明：
    - in_channels=10: 输入通道数，来自上一层conv2_的输出
    - out_channels=16: 输出通道数，增加特征图的深度
    - kernel_size=5: 5x5卷积核，比3x3更大的感受野
    - stride=2: 步长为2，实现2倍下采样，减小特征图尺寸
    - padding=1: 填充1像素，保持边界信息完整性
"""
data2_conv3 = conv3_(data2_conv2)
print(f"第三层卷积层的形状: {data2_conv3.shape}")
"""
data2_conv3_参数说明：
    - batch_size=10
    - channels=out_channels=16
    - height=width=(24+2×1-5)/2+1 = 11.5 = 11（向下取整）
"""

conv4_ = nn.Conv2d(in_channels=16, out_channels=3, kernel_size=5, stride=(2, 3), padding=2)
"""
conv4_参数说明：
    - in_channels=16: 输入通道数，来自上一层conv3_的输出
    - out_channels=3: 输出通道数，对应3个类别
    - kernel_size=5: 5x5卷积核，提取特征
    - stride=(2, 3): 垂直方向步长为2，水平方向步长为3，实现2倍垂直下采样，3倍水平下采样
    - padding=2: 填充2像素，保持边界信息完整性
"""
data2_conv4 = conv4_(data2_conv3)
print(f"第四层卷积层的形状: {data2_conv4.shape}")
"""
data2_conv4_参数说明：
    - batch_size=10
    - channels=out_channels=3
    - height=(11+2×2-5)/2+1 = 6
    - width=(11+2×2-5)/3+1 = 4.333333 = 4（向下取整）
"""
