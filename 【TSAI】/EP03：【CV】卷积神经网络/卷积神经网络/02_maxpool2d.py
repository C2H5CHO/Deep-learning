import torch
from torch import nn

# 1. 输入数据
data = torch.ones(size=(10, 3, 28, 28))

# 2. 卷积层
conv1 = nn.Conv2d(
    in_channels=3,
    out_channels=6,
    kernel_size=3
)
data_conv1 = conv1(data)
print(f"第一层卷积层的形状: {data_conv1.shape}")
conv2 = nn.Conv2d(
    in_channels=6,
    out_channels=16,
    kernel_size=3,
    stride=2,
    padding=1
)
data_conv2 = conv2(data_conv1)
print(f"第二层卷积层的形状: {data_conv2.shape}")

# 3. 池化层
pool1 = nn.MaxPool2d(
    kernel_size=2 # 池化核大小，表示2x2的池化窗口
)
data_pool1 = pool1(data_conv2)
print(f"第一层池化层的形状: {data_pool1.shape}")

max_pool1 = nn.AdaptiveMaxPool2d(7) # 表示输出特征图的大小为7x7
data_max_pool1 = max_pool1(data_conv2)
print(f"第一层最大池化层的形状: {data_max_pool1.shape}")

avg_pool1 = nn.AdaptiveAvgPool2d(output_size=(3, 7)) # 表示输出特征图的大小为3x7
data_avg_pool1 = avg_pool1(data_conv2)
print(f"第一层平均池化层的形状: {data_avg_pool1.shape}")
