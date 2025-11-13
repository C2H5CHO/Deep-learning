import torch
from torch import nn

# 1. 输入数据
data = torch.ones(size=(10, 3, 28, 28))

# 2. 卷积层
conv1 = nn.Conv2d(
    in_channels=3,
    out_channels=32,
    kernel_size=5,
    padding=2
)
data_conv1 = conv1(data)
print(f"第一层卷积层的形状: {data_conv1.shape}")

# 3. BN层
bn1 = nn.BatchNorm2d(32) # 表示对32个特征图进行归一化
data_bn1 = bn1(data_conv1)
print(f"第一层BN层的形状: {data_bn1.shape}")
