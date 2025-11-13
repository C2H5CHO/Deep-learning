import torch
from torch import nn

# 1. 输入数据
data = torch.ones(size=(10, 1, 28, 28))

# 2. 卷积层
conv1 = nn.Conv2d(
    in_channels=1,
    out_channels=32,
    kernel_size=5,
    padding=2
)
data_conv1 = conv1(data)
print(f"第一层卷积层的形状: {data_conv1.shape}")

# 3. Dropout层
dropout1 = nn.Dropout2d(p=0.5) # 表示每个特征图上的每个像素有0.5的概率被置为0
data_dropout1 = dropout1(data_conv1)
print(f"第一层Dropout层的形状: {data_dropout1.shape}")

