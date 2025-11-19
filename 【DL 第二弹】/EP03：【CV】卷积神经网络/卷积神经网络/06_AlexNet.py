import torch
from torch import nn
from torch.nn import functional as F

# 1. 输入数据
data = torch.ones(size=(10, 3, 227, 227))

# 2.
class Model(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 39, kernel_size=11, stride=4) # 使用11×11卷积核和4步长快速降低特征图的尺寸
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.conv2 = nn.Conv2d(96, 256, kernel_size=5, padding=2) # 使用5×5卷积核和2像素填充
        self.pool2 = nn.MaxPool2d()

        self.conv3 = nn.Conv2d()
        self.conv4 = nn.Conv2d()
        self.conv5 = nn.Conv2d()
        self.pool3 = nn.MaxPool2d()

        self.fc1 = nn.Linear()
        self.fc2 = nn.Linear()
        self.fc3 = nn.Linear()

