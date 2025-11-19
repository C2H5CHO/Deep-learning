import torch
from torch import nn
from torch.nn import functional as F

# 1. 输入数据
data = torch.ones(size=(10, 3, 227, 227))

# 2.
class Model(nn.Module):
    def __init__(self):
        super().__init__()

        # 使用较大的卷积核和较长的步长快速降低特征图的尺寸，并使用较多的通道数弥补降低尺寸所带来的数据损失
        self.conv1 = nn.Conv2d(3, 96, kernel_size=11, stride=4)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2)

        # 进一步扩大通道数提取数据特征
        self.conv2 = nn.Conv2d(96, 256, kernel_size=5, padding=2) # 使用5×5卷积核和2像素填充
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2)

        # 连续使用多个卷积层提取更复杂的特征
        # kernel_size=3, padding=1/kernel_size=5, padding=2 保持特征图的尺寸不变
        self.conv3 = nn.Conv2d(256, 384, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(384, 384, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(384, 256, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=3, stride=2)

        # 使用全连接层进行分类
        self.fc1 = nn.Linear(256 * 6 * 6, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, 1000)

    def forward(self, x):
        x = F.relu(self.conv1(x)) # 96个11×11卷积核提取特征
        x = self.pool1(x) # 3×3最大池化层，步长为2

        x = F.relu(self.conv2(x)) # 256个5×5卷积核提取特征
        x = self.pool2(x) # 3×3最大池化层，步长为2

        x = F.relu(self.conv3(x)) # 384个3×3卷积核提取特征
        x = F.relu(self.conv4(x)) # 384个3×3卷积核提取特征
        x = F.relu(self.conv5(x)) # 256个3×3卷积核提取特征
        x = self.pool3(x) # 3×3最大池化层，步长为2

        x = x.view(-1, 256 * 6 * 6) # 展平特征图

        x = F.dropout(x, p=0.5) # Dropout层，防止过拟合
        x = F.relu(F.dropout(self.fc1(x), p=0.5)) # 4096个神经元全连接层，Dropout层，防止过拟合
        x = F.relu(self.fc2(x)) # 4096个神经元全连接层，ReLU激活函数
        output = F.softmax(self.fc3(x), dim=1) # 1000个神经元全连接层，Softmax激活函数

        return output


