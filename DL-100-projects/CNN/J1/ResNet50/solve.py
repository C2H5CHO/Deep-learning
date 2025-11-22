import torch
import torch.nn as nn

# 1. 定义自动填充函数
def autopad(k, p=None):
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k] # 作用：计算填充大小
    return p

# 2. 定义残差块类
class IdentityBlock(nn.Module):
    def __init__(self, in_channel, kernel_size, filters):
       super(IdentityBlock, self).__init__()
       filters1, filters2, filters3 = filters # 作用：将filters列表中的元素分别赋值给filters1、filters2、filters3

       # (1) 1x1卷积层
       self.conv1 = nn.Sequential(
           nn.Conv2d(in_channels=in_channel, out_channels=filters1, kernel_size=1, stride=1, padding=0, bias=False), # 卷积层，作用：将输入的in_channel维度的数据通过1x1卷积核转换为filters1维度的数据
           nn.BatchNorm2d(filters1), # 归一化层，作用：对filters1维度的数据进行归一化处理
           nn.ReLU(inplace=True), # 激活函数层，作用：对filters1维度的数据进行非线性变换，引入非线性特征
       )
       # (2) 3x3卷积层
       self.conv2 = nn.Sequential(
           nn.Conv2d(in_channels=filters1, out_channels=filters2, kernel_size=kernel_size, stride=1, padding=autopad(kernel_size), bias=False), # 卷积层，作用：将filters1维度的数据通过kernel_sizexkernel_size卷积核转换为filters2维度的数据
           nn.BatchNorm2d(filters2), # 归一化层，作用：对filters2维度的数据进行归一化处理
           nn.ReLU(inplace=True), # 激活函数层，作用：对filters2维度的数据进行非线性变换，引入非线性特征
       )
       # (3) 1x1卷积层
       self.conv3 = nn.Sequential(
           nn.Conv2d(in_channels=filters2, out_channels=filters3, kernel_size=1, stride=1, padding=0, bias=False), # 卷积层，作用：将filters2维度的数据通过1x1卷积核转换为filters3维度的数据
           nn.BatchNorm2d(filters3), # 归一化层，作用：对filters3维度的数据进行归一化处理
       )
       # (4) 激活函数层
       self.relu = nn.ReLU(inplace=True) # 激活函数层，作用：对filters3维度的数据进行非线性变换，引入非线性特征

    def forward(self, x):
        x1 = self.conv1(x)
        x1 = self.conv2(x1)
        x1 = self.conv3(x1)
        x = x1 + x # 作用：将输入x与经过卷积层后的x1相加，实现残差连接
        x = self.relu(x)
        return x

# 3. 定义卷积块类
class ConvBlock(nn.Module):
    def __init__(self, in_channel, kernel_size, filters, stride=2):
        super(ConvBlock, self).__init__()
        filters1, filters2, filters3 = filters # 作用：将filters列表中的元素分别赋值给filters1、filters2、filters3

        # (1) 1x1卷积层
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=filters1, kernel_size=1, stride=stride, padding=0, bias=False), # 卷积层，作用：将输入的in_channel维度的数据通过1x1卷积核转换为filters1维度的数据
            nn.BatchNorm2d(filters1), # 归一化层，作用：对filters1维度的数据进行归一化处理
            nn.ReLU(inplace=True), # 激活函数层，作用：对filters1维度的数据进行非线性变换，引入非线性特征
        )
        # (2) 3x3卷积层
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=filters1, out_channels=filters2, kernel_size=kernel_size, stride=1, padding=autopad(kernel_size), bias=False), # 卷积层，作用：将filters1维度的数据通过kernel_sizexkernel_size卷积核转换为filters2维度的数据
            nn.BatchNorm2d(filters2), # 归一化层，作用：对filters2维度的数据进行归一化处理
            nn.ReLU(inplace=True), # 激活函数层，作用：对filters2维度的数据进行非线性变换，引入非线性特征
        )
        # (3) 1x1卷积层
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=filters2, out_channels=filters3, kernel_size=1, stride=1, padding=0, bias=False), # 卷积层，作用：将filters2维度的数据通过1x1卷积核转换为filters3维度的数据
            nn.BatchNorm2d(filters3), # 归一化层，作用：对filters3维度的数据进行归一化处理
        )
        # (4) 1x1卷积层（用于匹配维度）
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=filters3, kernel_size=1, stride=stride, padding=0, bias=False), # 卷积层，作用：将输入的in_channel维度的数据通过1x1卷积核转换为filters3维度的数据
            nn.BatchNorm2d(filters3), # 归一化层，作用：对filters3维度的数据进行归一化处理
        )
        # (5) 激活函数层
        self.relu = nn.ReLU(inplace=True) # 激活函数层，作用：对filters3维度的数据进行非线性变换，引入非线性特征

    def forward(self, x):
        x1 = self.conv1(x)
        x1 = self.conv2(x1)
        x1 = self.conv3(x1)
        x = self.conv4(x) + x1 # 作用：将输入x与经过卷积层后的x1相加，实现残差连接
        x = self.relu(x)
        return x


