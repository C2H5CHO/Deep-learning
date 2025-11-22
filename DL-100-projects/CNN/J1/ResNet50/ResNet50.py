import torch
import torch.nn as nn
from torchsummary import summary
from solve import ConvBlock, IdentityBlock

class ResNet50(nn.Module):
    def __init__(self, classes=1000):
        super(ResNet50, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False, padding_mode='zeros'), # 卷积层，作用：将输入的3通道数据通过7x7卷积核转换为64通道的数据
            nn.BatchNorm2d(64), # 归一化层，作用：对64通道的数据进行归一化处理
            nn.ReLU(), # 激活函数层，作用：对64通道的数据进行非线性变换，引入非线性特征
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0) # 最大池化层，作用：对64通道的数据进行最大池化操作，降低特征图的尺寸
        )
        self.conv2 = nn.Sequential(
            ConvBlock(in_channel=64, kernel_size=3, filters=[64, 64, 256], stride=1),
            IdentityBlock(in_channel=256, kernel_size=3, filters=[64, 64, 256]),
            IdentityBlock(in_channel=256, kernel_size=3, filters=[64, 64, 256])
        )
        self.conv3 = nn.Sequential(
            ConvBlock(in_channel=256, kernel_size=3, filters=[128, 128, 512], stride=2),
            IdentityBlock(in_channel=512, kernel_size=3, filters=[128, 128, 512]),
            IdentityBlock(in_channel=512, kernel_size=3, filters=[128, 128, 512]),
            IdentityBlock(in_channel=512, kernel_size=3, filters=[128, 128, 512])
        )
        self.conv4 = nn.Sequential(
            ConvBlock(in_channel=512, kernel_size=3, filters=[256, 256, 1024], stride=2),
            IdentityBlock(in_channel=1024, kernel_size=3, filters=[256, 256, 1024]),
            IdentityBlock(in_channel=1024, kernel_size=3, filters=[256, 256, 1024]),
            IdentityBlock(in_channel=1024, kernel_size=3, filters=[256, 256, 1024]),
            IdentityBlock(in_channel=1024, kernel_size=3, filters=[256, 256, 1024]),
            IdentityBlock(in_channel=1024, kernel_size=3, filters=[256, 256, 1024])
        )
        self.conv5 = nn.Sequential(
            ConvBlock(in_channel=1024, kernel_size=3, filters=[512, 512, 2048], stride=2),
            IdentityBlock(in_channel=2048, kernel_size=3, filters=[512, 512, 2048]),
            IdentityBlock(in_channel=2048, kernel_size=3, filters=[512, 512, 2048])
        )
        self.pool = nn.AvgPool2d(kernel_size=7, stride=1) # 平均池化层，作用：对2048通道的数据进行平均池化操作，降低特征图的尺寸为1x1
        self.fc = nn.Linear(2048, 3) # 全连接层，作用：将2048通道的数据映射到类别数（3）

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1) # 作用：将特征图的尺寸调整为(batch_size, 2048)
        x = self.fc(x)
        return x

if __name__ == '__main__':
    model = ResNet50().to('cuda')
    print(model)
    print('--'*50)
    summary(model, (3, 224, 224))
