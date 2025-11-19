import torch
from torch import nn
from torch.nn import functional as F
from torchinfo import summary

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5)
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
        self.pool2 = nn.AvgPool2d(kernel_size=2)
        self.fc1 = nn.Linear(5*5*16, 120) # 全连接层，表示输入5*5*16，输出120
        self.fc2 = nn.Linear(120, 84) # 全连接层，表示输入120，输出84

    def forward(self, x):
        x = F.tanh(self.conv1(x))
        x = self.pool1(x)
        x = F.tanh(self.conv2(x))
        x = self.pool2(x)
        x = x.view(-1, 5*5*16) # Flatten展平，表示将x的第0维（batch_size）保持不变，将第1维（通道数）展平为5*5*16
        x = F.tanh(self.fc1(x))
        output = F.softmax(self.fc2(x), dim=1)
        return output

if __name__ == '__main__':
    data = torch.ones(size=(10, 1, 32, 32))

    lenet5 = Model()
    output = lenet5(data)
    summary(lenet5, input_size=(10, 1, 32, 32)) # 输出模型结构

