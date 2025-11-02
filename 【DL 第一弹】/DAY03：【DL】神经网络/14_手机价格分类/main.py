# 1. 工具包
import torch
from torchsummary import summary
from torch.utils.data import TensorDataset # 导入PyTorch的TensorDataset类，用于将数据封装成数据集
from torch.utils.data import DataLoader    # 导入PyTorch的DataLoader类，用于批量加载数据
import torch.nn as nn                      # 导入PyTorch的神经网络模块
import torch.optim as optim                # 导入PyTorch的优化器模块
from sklearn.datasets import make_regression # 导入sklearn生成回归数据集的函数
from sklearn.model_selection import train_test_split # 导入sklearn的数据集分割函数
import matplotlib.pyplot as plt            # 导入matplotlib绘图库
import numpy as np                         # 导入numpy数值计算库
import pandas as pd                        # 导入pandas数据分析库
import time                                # 导入时间处理模块

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 2. 获取数据
## 2.1 读取数据
data = pd.read_csv('data\手机价格预测.csv')
x = data.iloc[:, :-1]
y = data.iloc[:, -1]
# print(x)
## 2.2 分割数据
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
# print(x_train)
## 2.3 转换数据
x_train = torch.tensor(x_train.values, dtype=torch.float32)
x_test = torch.tensor(x_test.values, dtype=torch.float32)
y_train = torch.tensor(y_train.values, dtype=torch.int64)
y_test = torch.tensor(y_test.values, dtype=torch.int64)
# print(x_train)
## 2.4 封装tensor
train_dataset = TensorDataset(x_train, y_train)
test_dataset = TensorDataset(x_test, y_test)
# print(train_dataset)
## 2.5 创建数据加载器
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)
# print(train_loader)

# 3. 构建模型
class model(nn.Module):
    # __init__ 初始化
    def __init__(self):
        super(model, self).__init__()
        self.layer1 = nn.Linear(in_features=20, out_features=64)
        self.layer2 = nn.Linear(in_features=64, out_features=128)
        self.layer3 = nn.Linear(in_features=128, out_features=4)
        self.dropout = nn.Dropout(p=0.9)

    # forward 前向传播
    def forward(self, x):
        x = self.layer1(x)
        x = self.dropout(x)
        x = torch.relu(x)
        x = self.layer2(x)
        x = self.dropout(x)
        x = torch.relu(x)
        out = self.layer3(x)

        return out

# 4. 训练模型
def train():
    phone_model = model()
    # 损失函数
    criterion = nn.CrossEntropyLoss()
    # 优化器
    optimizer = optim.SGD(phone_model.parameters(), lr=0.1)
    # 遍历epoch
    epoches = 20
    for epoch in range(epoches):
        loss_sum = 0
        samples = 0
        for x, y in train_loader:
            # 前向传播
            y_pred = phone_model(x)
            # 计算损失
            loss = criterion(y_pred, y)
            # 梯度清零
            optimizer.zero_grad()
            # 反向传播
            loss.backward()
            # 参数更新
            optimizer.step()

            loss_sum += loss.item()
            samples += 1
        print(f"平均损失：{loss_sum/samples*1.0}")
    # 保存模型
    torch.save(phone_model.state_dict(), 'model/phone_model.pth')

# 5. 测试模型
def test():
    my_model = model()
    my_model.load_state_dict(torch.load('model/phone_model.pth'))

    correct_sum = 0
    for x, y in test_loader:
        y_pred = my_model(x)
        y_index = torch.argmax(y_pred, dim=1)
        correct_sum += (y_index == y).sum()

    acc = correct_sum.item() / len(test_dataset)
    print(f"准确率：{acc*100.0}%")

# 实例化
if __name__ == '__main__':
    my_model = model()
    summary(my_model, input_size=(20,), batch_size=10, device='cpu')
    # train()
    test()

