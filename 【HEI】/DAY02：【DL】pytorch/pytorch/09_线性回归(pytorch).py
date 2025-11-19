# 1. 工具包
import torch
from torch.utils.data import TensorDataset # 构造数据集对象
from torch.utils.data import DataLoader # 数据加载器
from torch import nn # nn模块中有平方损失函数和假设函数
from torch import optim # optim模块中有优化器函数
from sklearn.datasets import make_regression # 创建线性回归模型数据集
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei'] # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False # 用来正常显示负号

# 2. 构建数据集
x, y, coef = make_regression(
    n_samples=100, # 样本数量
    n_features=1, # 特征数量
    noise=10, # 噪声
    bias=1.5, # 截距
    coef=True # 是否返回系数
)

plt.scatter(x, y)
plt.show()

# 3. 构建数据加载器
## 3.1 数据转换
x = torch.tensor(x)
y = torch.tensor(y)
## 3.2 构建torch数据集
dataset = TensorDataset(x, y)
## 3.3 构建batch数据
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

# 4. 模型构建
model = torch.nn.Linear(
    in_features=1, # 输入特征数量
    out_features=1 # 输出特征数量
)

# 5. 模型训练
## 5.1 定义损失函数
mse = torch.nn.MSELoss()
## 5.2 定义优化器
optimizer = torch.optim.SGD(params=model.parameters(), lr=0.01)
## 5.3 模型训练
loss_sum = []
for epoch in range(100):
    sum = 0
    sample = 0
    for x, y in dataloader:
        # 模型预测
        y_pred = model(x.type(torch.float32))
        # 计算损失
        loss = mse(y_pred, y.reshape(-1, 1).type(torch.float32))
        sum += loss.item()
        sample += len(y)
        # 梯度清零
        optimizer.zero_grad()
        # 反向传播
        loss.backward()
        # 优化器更新参数
        optimizer.step()

    loss_sum.append(sum / sample)

# 6. 模型预测
## 6.1 绘制损失变化曲线
plt.plot(range(100), loss_sum)
plt.grid()
plt.show()
## 6.2 绘制拟合直线
plt.scatter(x, y)
x = torch.linspace(x.min(), x.max(), 1000)
y1 = torch.tensor([v * model.weight + model.bias for v in x])
y2 = torch.tensor([v * coef + 1.5 for v in x])
plt.plot(x, y1, label='训练')
plt.plot(x, y2, label='真实')
plt.grid()
plt.legend()
plt.show()