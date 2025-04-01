import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
torch.manual_seed(10)

# 1. 生成数据
sample_nums = 100 # 样本数量
mean_value = 1.7 # 均值
bias = 1 # 偏置
n_data = torch.ones(sample_nums, 2) # 样本数据
x0 = torch.normal(mean_value*n_data, 1) + bias # 类别0的样本数据
y0 = torch.zeros(sample_nums) # 类别0的标签
x1 = torch.normal(-mean_value*n_data, 1) + bias # 类别1的样本数据
y1 = torch.ones(sample_nums) # 类别1的标签
train_x = torch.cat((x0, x1), 0) # 训练数据
train_y = torch.cat((y0, y1), 0) # 训练标签

# 2. 选择模型
class LR(nn.Module):
    def __init__(self):
        super(LR, self).__init__()
        self.features = nn.Linear(2, 1) # 特征提取层
        self.sigmoid = nn.Sigmoid() # 激活函数

    def forward(self, x):
        x = self.features(x)
        x = self.sigmoid(x) # 激活函数
        return x
    
lr_net = LR() # 实例化模型

# 3. 选择损失函数
loss_fn = nn.BCELoss() # 二分类交叉熵损失函数

# 4. 选择优化器
optimizer = torch.optim.SGD(lr_net.parameters(), lr=0.01, momentum=0.9) # 随机梯度下降优化器

# 5. 训练模型
for iteration in range(100):
    # （1）前向传播
    y_pred = lr_net(train_x) # 预测值

    # （2）计算损失
    loss = loss_fn(y_pred.squeeze(), train_y)

    # （3）反向传播
    loss.backward()


    # （4）更新参数
    optimizer.step()

    # （5）绘图
    if iteration % 20 == 0:
        mask = y_pred.ge(0.5).float().squeeze() # 预测值大于0.5的样本
        correct = (mask == train_y).sum() # 正确分类的样本数量
        accuracy = correct.item() / train_y.size(0) # 准确率

        plt.scatter(x0.data.numpy()[:, 0], x0.data.numpy()[:, 1], c='r', label='class 0')
        plt.scatter(x1.data.numpy()[:, 0], x1.data.numpy()[:, 1], c='b', label='class 1')

        w0, w1 = lr_net.features.weight[0]
        w0, w1 = float(w0.item()), float(w1.item())
        plot_b = float(lr_net.features.bias.item())
        plot_x = np.arange(-6, 6, 0.1)
        plot_y = (-w0 * plot_x - plot_b) / w1

        plt.xlim(-5, 7)
        plt.ylim(-7, 7)
        plt.plot(plot_x, plot_y)

        plt.text(-5, 5, f"Loss={loss.data.numpy():.4f}", fontdict={'size': 16, 'color': 'red'})
        plt.title(f"Iteration:{iteration}\nw0:{w0:.2f}, w1:{w1:.2f}, b:{plot_b:.2f}\naccuracy={accuracy:.2f}")
        plt.legend()

        plt.show()
        plt.pause(0.5)

        if accuracy > 0.99:
            break
