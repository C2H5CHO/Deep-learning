import torch
import matplotlib.pyplot as plt

lr = 0.1 # 学习率

# 1. 创建训练数据集
x = torch.rand(20, 1) * 10 # 20个随机数
y = 2 * x + (5 + torch.rand(20, 1)) # y = 2x + 5 + 噪声

# 2. 构建线性回归参数
w = torch.rand((1), requires_grad=True) # 权重
b = torch.rand((1), requires_grad=True) # 偏置

for iteration in range(1000):
    # （1）前向传播
    wx = torch.add(w, x) # wx = w * x
    y_pred = torch.add(wx, b) # y_pred = wx + b

    # （2）计算损失函数
    loss = (0.5 * (y_pred - y) ** 2).mean() # 均方误差损失函数

    # （3）反向传播
    loss.backward()

    # （4）更新参数
    b.data.sub_(lr * b.grad) # 更新偏置
    w.data.sub_(lr * w.grad) # 更新权重

    # （5）绘图
    if iteration % 20 == 0:
        plt.scatter(x.data.numpy(), y.data.numpy()) # 绘制训练数据
        plt.plot(x.data.numpy(), y_pred.data.numpy(), color='red', lw=5) # 绘制预测结果
        plt.text(2, 20, 'Loss:%.4f' % loss.data.numpy(), fontdict={'size': 20, 'color': 'red'}) # 显示损失
        plt.xlim(1.5, 10) # 设置x轴范围
        plt.ylim(8, 28) # 设置y轴范围
        plt.title(f"Iteration:{iteration}\nw:{w.data.numpy()}\nb:{b.data.numpy()}") # 设置标题
        plt.pause(0.5) # 暂停0.5秒

        if loss.data.numpy() < 1:
            break