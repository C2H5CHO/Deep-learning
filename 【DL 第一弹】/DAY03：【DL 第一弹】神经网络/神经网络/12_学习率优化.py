import torch
import matplotlib.pyplot as plt

# 显示中文和负号
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 1. 等间隔
# 参数初始化
lr = 0.1
iter = 100
epoches = 200

# 网络数据初始化
x = torch.tensor([1.0])
w = torch.tensor([1.0], requires_grad=True)
y = torch.tensor([1.0])

# 优化器
optimizer = torch.optim.SGD([w], lr=lr, momentum=0.9)

# 学习率策略
LR_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.8)

# 遍历轮次
epoches_list = []
lr_list = []
for epoch in range(epoches):
    lr_list.append(LR_scheduler.get_last_lr())
    epoches_list.append(epoch)

    # 遍历batch
    for i in range(iter):

        # 计算损失
        loss = ((w*x - y)**2) * 0.5

        # 更新参数
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # 更新lr
    LR_scheduler.step()

# 绘制结果
plt.plot(epoches_list, lr_list)
plt.title('等间隔')
plt.grid()
plt.show()

print('--'*50)
# 2. 指定间隔
# 参数初始化
lr = 0.1
iter = 100
epoches = 200

# 网络数据初始化
x = torch.tensor([1.0])
w = torch.tensor([1.0], requires_grad=True)
y = torch.tensor([1.0])

# 优化器
optimizer = torch.optim.SGD([w], lr=lr, momentum=0.9)

# 学习率策略
LR_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 20, 40, 60, 90, 120, 160], gamma=0.8)

# 遍历轮次
epoches_list = []
lr_list = []
for epoch in range(epoches):
    lr_list.append(LR_scheduler.get_last_lr())
    epoches_list.append(epoch)

    # 遍历batch
    for i in range(iter):

        # 计算损失
        loss = ((w*x - y)**2) * 0.5

        # 更新参数
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # 更新lr
    LR_scheduler.step()

# 绘制结果
plt.plot(epoches_list, lr_list)
plt.title('指定间隔')
plt.grid()
plt.show()

print('--'*50)
# 3. 指数
# 参数初始化
lr = 0.1
iter = 100
epoches = 200

# 网络数据初始化
x = torch.tensor([1.0])
w = torch.tensor([1.0], requires_grad=True)
y = torch.tensor([1.0])

# 优化器
optimizer = torch.optim.SGD([w], lr=lr, momentum=0.9)

# 学习率策略
LR_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.8)

# 遍历轮次
epoches_list = []
lr_list = []
for epoch in range(epoches):
    lr_list.append(LR_scheduler.get_last_lr())
    epoches_list.append(epoch)

    # 遍历batch
    for i in range(iter):

        # 计算损失
        loss = ((w*x - y)**2) * 0.5

        # 更新参数
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # 更新lr
    LR_scheduler.step()

# 绘制结果
plt.plot(epoches_list, lr_list)
plt.title('指数')
plt.grid()
plt.show()

