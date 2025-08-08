import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

x = np.arange(-1, 3, 0.05)
y = np.arange(-1, 3, 0.05)
a, b = np.meshgrid(x, y)
SSE = (2 - a - b)**2 + (4 - 3*a - b)**2

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.plot_surface(a, b, SSE, cmap='rainbow')
ax.contour(a, b, SSE, zdir='z', offset=0, cmap='rainbow')
plt.show()

# step 1：初始随机点
x1 = torch.tensor(0., requires_grad=True)
y1 = torch.tensor(0., requires_grad=True)
# step 2：计算梯度
S1 = torch.pow((2 - x1 - y1), 2) + torch.pow((4 - 3*x1 - y1), 2)
# step 3：反向传播
S1.backward()
# step 4：打印结果
print(f"x1的导数：{x1.grad}")
print(f"y1的导数：{y1.grad}")
# step 5：确定原点移动方向
x_ = np.arange(-30, 30, 0.1)
y_ = (12/28)*x_
plt.plot(x_, y_, '-')
plt.plot(0, 0, 'ro')
plt.plot(x1.grad.item(), y1.grad.item(), 'go')
plt.show()
# step 6：移动
x1_ = x1 - 0.01*x1.grad
y1_ = y1 - 0.01*y1.grad
x1_.retain_grad()
y1_.retain_grad()
S1_ = torch.pow((2 - x1_ - y1_), 2) + torch.pow((4 - 3*x1_ - y1_), 2)
S1_.backward()
print(f"x1_的导数：{x1_.grad}")
print(f"y1_的导数：{y1_.grad}")

