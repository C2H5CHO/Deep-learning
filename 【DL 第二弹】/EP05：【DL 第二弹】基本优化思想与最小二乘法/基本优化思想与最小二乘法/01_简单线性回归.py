import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import numpy as np

# 1. 简单线性回归的建模问题
A = torch.arange(1, 5).reshape(2, 2).float()

import matplotlib as mpl
import matplotlib.pyplot as plt
plt.plot(A[:, 0], A[:, 1], 'o')
plt.show()

# 2. 最优化问题
# 2.1 函数的3维空间
x = np.arange(-1, 3, 0.05)
y = np.arange(-1, 3, 0.05)
a, b = np.meshgrid(x, y)
SSE = (2 - a - b)**2 + (4 - 3*a - b)**2

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.plot_surface(a, b, SSE, cmap='rainbow')
ax.contour(a, b, SSE, zdir='z', offset=0, cmap='rainbow')
plt.show()

# 2.2 函数的凹凸性
x1 = np.arange(-10, 10, 0.01)
y1 = x1**2
plt.plot(x1, y1, '-')
plt.show()
