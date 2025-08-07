import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 设置字体，支持中文显示
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
# 解决负号显示问题
plt.rcParams['axes.unicode_minus'] = False

# 定义a和b的取值范围
a_range = np.arange(-1, 3, 0.05)  # a从-1到3，步长0.05
b_range = np.arange(-1, 3, 0.05)  # b从-1到3，步长0.05

# 生成网格点
a_grid, b_grid = np.meshgrid(a_range, b_range)

# 计算每个(a,b)对应的SSE
SSE = (2 - a_grid - b_grid)**2 + (4 - 3*a_grid - b_grid)** 2

# 绘制3D曲面和等高线
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# 3D曲面
surf = ax.plot_surface(a_grid, b_grid, SSE, cmap='rainbow', alpha=0.8)
# 等高线（投影到z=0平面）
ax.contour(a_grid, b_grid, SSE, zdir='z', offset=0, cmap='rainbow')

ax.set_xlabel('a（斜率）')
ax.set_ylabel('b（截距）')
ax.set_zlabel('SSE（误差平方和）')
ax.set_title('SSE的3D可视化')
fig.colorbar(surf, shrink=0.5, aspect=5)  # 颜色条表示SSE大小
plt.show()