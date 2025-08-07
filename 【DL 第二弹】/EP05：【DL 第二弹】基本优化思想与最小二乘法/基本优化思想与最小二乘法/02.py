import torch
import matplotlib.pyplot as plt

# 定义数据点：x坐标为[1,3]，y坐标为[2,4]
A = torch.arange(1, 5).reshape(2, 2).float()

# 绘制散点图
plt.figure(figsize=(6, 8))
plt.plot(A[:, 0], A[:, 1], 'o', color='blue', markersize=8)  # A[:,0]取x坐标，A[:,1]取y坐标
plt.xlabel('x')
plt.ylabel('y')
plt.grid(True)
plt.show()