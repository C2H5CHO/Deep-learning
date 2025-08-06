import torch

# 1. trace 迹
t1 = torch.tensor([[1, 2], [4, 5]]).float()
print(f"t1的迹：{torch.trace(t1)}")

# *1. 计算过程不需要是方阵
t2 = torch.arange(1, 7).reshape(2 ,3)
print(f"t2的迹：{torch.trace(t2)}")

print('--'*50)
# 2. rank 秩
t3 = torch.arange(1, 5).reshape(2, 2).float()
print(f"t3的秩：{torch.linalg.matrix_rank(t3)}")

print('--'*50)
# 3. det 行列式
t4 = torch.tensor([[1, 2], [4, 5]]).float()
print(f"t4的行列式：{torch.det(t4)}")

# ×1. 要求2维张量必须是矩阵
t5 = torch.arange(1, 13).reshape(3, 4)
# print(f"t5的行列式：{torch.det(t5)}") # RuntimeError: linalg.det: A must be batches of square matrices, but they are 3 by 4 matrices

print('--'*50)
# 4. 矩阵表达式
A = torch.arange(1, 5).reshape(2, 2).float()
print(f"A：{A}")
# 4.1 二维空间的点
import matplotlib as mpl
import matplotlib.pyplot as plt

plt.plot(A[:,0], A[:,1], 'o')
# plt.show()
# 4.2 线性回归方程的拟合线
# （1）创建浮点型矩阵
B = torch.tensor([[1.0, 1], [3, 1]])
C = torch.tensor([2.0, 4])
# （2）求解逆矩阵
B_ = torch.inverse(B)
# （3）探讨逆矩阵的基本特性
print(f"B_*B：{torch.mm(B_, B)}")
print(f"B*B_：{torch.mm(B, B_)}")
# （4）乘以逆矩阵，求解线性方程
print(torch.mv(B_, C))

