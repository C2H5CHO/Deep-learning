import torch
from torch.linalg import svd

# 1. eig 特征分解
t1 = torch.arange(1, 10).reshape(3, 3).float()
print(f"t1: {t1}")
print(f"t1的特征分解：{torch.linalg.eig(t1)}")

t2 = torch.tensor([1, 2, 2, 4]).reshape(2, 2).float()
print(f"t2: {t2}")
print(f"t2的秩：{torch.linalg.matrix_rank(t2)}")
print(f"t2的特征分解：{torch.linalg.eig(t2)}")

t3 = torch.tensor([[1, 2, 3], [2, 4, 6], [3, 6, 9]]).float()
print(f"t3的特征分解：{torch.linalg.eig(t3)}")

print('--'*50)
# 2. svd 奇异值分解
t4 = torch.tensor([[1, 2, 3], [2, 4, 6], [3, 6, 9]]).float()
print(f"t4的奇异值分解：{torch.linalg.svd(t4)}")
# 验证SVD分解
t4_U, t4_S, t4_Vh = torch.linalg.svd(t4)
print(torch.mm(torch.mm(t4_U, torch.diag(t4_S)), t4_Vh))
# 降维
t4_U1 = t4_U[:, 0].reshape(3, 1)
print(f"t4_U1：{t4_U1}")
t4_S1 = t4_S[0]
print(f"t4_S1：{t4_S1}")
t4_Vh1 = t4_Vh[0, :].reshape(1, 3)
print(f"t4_Vh1：{t4_Vh1}")
print(torch.mm((t4_U1 * t4_S1), t4_Vh1))

print('--'*50)
# 3. lstsq 最小二乘法
A = torch.tensor([[1.0, 2.0],
                  [3.0, 4.0],
                  [5.0, 6.0]])
b = torch.tensor([[7.0], [8.0], [9.0]])

result = torch.linalg.lstsq(b, A)
x_solution = result.solution
residuals = result.residuals

print("解x:\n", x_solution)
print("残差:\n", residuals)

print('--'*50)
# 4. solve 精确求解线性方程组
A = torch.tensor([[2.0, 3.0],
                  [4.0, 5.0]])
b = torch.tensor([[8.0], [14.0]])

# 求解方程 Ax = b
x = torch.linalg.solve(A, b)
print("解x:\n", x)
# 验证解：A @ x 应接近 b
print("验证 A@x:\n", torch.matmul(A, x), "\n")

print('--'*50)
# 5. lu LU分解
A = torch.tensor([[3.0, 1.0, 2.0],
                  [6.0, 3.0, 4.0],
                  [3.0, 2.0, 5.0]])

LU, pivots, info = torch.linalg.lu(A)
# 提取 L 和 U 矩阵
L = torch.tril(LU, diagonal=-1) + torch.eye(A.shape[0])  # 下三角矩阵（含单位对角线）
U = torch.triu(LU)  # 上三角矩阵

print("置换矩阵索引pivots:\n", pivots)
print("下三角矩阵L:\n", L)
print("上三角矩阵U:\n", U)
# 验证分解正确性
P = torch.eye(A.shape[0])[pivots.long()]  # 构建置换矩阵
print("验证 P@A ≈ L@U:\n", torch.matmul(P, A), "\nvs\n", torch.matmul(L, U))
