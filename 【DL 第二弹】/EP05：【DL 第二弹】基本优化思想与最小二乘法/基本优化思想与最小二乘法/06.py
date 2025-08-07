import torch

# 1. pytorch矩阵求解
# 定义特征矩阵X和标签向量y
X = torch.tensor([[1.0, 1.0], [3.0, 1.0]])  # 每行是(x_i, 1)
y = torch.tensor([[2.0], [4.0]])  # 形状为(2,1)

# 计算X^T * X
X_T_X = torch.mm(X.T, X)  # mm表示矩阵乘法

# 计算(X^T * X)的逆矩阵
X_T_X_inv = torch.inverse(X_T_X)

# 计算最优参数w^T = (X^T X)^{-1} X^T y
w = torch.mm(torch.mm(X_T_X_inv, X.T), y)

print("最优参数：")
print(w)

print('--'*50)
# 2. lstsq 求解最小二乘法函数
result = torch.linalg.lstsq(X, y)
print("最优参数：")
print(result.solution)