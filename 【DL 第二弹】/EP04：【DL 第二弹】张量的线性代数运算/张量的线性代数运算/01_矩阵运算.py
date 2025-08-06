import torch

# 1. 转置
t1 = torch.arange(1, 7).reshape(2 ,3).float()
print(f"t1：{t1}")
print(f"t1 转置：{torch.t(t1)}")
print(f"t1 转置：{t1.t()}")

print('--'*50)
# 2. 单位矩阵
t2 = torch.eye(3)
print(f"t2：{t2}")

print('--'*50)
# 3. 对角矩阵
t3 = torch.tensor([1, 2, 3, 4, 5])
print(f"t3：{t3}")
print(f"t3 对角：{torch.diag(t3)}")
print(f"t3 对角向上偏移一位：{torch.diag(t3, 1)}")
print(f"t3 对角向下偏移一位：{torch.diag(t3, -1)}")

t4 = torch.arange(9).reshape(3, 3)
print(f"t4：{t4}")
print(f"t4 上三角：{torch.triu(t4)}")
print(f"t4 上三角向上偏移一位：{torch.triu(t4, 1)}")
print(f"t4 下三角：{torch.tril(t4)}")

