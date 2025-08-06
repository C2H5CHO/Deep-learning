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