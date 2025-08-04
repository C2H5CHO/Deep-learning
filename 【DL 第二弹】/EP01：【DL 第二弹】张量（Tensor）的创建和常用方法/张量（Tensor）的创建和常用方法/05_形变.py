import torch
import numpy as np

# 1. flatten 拉平
t1 = torch.tensor([[1, 2, 3], [4, 5, 6]])
print(f"t1拉平：{t1.flatten()}")

arr1 = np.array([[1, 2, 3], [4, 5, 6]])
arr2 = np.array([[7, 8, 9], [10, 11, 12]])
t2 = torch.tensor([arr1, arr2])
print(f"t2拉平：{t2.flatten()}")

t0 = torch.tensor(1)
print(f"t0拉平：{t0.flatten()}")

# 2. reshape 任意变形
arr1 = np.array([[1, 2, 3], [4, 5, 6]])
arr2 = np.array([[7, 8, 9], [10, 11, 12]])
t3 = torch.tensor([arr1, arr2])
print(f"t3转为2行6列：{t3.reshape(2, 6)}")
print(f"t3转为1行12列：{t3.reshape(12)}")
print(f"t3转为1行12列：{t3.reshape(12,)}")
print(f"t3转为(1, 2, 6)：{t3.reshape(1, 2, 6)}")
