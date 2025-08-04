import torch
import numpy as np

# 1. 一维数组
t1 = torch.tensor([1, 2, 3])
print(f"t1维度：{t1.ndim}")
print(f"t1形状：{t1.shape}")
print(f"t1形状：{t1.size()}")
print(f"t1中（n-1）维元素的个数：{len(t1)}")
print(f"t1中元素的个数：{t1.numel()}")

# 2. 二维数组
t2 = torch.tensor([[1, 2, 3], [4, 5, 6]])
print(f"t2维度：{t2.ndim}")
print(f"t2形状：{t2.shape}")
print(f"t2形状：{t2.size()}")
print(f"t2中（n-1）维元素的个数：{len(t2)}")
print(f"t2中元素的个数：{t2.numel()}")

# 3. 0维张量
t3 = torch.tensor([3])
print(f"t3维度：{t3.ndim}")
print(f"t3形状：{t3.shape}")
print(f"t3形状：{t3.size()}")
print(f"t3中（n-1）维元素的个数：{len(t3)}")
print(f"t3中元素的个数：{t3.numel()}")

t0 = torch.tensor(3)
print(f"t0维度：{t0.ndim}")
print(f"t0形状：{t0.shape}")
print(f"t0形状：{t0.size()}")
print(f"t0中元素的个数：{t0.numel()}")

# 4. 高维张量
arr1 = np.array([[1, 2, 3], [4, 5, 6]])
arr2 = np.array([[7, 8, 9], [10, 11, 12]])
t4 = torch.tensor([arr1, arr2])
print(f"t4：{t4}")
print(f"t4维度：{t4.ndim}")
print(f"t4形状：{t4.shape}")
print(f"t4形状：{t4.size()}")
print(f"t4中（n-1）维元素的个数：{len(t4)}")
print(f"t4中元素的个数：{t4.numel()}")