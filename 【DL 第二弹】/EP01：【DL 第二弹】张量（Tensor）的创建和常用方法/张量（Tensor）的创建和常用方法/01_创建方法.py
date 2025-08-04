import torch
import numpy as np

# 1. 通过列表创建张量
t1 = torch.tensor([1, 2])
print(f"t1：{t1}")

# 2. 通过元组创建张量
t2 = torch.tensor((3, 4))
print(f"t2：{t2}")

# 3.通过数组创建张量
arr1 = np.array((5, 6))
t3 = torch.tensor(arr1)
print(f"t3：{t3}")
