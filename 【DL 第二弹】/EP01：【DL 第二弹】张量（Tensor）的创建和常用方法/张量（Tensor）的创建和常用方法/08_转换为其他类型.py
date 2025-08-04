import torch
import numpy as np

# 1. 转为数组
t1 = torch.tensor([1, 2, 3])
print(f"t1类型：{t1.dtype}")
# .numpy
t1_ = t1.numpy()
print(f"t1_类型：{t1_.dtype}")
# np.array()
t1_1 = np.array(t1)
print(f"t1_1类型：{t1_1.dtype}")

# 2. 转为列表
t2 = torch.tensor([1, 2, 3])
# .tolist
t2_ = t2.tolist()
print(f"t2_：{t2_}")
# list()
t2_1 = list(t2)
print(f"t2_1：{t2_1}")
