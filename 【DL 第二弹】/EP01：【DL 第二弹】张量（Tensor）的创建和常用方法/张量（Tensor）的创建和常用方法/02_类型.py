import torch
import numpy as np

# 1. 通过列表创建张量
t1 = torch.tensor([1, 2])
print(f"t1类型：{t1.dtype}")

# 2. 通过元组创建张量
t2 = torch.tensor((3, 4))
print(f"t2类型：{t2.dtype}")

# 3.通过数组创建张量
arr3 = np.array((5, 6))
t3 = torch.tensor(arr3)
print(f"t3类型：{t3.dtype}")

# 4. 浮点型
print(f"{np.array([1,1, 2.2]).dtype}")
print(f"{torch.tensor(np.array([1.1, 2.2])).dtype}")
print(f"{torch.tensor([1.1, 2.2]).dtype}")

# 5. 指定类型
t5 = torch.tensor([1.1, 2.5], dtype=torch.int16)
print(f"t5：{t5}")

# 6. 复数型
t6 = torch.tensor(1+2j)
print(f"t6：{t6}，t6类型：{t6.dtype}")
