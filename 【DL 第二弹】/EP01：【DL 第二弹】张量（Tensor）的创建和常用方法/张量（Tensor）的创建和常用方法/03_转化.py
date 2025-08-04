import torch

# 1. 隐式转化
# 浮点型与整型
t1 = torch.tensor([1.1, 2, 3.7])
print(f"t1: {t1}，t1类型: {t1.dtype}")
# 布尔型与整型
t2 = torch.tensor([False, 2])
print(f"t2: {t2}，t2类型: {t2.dtype}")

# 2. 转化方法
t3 = torch.tensor([1, 2])
print(f"t3类型：{t3.dtype}")
print(f"t3转为浮点型：{t3.float()}")
print(f"t3转为双精度浮点型：{t3.double()}")
print(f"t3转为16位整型：{t3.short()}")
