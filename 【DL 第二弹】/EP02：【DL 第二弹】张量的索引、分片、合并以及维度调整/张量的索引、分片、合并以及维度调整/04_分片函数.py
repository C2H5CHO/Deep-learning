import torch

# 1. chunk() 分块
t1 = torch.arange(12).reshape(4, 3)
print(f"t1：{t1}")
t1_ = torch.chunk(t1, 4, dim=0)
print(f"t1_ 在第0维度上进行4等分：{t1_}")

# *1. chunk() 函数返回的结果是一个视图，而非一个新对象
t1_[0][0][0] = 100
print(f"t1_：{t1_}")
print(f"t1：{t1}")
# *2. 若原张量无法均分，chunk() 函数不会报错，而是返回其他均分结果
t1_1 = torch.chunk(t1, 8, dim=0)
print(f"t1_1：{t1_1}")

print('--'*50)
# 2. split() 拆分
t2 = torch.arange(12).reshape(4, 3)
print(f"t2：{t2}")
t2_ = torch.split(t2, 2, 0)
print(f"t2_：{t2_}")
t2_1 = torch.split(t2, [1, 3], 0)
print(f"t2_1：{t2_1}")
print(f"[1, 1, 2]：{torch.split(t2, [1, 1, 2], 0)}")
t2_2 = torch.split(t2, [1, 2], 1)
print(f"t2_2：{t2_2}")

# *1. split() 函数返回的结果也是一个视图
t2_2[0][0] = 100
print(f"t2：{t2}")