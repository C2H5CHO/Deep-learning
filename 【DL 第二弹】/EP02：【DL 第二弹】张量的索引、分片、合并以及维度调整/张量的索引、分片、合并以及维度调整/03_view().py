import torch

t1 = torch.arange(6).reshape(2, 3)
print(f"t1: {t1}")
t1_ = t1.view(3, 2)
print(f"t1_: {t1_}")
t1_1 = t1.view(1, 2, 3)
print(f"t1_1: {t1_1}")

print('--'*50)
# *1. view() 构建的是一个数据相同，但形状不同的视图
t1[0] = 100
print(f"t1: {t1}")
print(f"t1_: {t1_}")