import torch

# 1. cat() 拼接
a = torch.zeros(2, 3)
b = torch.ones(2, 3)
c = torch.zeros(3, 3)
print(f"a+b 按行：{torch.cat([a,b])}")
print(f"a+b 按列：{torch.cat([a,b], dim=1)}")

print('--'*50)
# 2. stack() 堆叠
a = torch.zeros(2, 3)
b = torch.ones(2, 3)
c = torch.zeros(3, 3)
print(f"a+b 堆到一个三维张量中：{torch.stack([a, b])}")

# ×1. 堆叠时，必须保证两个张量的形状一致
print(f"a+c 拼接：{torch.cat([a, c])}")
# print(f"a+c 堆叠：{torch.stack([a, c])}") # RuntimeError: stack expects each tensor to be equal size, but got [2, 3] at entry 0 and [3, 3] at entry 1
