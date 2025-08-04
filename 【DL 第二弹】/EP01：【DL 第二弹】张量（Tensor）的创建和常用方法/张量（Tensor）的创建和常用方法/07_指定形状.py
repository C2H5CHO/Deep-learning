import torch

t1 = torch.tensor([1, 2, 3])
print(f"t1: {t1}")

t1_ = torch.full_like(t1, 5)
print(f"t1_: {t1_}")

t1_1 = torch.randint_like(t1, 1, 10)
print(f"t1_1: {t1_1}")

t1_2 = torch.zeros_like(t1)
print(f"t1_2: {t1_2}")

# ×1. 转化前后的数据类型需保持一致
# t1_3 = torch.randn_like(t1)

