import torch

t1 = torch.arange(1, 11)
print(f"t1：{t1}")

indices = torch.tensor([1, 2])
t1_ = torch.index_select(t1, 0, indices)
print(f"t1_：{t1_}")

print('--'*50)

t2 = torch.arange(12).reshape(4, 3)
print(f"t2：{t2}")
t2_ = torch.index_select(t2, 0, indices)
t2_1 = torch.index_select(t2, 1, indices)
print(f"t2_：{t2_}")
print(f"t2_1：{t2_1}")