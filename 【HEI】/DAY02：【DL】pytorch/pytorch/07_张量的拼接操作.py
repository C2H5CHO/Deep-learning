import torch

torch.random.manual_seed(41)
data1 = torch.randint(0, 10, [4, 5, 3])
print(f"data1：{data1}")

torch.random.manual_seed(42)
data2 = torch.randint(0, 10, [4, 5, 5])
print(f"data2：{data2}")

print(f"dim=2：{torch.cat([data1, data2], dim=2)}")

# 1×. 除拼接维度外，其余维度必须相同
# print(f"dim=1：{torch.cat([data1, data2], dim=1)}")
# # RuntimeError: Sizes of tensors must match except in dimension 1. Expected size 3 but got size 5 for tensor number 1 in the list.

