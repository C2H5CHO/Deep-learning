import torch

# 1. reshape
data1 = torch.tensor([[10, 20, 30], [40, 50, 60]])
print(f"data1：{data1}")

print(f"data1.shape：{data1.shape}")
print(f"data1.shape[0]：{data1.shape[0]}")
print(f"data1.shape[1]：{data1.shape[1]}")
print(f"data1.size：{data1.size()}")

data1_new = data1.reshape(3, 2)
print(f"data1_new：{data1_new}")
data1_new_1 = data1.reshape(-1)
print(f"data1_new_1：{data1_new_1}")

print('--'*50)
# 2. squeeze/unsqueeze
torch.random.manual_seed(42)
data2 = torch.randint(0, 10, [3, 4, 5])
print(f"data2：{data2}")

data2_0 = data2.unsqueeze(dim=0)
print(f"data2_0：{data2_0}")
print(f"data2_0.shape：{data2_0.shape}")
data2_1 = data2.unsqueeze(dim=1)
print(f"data2_1：{data2_1}")
print(f"data2_1.shape：{data2_1.shape}")
data2_2 = data2.unsqueeze(dim=-1)
print(f"data2_2：{data2_2}")
print(f"data2_2.shape：{data2_2.shape}")

data2_2_ = data2_2.squeeze()
print(f"data2_2_：{data2_2_}")
print(f"data2_2_.shape：{data2_2_.shape}")

print('--'*50)
# 3. transpose/permute
