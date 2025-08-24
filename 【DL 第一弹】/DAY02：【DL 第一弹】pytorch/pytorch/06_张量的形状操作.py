import torch
import numpy as np

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
torch.random.manual_seed(43)
data3 = torch.tensor(np.random.randint(0, 10, [4, 2, 3, 5]))
print(f"data3：{data3}")
print(f"data3.shape：{data3.shape}")
## 3.1 transpose
data3_t1 = torch.transpose(data3, 0, 2)
print(f"data3_t：{data3_t1}")
print(f"data3_t.shape：{data3_t1.shape}")
data3_t2 = data3_t1.transpose(1, 2)
data3_t3 = data3_t2.transpose(2, 3)
print(f"data3_t3：{data3_t3}")
print(f"data3_t3.shape：{data3_t3.shape}")
## 3.2 permute
data3_p1 = data3.permute(2, 0, 3, 1)
print(f"data3_p1：{data3_p1}")
print(f"data3_p1.shape：{data3_p1.shape}")

print('--'*50)
# 4. view/contiguous
torch.random.manual_seed(44)
data4 = torch.randint(0, 10, [2, 3])
print(f"data4：{data4}")
print(f"判断data4是否连续：{data4.is_contiguous()}")
print(f"修改data4的形状为(-1)：{data4.view(-1)}")

data4_t1 = data4.transpose(0, 1)
print(f"判断data4_t1是否连续：{data4_t1.is_contiguous()}")
data4_t1_c = data4_t1.contiguous()
print(f"判断data4_t1_c是否连续：{data4_t1_c.is_contiguous()}")
print(f"修改data4_t1_c的形状为(-1)：{data4_t1_c.view(-1)}")

