import torch
import numpy as np

# 1. 张量转换为Numpy数组
## 1.1 共享内存
torch.random.manual_seed(42)
data1 = torch.randint(0, 10, [2, 3])
print(f"data1：{data1}")
print(f"data1的类型：{type(data1)}")
data1_numpy = data1.numpy()
print(f"data1_numpy：{data1_numpy}")
print(f"data1_numpy的类型：{type(data1_numpy)}")

data1_numpy[0] = 100
print(f"data1_numpy：{data1_numpy}")
print(f"data1：{data1}")
## 1.2 不共享内存
data2 = torch.tensor([2, 3, 4])
data2_numpy = np.array(data2).copy()

data2_numpy[0] = 100
print(f"data2_numpy：{data2_numpy}")
print(f"data2：{data2}")

print('--'*50)
# 2. Numpy数组转换为张量
## 2.1 torch.from_numpy 共享内存
data3 = np.array([2, 3, 4])
data3_torch = torch.from_numpy(data3)
print(f"data3：{data3}")
print(f"data3_torch：{data3_torch}")

data3_torch[0] = 100
print(f"data3_torch：{data3_torch}")
print(f"data3：{data3}")
## 2.2 torch.tensor 不共享内存
data4 = np.array([2, 3, 4])
data4_torch = torch.tensor(data4)

data4_torch[0] = 100
print(f"data4_torch：{data4_torch}")
print(f"data4：{data4}")

print('--'*50)
# 3. 标量张量和数字转换
data5 = torch.tensor(5)
print(f"data5：{data5.item()}")
data5_ = torch.tensor([5, ])
print(f"data5_：{data5_.item()}")