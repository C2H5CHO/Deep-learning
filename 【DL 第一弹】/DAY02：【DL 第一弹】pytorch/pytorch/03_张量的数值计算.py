import torch
import numpy as np

# 1. 基本运算
data1 = torch.randint(0, 10, [2, 3])
print(f"data1：{data1}")
## 1.1 add/add_ 矩阵相加
data1_add = data1.add(10)
print(f"data1_add：{data1_add}")
print(f"data1：{data1}")

data1_add_ = data1.add_(100)
print(f"data1_add_：{data1_add_}")
print(f"data1：{data1}")
## 1.2 sub/sub_ 矩阵相减
data1_sub = data1.sub(10)
print(f"data1_sub：{data1_sub}")
## 1.3 mul/mul_ 矩阵相乘
data1_mul = data1.mul(100)
print(f"data1_mul：{data1_mul}")
## 1.4 div/div_ 矩阵相除
data1_div = data1.div(10)
print(f"data1_div：{data1_div}")
## 1.5 neg/neg_ 矩阵取负
data1_neg = data1.neg()
print(f"data1_neg：{data1_neg}")

print('--'*50)
# 2. 点乘运算
torch.random.manual_seed(42)
data2 = torch.randint(0, 10, [3, 4])
print(f"data2：{data2}")

torch.random.manual_seed(43)
data3 = torch.randint(0, 10, [3, 4])
print(f"data3：{data3}")
print(f"data2*data3：{torch.mul(data2, data3)}")
print(f"data2*data3：{data2*data3}")

# 1×. 维度不匹配
# torch.random.manual_seed(44)
# data4 = torch.randint(0, 10, [4, 4])
# print(f"data4：{data4}")
# print(f"data2*data4：{torch.mul(data2, data4)}")
# # RuntimeError: The size of tensor a (3) must match the size of tensor b (4) at non-singleton dimension 0

print('--'*50)
# 3. 矩阵乘法
torch.random.manual_seed(42)
data5 = torch.randint(0, 10, [3, 4])
print(f"data5：{data5}")

torch.random.manual_seed(43)
data6 = torch.randint(0, 10, [4, 5])
print(f"data6：{data6}")
print(f"data5@data6：{torch.matmul(data5, data6)}")
print(f"data5@data6：{data5@data6}")

