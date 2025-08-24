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

# 2. 点乘运算
data2 = torch.tensor([[1, 2], [3, 4]])
