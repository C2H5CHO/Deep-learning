import torch
import numpy as np

# 1. 一维张量的索引
t1 = torch.arange(1, 11)
print(f"t1：{t1}")

# 从左到右，从零开始
print(f"t1[0]：{t1[0]}")
print(f"t1[0]的类型：{type(t1[0])}")
# 切片
print(f"t1[1:8] 取2-9号元素，且左闭右开：{t1[1:8]}")
print(f"t1[1:8:2] 取2-9号元素，左闭右开，且隔2取一个数：{t1[1:8:2]}")
print(f"t1[1::2] 从2号开始取完，且隔2取一个数：{t1[1::2]}")
print(f"t1[:8:2]：{t1[:8:2]}")

# ×1：step位必须大于0
# print(f"t1[9:1:-1]：{t1[9:1:-1]}") # ValueError: step must be greater than zero

print('--'*50)
# 2. 二维张量的索引
t2 = torch.arange(1, 10).reshape(3, 3)
print(f"t2：{t2}")
print(f"t2[0, 1] 第1行第2列：{t2[0, 1]}")
print(f"t2[0, ::2] 第1行，且隔2取一个数：{t2[0, ::2]}")
print(f"t2[0, [0, 2]] 第1行的第1列和第3列：{t2[0, [0, 2]]}")
print(f"t2[::2, ::2]：{t2[::2, ::2]}")
print(f"t2[[0, 2], 0]：{t2[0, [0, 2]]}")

print('--'*50)
# 3. 三维张量的索引
t3 = torch.arange(1, 28).reshape(3, 3, 3)
print(f"t3：{t3}")
print(f"t3[1, 1, 1] 第2个矩阵第2行第2列：{t3[1, 1, 1]}")
print(f"t3[1, ::2, ::2]：{t3[1, ::2, ::2]}")
