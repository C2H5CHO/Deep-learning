import torch

# 1. 相同形状
t1 = torch.arange(3)
print(f"t1：{t1}")
print(f"t1+t1：{t1+t1}")

print('--'*50)
# 2. 不同形状
# 2.1 标量+任意形状的张量
t2 = torch.arange(3)
print(f"t2+1：{t2+1}")

# 2.2 相同维度、不同形状
t3 = torch.zeros(3, 4)
print(f"t3：{t3}")
t4 = torch.ones(1, 4)
print(f"t4：{t4}")
print(f"t3+t4：{t3+t4}")

# ×1. 若二者取值均不为1，则无法广播
t5 = torch.ones(2, 4)
# print(f"t3+t5：{t3+t5}") # RuntimeError: The size of tensor a (3) must match the size of tensor b (2) at non-singleton dimension 0

t6 = torch.arange(3).reshape(1, 3)
t7 = torch.arange(3).reshape(3, 1)
print(f"t6：{t6}，\nt7；{t7}，\nt6+t7：{t6+t7}")

print('--'*50)
# 3. 三维张量
t8 = torch.zeros(3, 4, 5)
t9 = torch.arange(3, 4, 1)
print(f"t8：{t8}，\nt9：{t9}，\nt8+t9：{t8+t9}")

t10 = torch.ones(1, 1, 5)
print(f"t8+t10：{t8+t10}")

print('--'*50)
# 4. 不同维度
t11 = torch.arange(4).reshape(2, 2)
t12 = torch.zeros(3, 2, 2)
# 转为3维张量
t11.reshape(1, 2, 2)
print(f"t11+t12：{t11+t12}")
