import torch

# 1. dot/vdot 点积运算
t1 = torch.arange(1, 4)
print(f"t1：{t1}")
print(f"t1*t1：{torch.dot(t1, t1)}")
print(f"t1*t1：{torch.vdot(t1, t1)}")

# ×1. 不能进行一维张量以外的运算
t2 = torch.arange(1, 5).reshape(2, 2)
# print(f"t2*t2：{torch.dot(t2, t2)}") # RuntimeError: 1D tensors expected, but got 2D and 2D tensors

print('--'*50)
# 2. mm 矩阵乘法
t3 = torch.arange(1, 7).reshape(2, 3)
t4 = torch.arange(1, 10).reshape(3, 3)
print(f"t3*t3：{t3 * t3}")
print(f"t3*t4：{torch.mm(t3, t4)}")

print('--'*50)
# 3. mv 矩阵和向量相乘
t5 = torch.arange(1, 7).reshape(2, 3)
t6 = torch.arange(1, 4)
print(f"t5*t6：{torch.mv(t5, t6)}")
print(f"t5*t6（列向量）：{torch.mv(t5, t6.reshape(3, ))}")

print('--'*50)
# 4. bmm 批量矩阵相乘
t7 = torch.arange(1, 13).reshape(3, 2, 2)
t8 = torch.arange(1, 19).reshape(3, 2, 3)
print(f"t7*t8：{torch.bmm(t7, t8)}")

print('--'*50)
# 5. addmm 矩阵相乘后相加
t9 = torch.arange(1, 7).reshape(2, 3)
t10 = torch.arange(1, 10).reshape(3, 3)
t11 = torch.arange(3)
print(f"mm：{torch.mm(t9, t10)}")
print(f"addmm：{torch.addmm(t11, t9, t10)}")
print(f"addmm，beta=0，alpha=10：{torch.addmm(t11, t9, t10, beta=0, alpha=10)}")

print('--'*50)
# 6. addbmm 批量相乘后相加
t12 = torch.arange(6).reshape(2 ,3)
t13 = torch.arange(1, 13).reshape(3, 2, 2)
t14 = torch.arange(1, 19).reshape(3, 2, 3)
print(f"bmm：{torch.bmm(t13, t14)}")
print(f"addbmm：{torch.addbmm(t12, t13, t14)}")
