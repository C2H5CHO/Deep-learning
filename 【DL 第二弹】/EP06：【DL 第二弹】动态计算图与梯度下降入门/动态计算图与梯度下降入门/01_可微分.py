import torch

# 1. requires_grad 可微分性
# 构建可微分张量
t1 = torch.tensor(1., requires_grad=True)
print(f't1：{t1}')

# 构建函数关系
y1 = t1**2
z1 = y1 + 1

# 2. grad_fn 存储微分函数
print(f"t1的微分函数：{t1.grad_fn}")
print(f"y1的微分函数：{y1.grad_fn}")
print(f"z1的微分函数：{z1.grad_fn}")

