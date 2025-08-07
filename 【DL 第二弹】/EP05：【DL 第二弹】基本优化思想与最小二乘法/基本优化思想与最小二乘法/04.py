import torch

# 定义可微分的参数a和b（初始值设为1）
a = torch.tensor(1.0, requires_grad=True)  # requires_grad=True表示需要计算梯度
b = torch.tensor(1.0, requires_grad=True)

# 计算SSE
sse = (2 - a - b)**2 + (4 - 3*a - b)** 2

# 计算偏导数
grads = torch.autograd.grad(sse, [a, b])
print(f"对a的偏导数：{grads[0]}")
print(f"对b的偏导数：{grads[1]}")