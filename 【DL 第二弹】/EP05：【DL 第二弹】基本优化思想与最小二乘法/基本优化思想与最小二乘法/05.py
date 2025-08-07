import torch

# 数据点：x=[1,3]，y=[2,4]
x = torch.tensor([1.0, 3.0])
y = torch.tensor([2.0, 4.0])
m = len(x)  # 数据点数量

# 计算均值
x_mean = x.mean()
y_mean = y.mean()

# 计算w
numerator = torch.sum(y * (x - x_mean))  # 分子：sum(y_i*(x_i - x_mean))
denominator = torch.sum(x**2) - (torch.sum(x)** 2) / m  # 分母：sum(x_i²) - (sum(x_i))²/m
w = numerator / denominator

# 计算b
b = y_mean - w * x_mean

print(f"w = {w.item()}")
print(f"b = {b.item()}")