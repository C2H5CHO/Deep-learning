import torch

# 1. 初始化数据--特征+目标
X = torch.tensor(5)
Y = torch.tensor(0.)
# 2. 初始化参数--权重+偏置
w = torch.tensor(1, requires_grad=True, dtype=torch.float32)
b = torch.tensor(3, requires_grad=True, dtype=torch.float32)
# 3. 预测
z = w*X + b
# 4. 损失
loss = torch.nn.MSELoss()
loss = loss(z, Y)
# 5. 微分
loss.backward()
# 6. 梯度
print(f"w.grad：{w.grad}")
print(f"b.grad：{b.grad}")

print('--'*50)
# 1. 初始化数据--特征+目标
x = torch.ones(2, 5)
y = torch.ones(2, 3)
# 2. 初始化参数--权重+偏置
w = torch.randn(5, 3, requires_grad=True)
b = torch.randn(3, requires_grad=True)
# 3. 预测
z = torch.matmul(x, w) + b
# 4. 损失
loss = torch.nn.MSELoss()
loss = loss(z, y)
# 5. 微分
loss.backward()
# 6. 梯度
print(f"w.grad：{w.grad}")
print(f"b.grad：{b.grad}")