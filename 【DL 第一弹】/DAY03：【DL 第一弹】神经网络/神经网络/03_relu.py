import torch
import matplotlib.pyplot as plt

# 原函数
x = torch.linspace(-20, 20, 1000)
y = torch.relu(x)
plt.plot(x.numpy(), y.numpy())
plt.grid()
plt.show()

# 导函数
x_ = torch.linspace(-20, 20, 1000, requires_grad=True)
torch.relu(x_).sum().backward()
plt.plot(x_.detach().numpy(), x_.grad.numpy())
plt.grid()
plt.show()