import torch

w = torch.tensor([1.], requires_grad=True)
x = torch.tensor([2.], requires_grad=True)

a = torch.add(w, x)
a.retain_grad() # 保留中间变量的梯度
b = torch.add(w, 1)
y = torch.mul(a, b)

y.backward() # 反向传播
print(w.grad)

# 查看叶子节点
print("is_leaf：", w.is_leaf, x.is_leaf, a.is_leaf, b.is_leaf, y.is_leaf)

# 查看梯度
print("grad：", w.grad, x.grad, a.grad, b.grad, y.grad)

# 查看梯度函数
print("grad_fn：", w.grad_fn, x.grad_fn, a.grad_fn, b.grad_fn, y.grad_fn)
