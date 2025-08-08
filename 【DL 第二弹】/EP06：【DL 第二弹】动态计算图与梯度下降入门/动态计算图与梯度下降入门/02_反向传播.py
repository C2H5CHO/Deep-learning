import torch

t1 = torch.tensor(1., requires_grad=True)
y1 = t1**2
z1 = y1 + 2

print(f"t1的导数值：{t1.grad}")
print(f"z1：{z1}")
print(f"z1的微分函数：{z1.grad_fn}")

# 反向传播
z1.backward()
print(f"t1的导数值：{t1.grad}")

# ×1. 不可以进行第二次反向传播
"""
z1.backward()
print(f"t1的导数值：{t1.grad}")
"""
# RuntimeError: Trying to backward through the graph a second time (or directly access saved tensors after they have already been freed). Saved intermediate values of the graph are freed when you call .backward() or autograd.grad(). Specify retain_graph=True if you need to backward through the graph a second time or if you need to access saved tensors after calling backward.

# *2. 中间节点和输出节点的反向传播的区别
t2 = torch.tensor(1., requires_grad=True)
y2 = t2**2
z2 = y2**2
print(f"t2的导数值：{t2.grad}")
z2.backward()
print(f"t2的导数值：{t2.grad}")

t2 = torch.tensor(1., requires_grad=True)
y2 = t2**2
z2 = y2**2
print(f"t2的导数值：{t2.grad}")
y2.backward()
print(f"t2的导数值：{t2.grad}")

# *3. 中间节点的梯度保存
t3 = torch.tensor(1., requires_grad=True)
y3 = t3**2
y3.retain_grad()
z3 = y3**2
z3.backward()
print(f"t3的导数值：{t3.grad}")
print(f"y3的导数值：{y3.grad}")

print('--'*50)
# 2. no_grad 阻止计算图记录
t4 = torch.tensor(1., requires_grad=True)
y4 = t4**2
with torch.no_grad():
    z4 = y4**2
print(f"z4：{z4}")
print(f"z4的微分函数：{z4.grad_fn}")
print(f"y4的微分函数：{y4.grad_fn}")

print('--'*50)
# 3. detach 创建一个不可导的相同张量
t5 = torch.tensor(1., requires_grad=True)
y5 = t5**2
y5_ = y5.detach()
z5 = y5_**2
print(f"y5的微分函数：{y5.grad_fn}")
print(f"z5的微分函数：{z5.grad_fn}")

print('--'*50)
# 4. is_leaf 叶结点
t6 = torch.tensor(1., requires_grad=True)
y6 = t6**2
z6 = y6**2
print(f"t6是否是叶结点：{t6.is_leaf}")
print(f"y6是否是叶结点：{y6.is_leaf}")
print(f"z6是否是叶结点：{z6.is_leaf}")

# *1. 任何一个新创建的张量都可以是叶结点
t7 = torch.tensor(1., requires_grad=True)
print(f"t7是否是叶结点：{t7.is_leaf}")

# *2. 经过detach的张量也可以叶结点
t8 = torch.tensor(1., requires_grad=True)
t8_ = t8.detach()
print(f"t8_是否是叶结点：{t8_.is_leaf}")

