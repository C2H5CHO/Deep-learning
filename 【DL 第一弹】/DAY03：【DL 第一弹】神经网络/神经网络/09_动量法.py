import torch
import torch.nn as nn

# （1）原始
w = torch.tensor([1.0], requires_grad=True, dtype=torch.float32)
loss = ((w**2) * 0.5).sum()

optimizer = torch.optim.SGD([w], lr=0.01)

optimizer.zero_grad()
loss.backward()
optimizer.step()
print(w.grad)
print(w.detach())

loss = ((w**2) * 0.5).sum()
optimizer.zero_grad()
loss.backward()
optimizer.step()
print(w.grad)
print(w.detach())

print('--'*50)
# （2）动量法
w = torch.tensor([1.0], requires_grad=True, dtype=torch.float32)
loss = ((w**2) * 0.5).sum()

optimizer = torch.optim.SGD([w], lr=0.01, momentum=0.9)

optimizer.zero_grad()
loss.backward()
optimizer.step()
print(w.grad)
print(w.detach())

loss = ((w**2) * 0.5).sum()
optimizer.zero_grad()
loss.backward()
optimizer.step()
print(w.grad)
print(w.detach())

