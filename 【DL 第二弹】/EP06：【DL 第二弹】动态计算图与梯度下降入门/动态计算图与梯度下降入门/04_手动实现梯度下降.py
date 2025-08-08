import torch

# 1. 手动实现
# step 1：设置初始参数
weights = torch.zeros(2, 1, requires_grad=True)
print(f"weights：{weights}")

# step 2：设置目标参数
X = torch.tensor([[1., 1], [3, 1]], requires_grad=True)
y = torch.tensor([2., 4], requires_grad=True).reshape(2, 1)
print(f"X：{X}")
print(f"y：{y}")

# step 3：设置学习率
eps = torch.tensor(0.01, requires_grad=True)
print(f"eps：{eps}")

# step 4：迭代
for k in range(1000):
    grad = torch.mm(X.t(), (torch.mm(X, weights) - y))/2
    weights = weights - eps * grad
print(f"weights：{weights}")

print('--'*50)
# 2. 封装函数
def Grad_Descent(X, y, eps = torch.tensor(0.01, requires_grad = True), max_iter = 1000):
    m, n = X.shape
    weights = torch.zeros(n, 1, requires_grad = True)
    for k in range(max_iter):
        grad = torch.mm(X.t(), (torch.mm(X, weights) - y))/2
        weights = weights - eps * grad
    return weights

X = torch.tensor([[1.,1],[3, 1]], requires_grad = True)
y = torch.tensor([2.,4], requires_grad = True).reshape(2,1)

weights = Grad_Descent(X, y)
print(f"weights：{weights}")

SSE = torch.mm((torch.mm(X, weights) - y).t(), (torch.mm(X, weights) - y))
print(f"SSE：{SSE}")

