import torch
import torch.nn as nn

# 1. 多分类
# （1）标签
# y_true1 = torch.tensor([0, 1, 2], dtype=torch.int64)
# （2）one-hot
y_true1 = torch.tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=torch.float32)

y_pred1 = torch.tensor([[18, 9, 10], [2, 14, 6], [3, 8, 16]], dtype=torch.float32)

loss1 = nn.CrossEntropyLoss()
print(loss1(y_pred1, y_true1))

print('--'*50)
# 2. 二分类
y_true2 = torch.tensor([0, 1, 0, 1], dtype=torch.float32)
y_pred2 = torch.tensor([0.1, 0.9, 0.2, 0.8], dtype=torch.float32)

loss2 = nn.BCELoss()
print(loss2(y_pred2, y_true2))

print('--'*50)
# 3. 回归
## 3.1 均方误差
y_true3_1 = torch.tensor([2.0, 3.0, 1.0], dtype=torch.float32)
y_pred3_1 = torch.tensor([1.0, 5.0, 4.0], dtype=torch.float32)
loss3_1 = nn.L1Loss()
print(loss3_1(y_pred3_1, y_true3_1))
## 3.2 绝对误差
loss3_2 = nn.MSELoss()
print(loss3_2(y_pred3_1, y_true3_1))
## 3.3 平滑绝对误差
loss3_3 = nn.SmoothL1Loss()
print(loss3_3(y_pred3_1, y_true3_1))

