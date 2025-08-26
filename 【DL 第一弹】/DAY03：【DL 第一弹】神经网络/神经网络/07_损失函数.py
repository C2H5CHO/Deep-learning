import torch
import torch.nn as nn

# （1）标签
# y_true = torch.tensor([0, 1, 2], dtype=torch.int64)
# （2）one-hot
y_true = torch.tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=torch.float32)

y_pred = torch.tensor([[18, 9, 10], [2, 14, 6], [3, 8, 16]], dtype=torch.float32)

loss = nn.CrossEntropyLoss()
print(loss(y_pred, y_true))

