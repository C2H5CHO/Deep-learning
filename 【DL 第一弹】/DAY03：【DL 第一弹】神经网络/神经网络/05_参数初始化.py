import torch
import torch.nn as nn

# 1. 均匀分布初始化
linear1 = nn.Linear(in_features=5, out_features=3)
nn.init.uniform_(linear1.weight)
print(f"linear1：{linear1.weight.data}")

print('--'*50)
# 2. 固定值初始化
linear2 = nn.Linear(in_features=5, out_features=3)
nn.init.constant_(linear2.weight, 5)
print(f"linear2：{linear2.weight.data}")

print('--'*50)
# 3. 全0初始化
linear3 = nn.Linear(in_features=5, out_features=3)
nn.init.zeros_(linear3.weight)
print(f"linear3：{linear3.weight.data}")

print('--'*50)
# 4. 全1初始化
linear4 = nn.Linear(in_features=5, out_features=3)
nn.init.ones_(linear4.weight)
print(f"linear4：{linear4.weight.data}")

print('--'*50)
# 5. 正态分布随机初始化
linear5 = nn.Linear(in_features=5, out_features=3)
nn.init.normal_(linear5.weight, mean=0, std=1)
print(f"linear5：{linear5.weight.data}")

print('--'*50)
# 6. kaiming 随机初始化
## 6.1 正态分布
linear6 = nn.Linear(in_features=5, out_features=3)
nn.init.kaiming_normal_(linear6.weight)
print(f"linear6：{linear6.weight.data}")
## 6.2 均匀分布
linear7 = nn.Linear(in_features=5, out_features=3)
nn.init.kaiming_uniform_(linear7.weight)
print(f"linear7：{linear7.weight.data}")

print('--'*50)
# 7. xavier 随机初始化
## 7.1 正态分布
linear8 = nn.Linear(in_features=5, out_features=3)
nn.init.xavier_normal_(linear8.weight)
print(f"linear8：{linear8.weight.data}")
## 7.2 均匀分布
linear9 = nn.Linear(in_features=5, out_features=3)
nn.init.xavier_uniform_(linear9.weight)
print(f"linear9：{linear9.weight.data}")

