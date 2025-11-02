import torch
import torch.nn as nn

input = torch.randn([1, 4])
layer = nn.Linear(in_features=4, out_features=5)
output = layer(input)
print(f"没有正则化的输出为：{output}")

dropout = nn.Dropout(p=0.75)
output = dropout(output)
print(f"正则化后的输出为：{output}")

