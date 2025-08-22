import torch
import numpy as np

# 1. torch.tensor() 根据指定数据创建张量
# 1.1 创建一个标量张量
data1 = torch.tensor(10)
print(f"data1：{data1}")
# 1.2 创建一个数组张量
data2 = np.random.randn(2, 3)
print(f"data2：{data2}")
data2_ = torch.tensor(data2)
print(f"data2_：{data2_}")
# 1.3 创建一个矩阵张量
data3 = [[10., 20., 30.], [40., 50., 60.]]
print(f"data3：{data3}")
data3_ = torch.tensor(data3)
print(f"data3_：{data3_}")

print('--'*50)
# 2. torch.Tensor 根据指定形状创建张量，也可以用来创建指定数据的张量
data4 = torch.Tensor(2, 3)
print(f"data4：{data4}")
data5 = torch.Tensor([10])
print(f"data5：{data5}")
data6 = torch.Tensor([10, 20])
print(f"data6：{data6}")

print('--'*50)
# 3. torch.IntTensor、torch.LongTensor、torch.FloatTensor、torch.DoubleTensor 创建指定类型的张量
data7 = torch.IntTensor(2, 3)
print(f"data7：{data7}")

data8 = torch.IntTensor([2.5, 3.7])
print(f"data8：{data8}")
data8_S = torch.ShortTensor([2.5, 3.7])
print(f"data8_S：{data8_S}")
data8_L = torch.LongTensor([2.5, 3.7])
print(f"data8_L：{data8_L}")
data8_F = torch.FloatTensor([2.5, 3.7])
print(f"data8_F：{data8_F}")
data8_D = torch.DoubleTensor([2.5, 3.7])
print(f"data8_D：{data8_D}")