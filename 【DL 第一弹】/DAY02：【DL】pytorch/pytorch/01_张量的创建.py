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

print('--'*50)
# 4. torch.arange、torch.linspace 创建线性张量
# 4.1 torch.arange(start, end, step)
data9 = torch.arange(0, 10, 2)
print(f"data9：{data9}")
# 4.2 torch.linspace(start, end, steps)
data10 = torch.linspace(0, 9, 10)
print(f"data10：{data10}")

print('--'*50)
# 5. torch.random.initial_seed、torch.random.manual_seed 创建随机数种子
data10 = torch.randn(2, 3)
print(f"data10：{data10}")
initial_seed = torch.random.initial_seed()
print(f"随机数种子：{initial_seed}")
torch.random.manual_seed(seed=initial_seed)
data11 = torch.randn(2, 3)
print(f"data11：{data11}")

print('--'*50)
# 6. torch.zeros、torch.zeros_like 创建全0张量
data12 = torch.zeros(2, 3)
print(f"data12：{data12}")
data12_ = torch.zeros_like(data12)
print(f"data13：{data12_}")

print('--'*50)
# 7. torch.ones、torch.ones_like 创建全1张量
data13 = torch.ones(2, 3)
print(f"data13：{data13}")
data13_ = torch.ones_like(data13)
print(f"data13_：{data13_}")

print('--'*50)
# 8. torch.full，torch.full_like 创建全指定值张量
data14 = torch.full([2, 3], 10)
print(f"data14：{data14}")
data14_ = torch.full_like(data14, 999)
print(f"data14_：{data14_}")

print('--'*50)
# 9. torch.type(torch.DoubleTensor)
data15 = torch.full([2, 3], 100)
print(f"data15的类型：{data15.type()}")
data15_D = data15.type(torch.DoubleTensor)
print(f"data15_D的类型：{data15_D.type()}")
data15_D1 = data15.double()
print(f"data15_D1的类型：{data15_D1.type()}")
data15_S = data15.short()
print(f"data15_S的类型：{data15_S.type()}")
data15_L = data15.long()
print(f"data15_L的类型：{data15_L.type()}")
data15_I = data15.int()
print(f"data15_I的类型：{data15_I.type()}")