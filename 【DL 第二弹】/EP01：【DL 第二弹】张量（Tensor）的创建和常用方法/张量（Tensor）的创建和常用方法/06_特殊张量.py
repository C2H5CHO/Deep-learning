import torch

# 1. 全0张量
t1 = torch.zeros([2, 3])
print(f"t1：{t1}")

# 2. 全1张量
t2 = torch.ones([3, 3])
print(f"t2：{t2}")

# 3. 对角矩阵
t3 = torch.tensor([1, 2, 3])
t3_ = torch.diag(t3)
print(f"t3_：{t3_}")

# 4. rand 服从0-1均匀分布
t4 = torch.rand(2, 3)
print(f"t4：{t4}")

# 5. randn 服从标准正态分布
t5 = torch.randn(2, 3)
print(f"t5：{t5}")

# 6. normal 服从指定正态分布
t6 = torch.normal(2, 3, size=(2, 2))
print(f"t6 均值为2，标准差为3：{t6}")

# 7. randint 整数随机采样结果
t7 = torch.randint(1, 10, [2, 4])
print(f"t7 在1-10之间随机抽取整数组成2行4列的张量：{t7}")

# 8. arange/linspace
t8 = torch.arange(5)
print(f"t8：{t8}")
t9 = torch.arange(1, 5, 0.5)
print(f"t9 从1-5（左闭右开），每隔0.5取1个值：{t9}")
t10 = torch.linspace(1, 5, 3)
print(f"t10 从1-5（闭），等距取3个数：{t10}")

# 9. empty 未初始化的指定形状矩阵
t11 = torch.empty(2, 3)
print(f"t11 2行3列：{t11}")

# 10. full 根据指定形状填充指定数据
t12 = torch.full([2, 3], 5)
print(f"t12：{t12}")
