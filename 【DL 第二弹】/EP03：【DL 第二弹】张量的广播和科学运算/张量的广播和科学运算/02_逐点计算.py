import torch

# 1. 数学基本运算
t1 = torch.tensor([1, 2, 3])
t2 = torch.tensor([4, 5, 6])
print(f"相加：{torch.add(t1, t2)}")
print(f"相减：{torch.subtract(t2, t1)}")
print(f"相乘：{torch.multiply(t1, t2)}")
print(f"相除：{torch.divide(t2, t1)}")

print('--'*50)
# 2. 数值调整函数
t2 = torch.randn(5)
print(f"绝对值：{torch.abs(t2)}")
print(f"向上取整：{torch.ceil(t2)}")
print(f"向下取整：{torch.floor(t2)}")
print(f"四舍五入取整：{torch.round(t2)}")
print(f"相反数：{torch.neg(t2)}")

# *1. 若需要对原对象本身进行修改，则考虑使用 方法_()
print(f"t2：{t2}")
t2_ = torch.neg_(t2)
print(f"t2_：{t2_}")
print(f"t2：{t2}")
# *2. 许多科学计算也都有同名方法
t2_e = torch.exp_(t2)
print(f"t2_e：{t2_e}")
print(f"t2：{t2}")

print('--'*50)
# 3. 科学计算函数

# ×1. 只能作用于tensor对象
# print(f"2的2次方 整型：{torch.pow(2, 2)}") # TypeError: pow() received an invalid combination of arguments - got (int, int), but expected one of
print(f"2的2次方 0维张量：{torch.pow(torch.tensor(2), 2)}")
# *2. 具有一定的静态性
t3 = torch.arange(1, 4)
print(torch.expm1(t3))
# *3. 区分2的t次方和t的2次方
t4 = torch.randn(5)
print(torch.square(t4))
print(torch.sqrt(t4))
print(torch.pow(t4, 2))
print(torch.pow(t4, 0.5))
# *4. 幂运算 VS 对数运算
t5 = torch.tensor([1, 2, 3])
print(torch.exp(torch.log(t5)))
print(torch.exp2(torch.log2(t5)))

print('--'*50)
# 4. sort() 排序运算
t6 = torch.tensor([1, 3, 5, 2, 4])
print(f"升序：{torch.sort(t6)}")
print(f"降序：{torch.sort(t6, descending=True)}")
