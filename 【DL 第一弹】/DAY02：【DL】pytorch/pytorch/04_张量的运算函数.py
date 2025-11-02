import torch

torch.random.manual_seed(42)
data1 = torch.rand([3, 4])
print(f"data1：{data1}")

# 1. mean 求均值
data1_mean = data1.mean()
print(f"data1_mean：{data1_mean}")
print(f"dim=0：{data1.mean(dim=0)}")
print(f"dim=1：{data1.mean(dim=1)}")

print('--'*50)
# 2. sum 求和
data1_sum = data1.sum()
print(f"data1_sum：{data1_sum}")
print(f"dim=0：{data1.sum(dim=0)}")
print(f"dim=1：{data1.sum(dim=1)}")

print('--'*50)
# 3. sqrt 求平方根
data1_sqrt = data1.sqrt()
print(f"data1_sqrt：{data1_sqrt}")

print('--'*50)
# 4. pow 求指数
data1_pow = data1.pow(2)
print(f"data1_pow：{data1_pow}")
data1_exp = data1.exp()
print(f"data1_exp：{data1_exp}")

print(f"2**data1：{torch.pow(2, data1)}")
print(f"data1**2：{torch.pow(data1, 2)}")

print('--'*50)
# 5. log 求对数
print(f"loge：{data1.log()}")
print(f"log2：{data1.log2()}")
print(f"log10：{data1.log10()}")

