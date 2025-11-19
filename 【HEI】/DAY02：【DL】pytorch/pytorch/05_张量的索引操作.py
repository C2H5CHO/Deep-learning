import torch

torch.random.manual_seed(42)
data1 = torch.randint(0, 10, [4, 5])
print(f"data1：{data1}")

# 1. 行列索引
print(f"data1[0]：{data1[0]}")
print(f"data1[0, 0]：{data1[0, 0]}")
print(f"data1[:, 0]：{data1[:, 0]}")

print('--'*50)
# 2. 列表索引
print(f"data1[[0, 1], [1, 2]]：{data1[[0, 1], [1, 2]]}")
print(f"data1[[[0], [1]], [1, 2]]：{data1[[[0], [1]], [1, 2]]}")

print('--'*50)
# 3. 范围索引
print(f"data1[:3, :2]：{data1[:3, :2]}")
print(f"data1[2:, 2:]：{data1[2:, 2:]}")

print('--'*50)
# 4. 布尔索引
index_1 = data1[:, 2]>5
print(f"index_1：{index_1}")
print(f"data1[index_1]：{data1[index_1]}")

index_2 = data1[2]>3
print(f"index_2：{index_2}")
print(f"data1[:, index_2]：{data1[:, index_2]}")

print('--'*50)
# 5. 多维索引
torch.random.manual_seed(42)
data2 = torch.randint(0, 10, [3, 4, 5])
print(f"data2：{data2}")
print(f"data2[0, 0, 0]：{data2[0, 0, 0]}")
print(f"data2[0, :, 0]：{data2[0, :, 0]}")
print(f"data2[0, :, :]：{data2[0, :, :]}")

