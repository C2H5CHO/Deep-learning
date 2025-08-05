import torch

# 1. dist()
t1 = torch.tensor([1.0, 2.0, 3.0])
t2 = torch.tensor([4.0, 5.0, 6.0])
print("曼哈顿距离：", torch.dist(t1, t2, 2))
print("街道距离：", torch.dist(t1, t2, 1))

print('--'*50)
# 2. 维度
t2 = torch.arange(24).float().reshape(2 ,3 ,4)
print(t2)
print(torch.sum(t2, dim=0))
print(torch.sum(t2, dim=1))

print('--'*50)
# 3. 2维张量的排序
t3 = torch.randn(3 ,4)
print(t3)
print(f"默认按行进行升序：{torch.sort(t3)}")
print(f"按列进行降序：{torch.sort(t3, dim=1, descending=True)}")
