import torch

t1 = torch.tensor([1.0, 2, 3])
t2 = torch.tensor([1.0, 4, 5])
print(f"张量是否相同：{torch.equal(t1, t2)}")
print(f"元素是否相同：{torch.eq(t1, t2)}")