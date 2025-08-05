import torch

# 1. squeeze() 除去不必要的维度
t1 = torch.arange(4)
print(f"t1：{t1}")
t1_ = t1.reshape(1, 4)
print(f"t1_：{t1_}")
print(f"t1_的维度：{t1_.ndim}")
t1_1 = torch.squeeze(t1_)
print(f"t1_1：{t1_1}")
print(f"t1_1的维度：{t1_1.ndim}")

t2 = torch.zeros(1, 1, 3, 1)
print(f"t2：{t2}")
print(f"t2 降维：{torch.squeeze(t2)}")

t3 = torch.ones(1, 1, 3, 2, 1, 2)
print(f"t3：{t3}")
print(f"t3 降维：{torch.squeeze(t3)}，形状：{torch.squeeze(t3).shape}")

print('--'*50)
# 2. unsqeeze() 手动升维
t4 = torch.zeros(1, 2, 1, 2)
print(f"t4：{t4}")
print(f"在第1个维度上升高一个维度：{torch.unsqueeze(t4, 0)}，形状：{torch.unsqueeze(t4, 0).shape}")
print(f"在第3个维度上升高一个维度：{torch.unsqueeze(t4, 2)}，形状：{torch.unsqueeze(t4, 2).shape}")
