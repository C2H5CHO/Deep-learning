import torch

t1 = torch.arange(10)
print(f"t1：{t1}")

print('--'*50)

t1_1 = t1
print(f"t1_1：{t1_1}")
t1[0] = 100
print(f"t1：{t1}")
print(f"t1_1：{t1_1}")

print('--'*50)

t1_2 = t1.clone()
print(f"t1_2：{t1_2}")
t1[9] = 900
print(f"t1：{t1}")
print(f"t1_2：{t1_2}")
