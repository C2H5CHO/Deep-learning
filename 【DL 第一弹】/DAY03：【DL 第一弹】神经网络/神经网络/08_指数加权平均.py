import torch
import matplotlib.pyplot as plt

temperatures = torch.randint(0, 40, [30])
print(temperatures)
days = torch.arange(0, 30, 1)
plt.plot(days, temperatures)
plt.scatter(days, temperatures)
plt.grid()
plt.show()

t_avg = []
beta = 0.9
for i, temp in enumerate(temperatures):
    if i == 0:
        t_avg.append(temp)
        continue

    t2 = beta*t_avg[i-1] + (1-beta)*temp
    t_avg.append(t2)

plt.plot(days, t_avg)
plt.scatter(days, temperatures)
plt.grid()
plt.show()