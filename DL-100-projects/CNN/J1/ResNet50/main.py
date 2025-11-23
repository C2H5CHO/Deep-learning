# 1. 设置GPU
import torch
import torch.nn as nn
from torchvision import transforms, datasets
import os, PIL, pathlib, warnings
warnings.filterwarnings("ignore") # 作用：忽略警告信息

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(device) # 作用：打印当前使用的设备（GPU或CPU）

# 2. 可视化数据
import matplotlib.pyplot as plt
from data_factory.data_loader import visualize_images
fig, axes = visualize_images('./data/2Mild')
plt.show()

# 3. 输入数据
from data_factory.data_loader import load_dataset
data_total = load_dataset('./data')

# 4. 划分数据集
# (1) 计算训练集的大小（80%）和测试集的大小（20%）
train_size = int(0.8 * len(data_total)) # 作用：计算训练集的大小（80%）
test_size = len(data_total) - train_size # 作用：计算测试集的大小（20%）
train_dataset, test_dataset = torch.utils.data.random_split(data_total, [train_size, test_size]) # 作用：将数据集划分为训练集和测试集

# print(train_dataset)
# print(test_dataset)

# (2) 定义批次大小
batch_size = 3
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True) # 作用：将训练集划分为批次，并打乱顺序
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False) # 作用：将测试集划分为批次，但不打乱顺序

# 作用：查看一个批次的数据
"""
for X, y in test_loader:
    print(X.shape)
    print(y.shape, y.dtype)
    break
"""

# 5. 定义ResNet50模型
from ResNet50 import ResNet50

resnet50 = ResNet50().to(device)

# 6. 训练模型
import copy
from solve import train, test

# (1) 定义损失函数和优化器
loss_fn = nn.CrossEntropyLoss() # 作用：定义交叉熵损失函数
optimizer = torch.optim.Adam(resnet50.parameters(), lr=1e-4) # 作用：定义Adam优化器，学习率为1e-4

# (2) 训练模型
epochs = 3

train_loss = []
train_acc = []
test_loss = []
test_acc = []
best_acc = 0.0

for epoch in range(epochs):
    resnet50.train()
    epoch_train_acc, epoch_train_loss = train(train_loader, resnet50, loss_fn, optimizer, device)

    resnet50.eval()
    epoch_test_acc, epoch_test_loss = test(test_loader, resnet50, loss_fn, device)

    if epoch_test_acc > best_acc:
        best_acc = epoch_test_acc
        best_model = copy.deepcopy(resnet50)

    train_acc.append(epoch_train_acc)
    train_loss.append(epoch_train_loss)
    test_acc.append(epoch_test_acc)
    test_loss.append(epoch_test_loss)

    lr = optimizer.state_dict()['param_groups'][0]['lr'] # 作用：获取当前学习率
    template = (f"Epoch: {epoch+1:2d}, Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc*100:.1f}%, Test Loss: {epoch_test_loss:.4f}, Test Acc: {epoch_test_acc*100:.1f}%, LR: {lr:.2E}")
    print(template)

# (3) 保存模型
PATH = './model/resnet50.pth'
os.makedirs(os.path.dirname(PATH), exist_ok=True) # 作用：创建目录，如果目录已经存在，则不会报错
torch.save(best_model.state_dict(), PATH)
print("Model saved at", PATH)


# 7. 可视化loss和accuracy
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei'] # 作用：设置字体为黑体
plt.rcParams['axes.unicode_minus'] = False # 作用：解决负号'-'显示为方块的问题
plt.rcParams['figure.dpi'] = 100 # 作用：设置图像的分辨率

import warnings
warnings.filterwarnings("ignore") # 作用：忽略警告信息

from datetime import datetime
current_time = datetime.now()

epochs_range = range(epochs)
plt.figure(figsize=(25, 9))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, train_acc, label='Train Accuracy')
plt.plot(epochs_range, test_acc, label='Test Accuracy')
plt.legend(loc='lower right')
plt.title('Accuracy')
plt.xlabel(str(current_time))
plt.subplot(1, 2, 2)
plt.plot(epochs_range, train_loss, label='Train Loss')
plt.plot(epochs_range, test_loss, label='Test Loss')
plt.legend(loc='upper right')
plt.title('Loss')
plt.show()

# 8. 测试模型
best_model.load_state_dict(torch.load(PATH, map_location=device))
epoch_test_acc, epoch_test_loss = test(test_loader, best_model, loss_fn, device)
print(f"Test Accuracy: {epoch_test_acc*100:.1f}%, Test Loss: {epoch_test_loss:.4f}")

