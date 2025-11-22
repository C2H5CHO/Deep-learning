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
from PIL import Image

# (1) 指定要可视化的文件夹路径
data_plt = './data/2Mild'
# (2) 获取文件夹中的所有图片文件
image_files = [f for f in os.listdir(data_plt) if f.endswith(('.jpg', '.jpeg', '.png'))]
fig, axes = plt.subplots(3, 8, figsize=(16, 9))
# (3) 可视化图片
for ax, image_file in zip(axes.flat, image_files):
    # 读取图片
    img_path = os.path.join(data_plt, image_file)
    img = Image.open(img_path)

    # 显示图片
    ax.imshow(img)
    ax.axis('off')

plt.tight_layout() # 作用：调整子图参数，使之填充整个图像区域
# plt.show()

# 3. 输入数据
# (1) 指定数据目录
data_dir = './data'
# (2) 定义训练集的路径
transforms_train = transforms.Compose([
    transforms.Resize((224, 224)), # 作用：将图片调整为指定大小（224x224）
    transforms.ToTensor(),# 作用：将图片转换为Tensor格式，范围为[0, 1]
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # 作用：对图片进行归一化处理，将像素值范围从[0, 1]调整为[-1, 1]，
# 其中[0.485, 0.456, 0.406]是每个通道的均值，[0.229, 0.224, 0.225]是每个通道的标准差
])
# (3) 加载训练集
data_total = datasets.ImageFolder(data_dir, transform=transforms_train) # 作用：将数据集加载到内存中，并应用指定的转换

# print(data_total)
# print(data_total.class_to_idx)

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

