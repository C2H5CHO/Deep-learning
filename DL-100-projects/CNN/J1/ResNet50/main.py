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


