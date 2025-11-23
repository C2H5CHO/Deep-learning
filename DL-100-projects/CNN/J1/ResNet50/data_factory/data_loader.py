# 1. 可视化数据
import matplotlib.pyplot as plt
from PIL import Image
import os

def visualize_images(data_path, rows=3, cols=8, figsize=(25, 9)):
    """
    可视化指定文件夹中的图片
    
    Args:
        data_path (str): 包含图片的文件夹路径
        rows (int): 子图行数，默认为3
        cols (int): 子图列数，默认为8
        figsize (tuple): 图形大小，默认为(25, 9)
    
    Returns:
        fig: matplotlib的Figure对象
        axes: matplotlib的Axes对象数组
    """
    # 获取文件夹中的所有图片文件
    image_files = [f for f in os.listdir(data_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
    
    # 创建子图
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    
    # 可视化图片
    for ax, image_file in zip(axes.flat, image_files):
        # 读取图片
        img_path = os.path.join(data_path, image_file)
        img = Image.open(img_path)
        
        # 显示图片
        ax.imshow(img)
        ax.axis('off')
    
    plt.tight_layout()  # 调整子图参数，使之填充整个图像区域
    return fig, axes

"""
if __name__ == '__main__':
    fig, axes = visualize_images('../data/2Mild')
    plt.show()
"""

# 2. 输入数据
import torch
from torchvision import transforms, datasets

def load_dataset(data_dir):
    """
    加载和预处理图像数据集

    Args:
        data_dir (str): 数据集根目录路径

    Returns:
        dataset: 预处理后的数据集
    """
    # 定义数据转换
    transforms_train = transforms.Compose([
        transforms.Resize((224, 224)),  # 将图片调整为224x224
        transforms.ToTensor(),  # 将图片转换为Tensor格式，范围为[0, 1]
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # 归一化处理
    ])

    # 加载数据集
    dataset = datasets.ImageFolder(data_dir, transform=transforms_train)
    return dataset

"""
if __name__ == '__main__':
    data_total = load_dataset('../data')
    print(data_total)
    print(data_total.class_to_idx)
"""

