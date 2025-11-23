- **🍨 本文为[🔗365天深度学习训练营](https://mp.weixin.qq.com/s/o-DaK6aQQLkJ8uE4YX1p3Q) 中的学习记录博客**
- **🍖 原作者：[K同学啊](https://mtyjkh.blog.csdn.net/)**


&emsp;&emsp;在深度学习的发展史上，ResNet（残差网络）无疑是一座里程碑。特别是ResNet50，凭借其在精度和计算成本之间的完美平衡，至今仍是计算机视觉任务中最常用的骨干网络。
# 一、解决问题
&emsp;&emsp;在2015年之前，深度学习面临着一个尴尬的瓶颈：==网络越深，效果反而越差==。

&emsp;&emsp;按照常理，神经网络的层数越多，提取特征的能力应该越强，准确率应该越高。然而，科学家们发现，当网络深度达到一定程度（比如20层以上）时，单纯地堆叠层数会导致两个严重问题：
1. **梯度消失/爆炸**（Gradient Vanishing/Exploding）：导致网络无法收敛（虽然可以通过BatchNorm缓解）；
2. **退化问题**（Degradation Problem）：这是最关键的。随着层数增加，训练集的准确率反而下降了。这不是过拟合，而是网络学不动了。

&emsp;&emsp;ResNet（Residual Network） 的出现，就是为了解决这个问题。它由何恺明（Kaiming He）等人在2015年提出，一举拿下了当年ImageNet比赛的冠军，将网络深度从之前的十几层瞬间推到了152层甚至更深。
# 二、核心思想
&emsp;&emsp;ResNet的核心思想非常简单，却极其巧妙，它引入了**跳跃连接**（Skip Connection/Shortcut）。
## 2.1 是什么？
在传统的卷积神经网络（CNN）中，数据是一层一层往下传的。而在ResNet中，输入数据$x$除了通过正常的卷积层$F(x)$进行变换外，还通过一条捷径直接加到了输出上。

数学公式表示为：
$$H(x) = F(x) + x$$

其中：
- $x$：输入数据。
- $F(x)$：残差映射（经过卷积、激活等操作后的结果）。
- $H(x)$：最终输出。
## 2.2 为什么？
想象一下，如果网络已经达到了最优状态，再增加层数，我们希望这些新层什么都不做，即实现**恒等映射**（$Output = Input$）。
- **传统网络**：要让卷积层参数学习出$F(x) = x$是很难的。
- **ResNet**：只需要将$F(x)$的参数全部学习为0，那么$H(x) = 0 + x = x$。让网络学习0比学习恒等映射要容易得多。

这就是**残差**的含义：网络只需要学习输入和输出之间的**差值**（Residual）。
# 三、核心结构
&emsp;&emsp;ResNet有很多版本（18，34，50，101，152）。其中ResNet50引入了特殊的**Bottleneck**（瓶颈）结构，这也是它能在保持深度的同时控制计算量的关键。
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/6d4359568bfb4983857b224a2305aea1.png#pic_center)

## 3.1 BasicBlock `VS` Bottleneck

| 对比 | BasicBlock（用于ResNet18/34） | Bottleneck（用于ResNet50/101/152） |
| :--- | :--- | :--- |
| **结构** | 2个$3 \times 3$卷积 | **3个卷积**：$1 \times 1 \to 3 \times 3 \to 1 \times 1$ |
| **目的** | 基础特征提取 | **降维 $\to$ 卷积 $\to$ 升维**（减少参数量） |
| **通道变化** | 保持不变（如64 $\to$ 64） | **两头大中间小**（如256 $\to$ 64 $\to$ 256） |

## 3.2 Bottleneck的核心计算
假设输入通道是256：
1. **$1 \times 1$Conv**（降维）：将通道数从256降到64，大幅减少了后续计算量；
2. **$3 \times 3$Conv**（特征提取）：在64个通道上进行卷积，计算开销小；
3. **$1 \times 1$Conv**（升维）：将通道数恢复至256，以便与Shortcut($x$)相加。
# 四、整体架构
&emsp;&emsp;ResNet50可以拆解为**Stem**（作为起点的预处理） + **4个Stage** + **Head**（分类头）。

| 阶段 (Stage) | 输出尺寸 | 结构详解（Block $\times$ 数量） | 输出通道数 |
| :--- | :--- | :--- | :--- |
| **Input** | $224 \times 224$ | RGB 图像 | 3 |
| **Stem** | $112 \times 112$ | $7 \times 7$ Conv，stride 2 <br> $3 \times 3$ MaxPool，stride 2 | 64 |
| **Stage 1** | $56 \times 56$ | $\begin{bmatrix} 1\times1, 64 \\ 3\times3, 64 \\ 1\times1, 256 \end{bmatrix} \times 3$ | 256 |
| **Stage 2** | $28 \times 28$ | $\begin{bmatrix} 1\times1, 128 \\ 3\times3, 128 \\ 1\times1, 512 \end{bmatrix} \times 4$ | 512 |
| **Stage 3** | $14 \times 14$ | $\begin{bmatrix} 1\times1, 256 \\ 3\times3, 256 \\ 1\times1, 1024 \end{bmatrix} \times 6$ | 1024 |
| **Stage 4** | $7 \times 7$ | $\begin{bmatrix} 1\times1, 512 \\ 3\times3, 512 \\ 1\times1, 2048 \end{bmatrix} \times 3$ | 2048 |
| **Head** | $1 \times 1$ | Global Avg Pool + FC (1000) | 1000 |

> **🔍计算层数**：$(3+4+6+3) \times 3 = 48$个卷积层，加上开头的$7 \times 7$卷积和最后的FC层，刚好**50**层。
# 五、代码实战
## 5.1 设置GPU（$\to$ `main.py`）

```python
import torch
import torch.nn as nn
from torchvision import transforms, datasets
import os, PIL, pathlib, warnings
warnings.filterwarnings("ignore") # 作用：忽略警告信息

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device) # 作用：打印当前使用的设备（GPU或CPU）
```
- 运行结果：
```
cuda
```
## 5.2 可视化数据集（$\to$ `data_loader.py`）

```python
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

if __name__ == '__main__':
    fig, axes = visualize_images('../data/2Mild')
    plt.show()
```
- 运行结果：
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/95595958a3674f9c9ecba9e3b1ff0a17.png)
## 5.3 输入数据集（$\to$ `data_loader.py`）

```python
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

if __name__ == '__main__':
    data_total = load_dataset('../data')
    print(data_total)
    print(data_total.class_to_idx)
```
- 运行结果：
```
Dataset ImageFolder
    Number of datapoints: 1661
    Root location: ../data
    StandardTransform
Transform: Compose(
               Resize(size=(224, 224), interpolation=bilinear, max_size=None, antialias=True)
               ToTensor()
               Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
           )
{'0Normal': 0, '2Mild': 1, '4Severe': 2}
```
## 5.4 划分数据集（$\to$ `main.py`）

```python
# (1) 计算训练集的大小（80%）和测试集的大小（20%）
train_size = int(0.8 * len(data_total)) # 作用：计算训练集的大小（80%）
test_size = len(data_total) - train_size # 作用：计算测试集的大小（20%）
train_dataset, test_dataset = torch.utils.data.random_split(data_total, [train_size, test_size]) # 作用：将数据集划分为训练集和测试集

print(train_dataset)
print(test_dataset)

# (2) 定义批次大小
batch_size = 10
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True) # 作用：将训练集划分为批次，并打乱顺序
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False) # 作用：将测试集划分为批次，但不打乱顺序

# 作用：查看一个批次的数据
for X, y in test_loader:
    print(X.shape)
    print(y.shape, y.dtype)
    break
```
- 运行结果：
```
<torch.utils.data.dataset.Subset object at 0x00000236D9AC0350>
<torch.utils.data.dataset.Subset object at 0x00000236D979CA50>
torch.Size([10, 3, 224, 224])
torch.Size([10]) torch.int64
```
## 5.5 定义功能函数（$\to$ `solver.py`）
### 5.5.1 定义自动填充函数

```python
def autopad(k, p=None):
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k] # 作用：计算填充大小
    return p
```
### 5.5.2 定义残差块类

```python
class IdentityBlock(nn.Module):
    """
    定义残差块类

    Args:
        in_channel: 输入通道数
        kernel_size: 卷积核大小
        filters: 卷积核个数列表

    Returns:
        x: 输出特征图
    """
    def __init__(self, in_channel, kernel_size, filters):
       super(IdentityBlock, self).__init__()
       filters1, filters2, filters3 = filters # 作用：将filters列表中的元素分别赋值给filters1、filters2、filters3

       # (1) 1x1卷积层
       self.conv1 = nn.Sequential(
           nn.Conv2d(in_channels=in_channel, out_channels=filters1, kernel_size=1, stride=1, padding=0, bias=False), # 卷积层，作用：将输入的in_channel维度的数据通过1x1卷积核转换为filters1维度的数据
           nn.BatchNorm2d(filters1), # 归一化层，作用：对filters1维度的数据进行归一化处理
           nn.ReLU(inplace=True), # 激活函数层，作用：对filters1维度的数据进行非线性变换，引入非线性特征
       )
       # (2) 3x3卷积层
       self.conv2 = nn.Sequential(
           nn.Conv2d(in_channels=filters1, out_channels=filters2, kernel_size=kernel_size, stride=1, padding=autopad(kernel_size), bias=False), # 卷积层，作用：将filters1维度的数据通过kernel_sizexkernel_size卷积核转换为filters2维度的数据
           nn.BatchNorm2d(filters2), # 归一化层，作用：对filters2维度的数据进行归一化处理
           nn.ReLU(inplace=True), # 激活函数层，作用：对filters2维度的数据进行非线性变换，引入非线性特征
       )
       # (3) 1x1卷积层
       self.conv3 = nn.Sequential(
           nn.Conv2d(in_channels=filters2, out_channels=filters3, kernel_size=1, stride=1, padding=0, bias=False), # 卷积层，作用：将filters2维度的数据通过1x1卷积核转换为filters3维度的数据
           nn.BatchNorm2d(filters3), # 归一化层，作用：对filters3维度的数据进行归一化处理
       )
       # (4) 激活函数层
       self.relu = nn.ReLU(inplace=True) # 激活函数层，作用：对filters3维度的数据进行非线性变换，引入非线性特征

    def forward(self, x):
        x1 = self.conv1(x)
        x1 = self.conv2(x1)
        x1 = self.conv3(x1)
        x = x1 + x # 作用：将输入x与经过卷积层后的x1相加，实现残差连接
        x = self.relu(x)
        return x
```
### 5.5.3 定义卷积类

```python
# 3. 定义卷积块类
class ConvBlock(nn.Module):
    """
    定义卷积块类

    Args:
        in_channel: 输入通道数
        kernel_size: 卷积核大小
        filters: 卷积核个数列表
        stride: 步长

    Returns:
        x: 输出特征图
    """

    def __init__(self, in_channel, kernel_size, filters, stride=2):
        super(ConvBlock, self).__init__()
        filters1, filters2, filters3 = filters # 作用：将filters列表中的元素分别赋值给filters1、filters2、filters3

        # (1) 1x1卷积层
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=filters1, kernel_size=1, stride=stride, padding=0, bias=False), # 卷积层，作用：将输入的in_channel维度的数据通过1x1卷积核转换为filters1维度的数据
            nn.BatchNorm2d(filters1), # 归一化层，作用：对filters1维度的数据进行归一化处理
            nn.ReLU(inplace=True), # 激活函数层，作用：对filters1维度的数据进行非线性变换，引入非线性特征
        )
        # (2) 3x3卷积层
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=filters1, out_channels=filters2, kernel_size=kernel_size, stride=1, padding=autopad(kernel_size), bias=False), # 卷积层，作用：将filters1维度的数据通过kernel_sizexkernel_size卷积核转换为filters2维度的数据
            nn.BatchNorm2d(filters2), # 归一化层，作用：对filters2维度的数据进行归一化处理
            nn.ReLU(inplace=True), # 激活函数层，作用：对filters2维度的数据进行非线性变换，引入非线性特征
        )
        # (3) 1x1卷积层
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=filters2, out_channels=filters3, kernel_size=1, stride=1, padding=0, bias=False), # 卷积层，作用：将filters2维度的数据通过1x1卷积核转换为filters3维度的数据
            nn.BatchNorm2d(filters3), # 归一化层，作用：对filters3维度的数据进行归一化处理
        )
        # (4) 1x1卷积层（用于匹配维度）
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=filters3, kernel_size=1, stride=stride, padding=0, bias=False), # 卷积层，作用：将输入的in_channel维度的数据通过1x1卷积核转换为filters3维度的数据
            nn.BatchNorm2d(filters3), # 归一化层，作用：对filters3维度的数据进行归一化处理
        )
        # (5) 激活函数层
        self.relu = nn.ReLU(inplace=True) # 激活函数层，作用：对filters3维度的数据进行非线性变换，引入非线性特征

    def forward(self, x):
        x1 = self.conv1(x)
        x1 = self.conv2(x1)
        x1 = self.conv3(x1)
        x = self.conv4(x) + x1 # 作用：将输入x与经过卷积层后的x1相加，实现残差连接
        x = self.relu(x)
        return x
```
### 5.5.4  定义训练函数

```python
def train(dataloader, model, loss_fn, optimizer, device):
    """
    定义训练函数

    Args:
        dataloader: 训练数据集
        model: 模型
        loss_fn: 损失函数
        optimizer: 优化器
        device: 计算设备

    Returns:
        train_loss: 训练损失
        train_acc: 训练准确率
    """
    
    size = len(dataloader.dataset)  # 作用：获取训练集的样本数量
    num_batches = len(dataloader) # 作用：获取训练集的批次数量
    train_loss, train_acc = 0, 0 # 作用：初始化训练损失和训练准确率为0

    for x, y in dataloader:
        x, y = x.to(device), y.to(device) # 作用：将输入数据和标签数据移动到GPU上
        pred = model(x) # 作用：将输入数据x输入到模型中，得到预测结果pred
        loss = loss_fn(pred, y) # 作用：计算预测结果pred与真实标签y之间的损失

        # 反向传播
        optimizer.zero_grad() # 作用：将优化器的梯度清零
        loss.backward() # 作用：计算损失函数的梯度
        optimizer.step() # 作用：根据计算得到的梯度，更新模型的参数

        train_acc += (pred.argmax(1) == y).type(torch.float).sum().item() # 作用：计算当前批次的准确率，并累加到train_acc中
        train_loss += loss.item() # 作用：将当前批次的损失值累加到train_loss中

    train_acc /= size # 作用：计算训练集的平均准确率
    train_loss /= num_batches # 作用：计算训练集的平均损失

    return train_loss, train_acc
```
### 5.5.5 定义测试函数

```python
def test(dataloader, model, loss_fn, device):
    """
    定义测试函数
    
    Args:
        dataloader: 测试数据集
        model: 模型
        loss_fn: 损失函数
        device: 计算设备
        
    Returns:
         test_loss: 测试损失
         test_acc: 测试准确率
    """
    
    size = len(dataloader.dataset) # 作用：获取测试集的样本数量
    num_batches = len(dataloader) # 作用：获取测试集的批次数量
    test_loss, test_acc = 0, 0 # 作用：初始化测试损失和测试准确率为0

    with torch.no_grad():
        for imgs, target in dataloader:
            imgs, target = imgs.to(device), target.to(device) # 作用：将输入数据和标签数据移动到GPU上
            pred = model(imgs) # 作用：将输入数据imgs输入到模型中，得到预测结果pred
            loss = loss_fn(pred, target) # 作用：计算预测结果pred与真实标签targets之间的损失

            test_loss += loss.item() # 作用：将当前批次的损失值累加到test_loss中
            test_acc += (pred.argmax(1) == target).type(torch.float).sum().item() # 作用：计算当前批次的准确率，并累加到test_acc中

    test_acc /= size # 作用：计算测试集的平均准确率
    test_loss /= num_batches # 作用：计算测试集的平均损失

    return test_loss, test_acc
```
## 5.6 构建ResNet50模型（$\to$ `ResNet50.py`）

```python
import torch
import torch.nn as nn
from torchsummary import summary
from solve import ConvBlock, IdentityBlock

class ResNet50(nn.Module):
    def __init__(self, classes=1000):
        super(ResNet50, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False, padding_mode='zeros'), # 卷积层，作用：将输入的3通道数据通过7x7卷积核转换为64通道的数据
            nn.BatchNorm2d(64), # 归一化层，作用：对64通道的数据进行归一化处理
            nn.ReLU(), # 激活函数层，作用：对64通道的数据进行非线性变换，引入非线性特征
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0) # 最大池化层，作用：对64通道的数据进行最大池化操作，降低特征图的尺寸
        )
        self.conv2 = nn.Sequential(
            ConvBlock(in_channel=64, kernel_size=3, filters=[64, 64, 256], stride=1),
            IdentityBlock(in_channel=256, kernel_size=3, filters=[64, 64, 256]),
            IdentityBlock(in_channel=256, kernel_size=3, filters=[64, 64, 256])
        )
        self.conv3 = nn.Sequential(
            ConvBlock(in_channel=256, kernel_size=3, filters=[128, 128, 512], stride=2),
            IdentityBlock(in_channel=512, kernel_size=3, filters=[128, 128, 512]),
            IdentityBlock(in_channel=512, kernel_size=3, filters=[128, 128, 512]),
            IdentityBlock(in_channel=512, kernel_size=3, filters=[128, 128, 512])
        )
        self.conv4 = nn.Sequential(
            ConvBlock(in_channel=512, kernel_size=3, filters=[256, 256, 1024], stride=2),
            IdentityBlock(in_channel=1024, kernel_size=3, filters=[256, 256, 1024]),
            IdentityBlock(in_channel=1024, kernel_size=3, filters=[256, 256, 1024]),
            IdentityBlock(in_channel=1024, kernel_size=3, filters=[256, 256, 1024]),
            IdentityBlock(in_channel=1024, kernel_size=3, filters=[256, 256, 1024]),
            IdentityBlock(in_channel=1024, kernel_size=3, filters=[256, 256, 1024])
        )
        self.conv5 = nn.Sequential(
            ConvBlock(in_channel=1024, kernel_size=3, filters=[512, 512, 2048], stride=2),
            IdentityBlock(in_channel=2048, kernel_size=3, filters=[512, 512, 2048]),
            IdentityBlock(in_channel=2048, kernel_size=3, filters=[512, 512, 2048])
        )
        self.pool = nn.AvgPool2d(kernel_size=7, stride=1) # 平均池化层，作用：对2048通道的数据进行平均池化操作，降低特征图的尺寸为1x1
        self.fc = nn.Linear(2048, 3) # 全连接层，作用：将2048通道的数据映射到类别数（3）

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1) # 作用：将特征图的尺寸调整为(batch_size, 2048)
        x = self.fc(x)
        return x

if __name__ == '__main__':
    model = ResNet50().to('cuda')
    print(model)
    print('--'*50)
    summary(model, (3, 224, 224))
```
- 运行结果：
```
ResNet50(
  (conv1): Sequential(
    (0): Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): ReLU()
    (3): MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)
  )
  (conv2): Sequential(
    (0): ConvBlock(
      (conv1): Sequential(
        (0): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): ReLU(inplace=True)
      )
      (conv2): Sequential(
        (0): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): ReLU(inplace=True)
      )
      (conv3): Sequential(
        (0): Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
      (conv4): Sequential(
        (0): Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
      (relu): ReLU(inplace=True)
    )
    (1): IdentityBlock(
      (conv1): Sequential(
        (0): Conv2d(256, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): ReLU(inplace=True)
      )
      (conv2): Sequential(
        (0): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): ReLU(inplace=True)
      )
      (conv3): Sequential(
        (0): Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
      (relu): ReLU(inplace=True)
    )
    (2): IdentityBlock(
      (conv1): Sequential(
        (0): Conv2d(256, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): ReLU(inplace=True)
      )
      (conv2): Sequential(
        (0): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): ReLU(inplace=True)
      )
      (conv3): Sequential(
        (0): Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
      (relu): ReLU(inplace=True)
    )
  )
  (conv3): Sequential(
    (0): ConvBlock(
      (conv1): Sequential(
        (0): Conv2d(256, 128, kernel_size=(1, 1), stride=(2, 2), bias=False)
        (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): ReLU(inplace=True)
      )
      (conv2): Sequential(
        (0): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): ReLU(inplace=True)
      )
      (conv3): Sequential(
        (0): Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
      (conv4): Sequential(
        (0): Conv2d(256, 512, kernel_size=(1, 1), stride=(2, 2), bias=False)
        (1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
      (relu): ReLU(inplace=True)
    )
    (1): IdentityBlock(
      (conv1): Sequential(
        (0): Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): ReLU(inplace=True)
      )
      (conv2): Sequential(
        (0): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): ReLU(inplace=True)
      )
      (conv3): Sequential(
        (0): Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
      (relu): ReLU(inplace=True)
    )
    (2): IdentityBlock(
      (conv1): Sequential(
        (0): Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): ReLU(inplace=True)
      )
      (conv2): Sequential(
        (0): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): ReLU(inplace=True)
      )
      (conv3): Sequential(
        (0): Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
      (relu): ReLU(inplace=True)
    )
    (3): IdentityBlock(
      (conv1): Sequential(
        (0): Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): ReLU(inplace=True)
      )
      (conv2): Sequential(
        (0): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): ReLU(inplace=True)
      )
      (conv3): Sequential(
        (0): Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
      (relu): ReLU(inplace=True)
    )
  )
  (conv4): Sequential(
    (0): ConvBlock(
      (conv1): Sequential(
        (0): Conv2d(512, 256, kernel_size=(1, 1), stride=(2, 2), bias=False)
        (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): ReLU(inplace=True)
      )
      (conv2): Sequential(
        (0): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): ReLU(inplace=True)
      )
      (conv3): Sequential(
        (0): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (1): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
      (conv4): Sequential(
        (0): Conv2d(512, 1024, kernel_size=(1, 1), stride=(2, 2), bias=False)
        (1): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
      (relu): ReLU(inplace=True)
    )
    (1): IdentityBlock(
      (conv1): Sequential(
        (0): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): ReLU(inplace=True)
      )
      (conv2): Sequential(
        (0): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): ReLU(inplace=True)
      )
      (conv3): Sequential(
        (0): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (1): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
      (relu): ReLU(inplace=True)
    )
    (2): IdentityBlock(
      (conv1): Sequential(
        (0): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): ReLU(inplace=True)
      )
      (conv2): Sequential(
        (0): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): ReLU(inplace=True)
      )
      (conv3): Sequential(
        (0): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (1): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
      (relu): ReLU(inplace=True)
    )
    (3): IdentityBlock(
      (conv1): Sequential(
        (0): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): ReLU(inplace=True)
      )
      (conv2): Sequential(
        (0): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): ReLU(inplace=True)
      )
      (conv3): Sequential(
        (0): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (1): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
      (relu): ReLU(inplace=True)
    )
    (4): IdentityBlock(
      (conv1): Sequential(
        (0): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): ReLU(inplace=True)
      )
      (conv2): Sequential(
        (0): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): ReLU(inplace=True)
      )
      (conv3): Sequential(
        (0): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (1): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
      (relu): ReLU(inplace=True)
    )
    (5): IdentityBlock(
      (conv1): Sequential(
        (0): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): ReLU(inplace=True)
      )
      (conv2): Sequential(
        (0): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): ReLU(inplace=True)
      )
      (conv3): Sequential(
        (0): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (1): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
      (relu): ReLU(inplace=True)
    )
  )
  (conv5): Sequential(
    (0): ConvBlock(
      (conv1): Sequential(
        (0): Conv2d(1024, 512, kernel_size=(1, 1), stride=(2, 2), bias=False)
        (1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): ReLU(inplace=True)
      )
      (conv2): Sequential(
        (0): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): ReLU(inplace=True)
      )
      (conv3): Sequential(
        (0): Conv2d(512, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (1): BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
      (conv4): Sequential(
        (0): Conv2d(1024, 2048, kernel_size=(1, 1), stride=(2, 2), bias=False)
        (1): BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
      (relu): ReLU(inplace=True)
    )
    (1): IdentityBlock(
      (conv1): Sequential(
        (0): Conv2d(2048, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): ReLU(inplace=True)
      )
      (conv2): Sequential(
        (0): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): ReLU(inplace=True)
      )
      (conv3): Sequential(
        (0): Conv2d(512, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (1): BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
      (relu): ReLU(inplace=True)
    )
    (2): IdentityBlock(
      (conv1): Sequential(
        (0): Conv2d(2048, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): ReLU(inplace=True)
      )
      (conv2): Sequential(
        (0): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): ReLU(inplace=True)
      )
      (conv3): Sequential(
        (0): Conv2d(512, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (1): BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
      (relu): ReLU(inplace=True)
    )
  )
  (pool): AvgPool2d(kernel_size=7, stride=1, padding=0)
  (fc): Linear(in_features=2048, out_features=3, bias=True)
)
----------------------------------------------------------------------------------------------------
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1         [-1, 64, 112, 112]           9,408
       BatchNorm2d-2         [-1, 64, 112, 112]             128
              ReLU-3         [-1, 64, 112, 112]               0
         MaxPool2d-4           [-1, 64, 55, 55]               0
            Conv2d-5           [-1, 64, 55, 55]           4,096
       BatchNorm2d-6           [-1, 64, 55, 55]             128
              ReLU-7           [-1, 64, 55, 55]               0
            Conv2d-8           [-1, 64, 55, 55]          36,864
       BatchNorm2d-9           [-1, 64, 55, 55]             128
             ReLU-10           [-1, 64, 55, 55]               0
           Conv2d-11          [-1, 256, 55, 55]          16,384
      BatchNorm2d-12          [-1, 256, 55, 55]             512
           Conv2d-13          [-1, 256, 55, 55]          16,384
      BatchNorm2d-14          [-1, 256, 55, 55]             512
             ReLU-15          [-1, 256, 55, 55]               0
        ConvBlock-16          [-1, 256, 55, 55]               0
           Conv2d-17           [-1, 64, 55, 55]          16,384
      BatchNorm2d-18           [-1, 64, 55, 55]             128
             ReLU-19           [-1, 64, 55, 55]               0
           Conv2d-20           [-1, 64, 55, 55]          36,864
      BatchNorm2d-21           [-1, 64, 55, 55]             128
             ReLU-22           [-1, 64, 55, 55]               0
           Conv2d-23          [-1, 256, 55, 55]          16,384
      BatchNorm2d-24          [-1, 256, 55, 55]             512
             ReLU-25          [-1, 256, 55, 55]               0
    IdentityBlock-26          [-1, 256, 55, 55]               0
           Conv2d-27           [-1, 64, 55, 55]          16,384
      BatchNorm2d-28           [-1, 64, 55, 55]             128
             ReLU-29           [-1, 64, 55, 55]               0
           Conv2d-30           [-1, 64, 55, 55]          36,864
      BatchNorm2d-31           [-1, 64, 55, 55]             128
             ReLU-32           [-1, 64, 55, 55]               0
           Conv2d-33          [-1, 256, 55, 55]          16,384
      BatchNorm2d-34          [-1, 256, 55, 55]             512
             ReLU-35          [-1, 256, 55, 55]               0
    IdentityBlock-36          [-1, 256, 55, 55]               0
           Conv2d-37          [-1, 128, 28, 28]          32,768
      BatchNorm2d-38          [-1, 128, 28, 28]             256
             ReLU-39          [-1, 128, 28, 28]               0
           Conv2d-40          [-1, 128, 28, 28]         147,456
      BatchNorm2d-41          [-1, 128, 28, 28]             256
             ReLU-42          [-1, 128, 28, 28]               0
           Conv2d-43          [-1, 512, 28, 28]          65,536
      BatchNorm2d-44          [-1, 512, 28, 28]           1,024
           Conv2d-45          [-1, 512, 28, 28]         131,072
      BatchNorm2d-46          [-1, 512, 28, 28]           1,024
             ReLU-47          [-1, 512, 28, 28]               0
        ConvBlock-48          [-1, 512, 28, 28]               0
           Conv2d-49          [-1, 128, 28, 28]          65,536
      BatchNorm2d-50          [-1, 128, 28, 28]             256
             ReLU-51          [-1, 128, 28, 28]               0
           Conv2d-52          [-1, 128, 28, 28]         147,456
      BatchNorm2d-53          [-1, 128, 28, 28]             256
             ReLU-54          [-1, 128, 28, 28]               0
           Conv2d-55          [-1, 512, 28, 28]          65,536
      BatchNorm2d-56          [-1, 512, 28, 28]           1,024
             ReLU-57          [-1, 512, 28, 28]               0
    IdentityBlock-58          [-1, 512, 28, 28]               0
           Conv2d-59          [-1, 128, 28, 28]          65,536
      BatchNorm2d-60          [-1, 128, 28, 28]             256
             ReLU-61          [-1, 128, 28, 28]               0
           Conv2d-62          [-1, 128, 28, 28]         147,456
      BatchNorm2d-63          [-1, 128, 28, 28]             256
             ReLU-64          [-1, 128, 28, 28]               0
           Conv2d-65          [-1, 512, 28, 28]          65,536
      BatchNorm2d-66          [-1, 512, 28, 28]           1,024
             ReLU-67          [-1, 512, 28, 28]               0
    IdentityBlock-68          [-1, 512, 28, 28]               0
           Conv2d-69          [-1, 128, 28, 28]          65,536
      BatchNorm2d-70          [-1, 128, 28, 28]             256
             ReLU-71          [-1, 128, 28, 28]               0
           Conv2d-72          [-1, 128, 28, 28]         147,456
      BatchNorm2d-73          [-1, 128, 28, 28]             256
             ReLU-74          [-1, 128, 28, 28]               0
           Conv2d-75          [-1, 512, 28, 28]          65,536
      BatchNorm2d-76          [-1, 512, 28, 28]           1,024
             ReLU-77          [-1, 512, 28, 28]               0
    IdentityBlock-78          [-1, 512, 28, 28]               0
           Conv2d-79          [-1, 256, 14, 14]         131,072
      BatchNorm2d-80          [-1, 256, 14, 14]             512
             ReLU-81          [-1, 256, 14, 14]               0
           Conv2d-82          [-1, 256, 14, 14]         589,824
      BatchNorm2d-83          [-1, 256, 14, 14]             512
             ReLU-84          [-1, 256, 14, 14]               0
           Conv2d-85         [-1, 1024, 14, 14]         262,144
      BatchNorm2d-86         [-1, 1024, 14, 14]           2,048
           Conv2d-87         [-1, 1024, 14, 14]         524,288
      BatchNorm2d-88         [-1, 1024, 14, 14]           2,048
             ReLU-89         [-1, 1024, 14, 14]               0
        ConvBlock-90         [-1, 1024, 14, 14]               0
           Conv2d-91          [-1, 256, 14, 14]         262,144
      BatchNorm2d-92          [-1, 256, 14, 14]             512
             ReLU-93          [-1, 256, 14, 14]               0
           Conv2d-94          [-1, 256, 14, 14]         589,824
      BatchNorm2d-95          [-1, 256, 14, 14]             512
             ReLU-96          [-1, 256, 14, 14]               0
           Conv2d-97         [-1, 1024, 14, 14]         262,144
      BatchNorm2d-98         [-1, 1024, 14, 14]           2,048
             ReLU-99         [-1, 1024, 14, 14]               0
   IdentityBlock-100         [-1, 1024, 14, 14]               0
          Conv2d-101          [-1, 256, 14, 14]         262,144
     BatchNorm2d-102          [-1, 256, 14, 14]             512
            ReLU-103          [-1, 256, 14, 14]               0
          Conv2d-104          [-1, 256, 14, 14]         589,824
     BatchNorm2d-105          [-1, 256, 14, 14]             512
            ReLU-106          [-1, 256, 14, 14]               0
          Conv2d-107         [-1, 1024, 14, 14]         262,144
     BatchNorm2d-108         [-1, 1024, 14, 14]           2,048
            ReLU-109         [-1, 1024, 14, 14]               0
   IdentityBlock-110         [-1, 1024, 14, 14]               0
          Conv2d-111          [-1, 256, 14, 14]         262,144
     BatchNorm2d-112          [-1, 256, 14, 14]             512
            ReLU-113          [-1, 256, 14, 14]               0
          Conv2d-114          [-1, 256, 14, 14]         589,824
     BatchNorm2d-115          [-1, 256, 14, 14]             512
            ReLU-116          [-1, 256, 14, 14]               0
          Conv2d-117         [-1, 1024, 14, 14]         262,144
     BatchNorm2d-118         [-1, 1024, 14, 14]           2,048
            ReLU-119         [-1, 1024, 14, 14]               0
   IdentityBlock-120         [-1, 1024, 14, 14]               0
          Conv2d-121          [-1, 256, 14, 14]         262,144
     BatchNorm2d-122          [-1, 256, 14, 14]             512
            ReLU-123          [-1, 256, 14, 14]               0
          Conv2d-124          [-1, 256, 14, 14]         589,824
     BatchNorm2d-125          [-1, 256, 14, 14]             512
            ReLU-126          [-1, 256, 14, 14]               0
          Conv2d-127         [-1, 1024, 14, 14]         262,144
     BatchNorm2d-128         [-1, 1024, 14, 14]           2,048
            ReLU-129         [-1, 1024, 14, 14]               0
   IdentityBlock-130         [-1, 1024, 14, 14]               0
          Conv2d-131          [-1, 256, 14, 14]         262,144
     BatchNorm2d-132          [-1, 256, 14, 14]             512
            ReLU-133          [-1, 256, 14, 14]               0
          Conv2d-134          [-1, 256, 14, 14]         589,824
     BatchNorm2d-135          [-1, 256, 14, 14]             512
            ReLU-136          [-1, 256, 14, 14]               0
          Conv2d-137         [-1, 1024, 14, 14]         262,144
     BatchNorm2d-138         [-1, 1024, 14, 14]           2,048
            ReLU-139         [-1, 1024, 14, 14]               0
   IdentityBlock-140         [-1, 1024, 14, 14]               0
          Conv2d-141            [-1, 512, 7, 7]         524,288
     BatchNorm2d-142            [-1, 512, 7, 7]           1,024
            ReLU-143            [-1, 512, 7, 7]               0
          Conv2d-144            [-1, 512, 7, 7]       2,359,296
     BatchNorm2d-145            [-1, 512, 7, 7]           1,024
            ReLU-146            [-1, 512, 7, 7]               0
          Conv2d-147           [-1, 2048, 7, 7]       1,048,576
     BatchNorm2d-148           [-1, 2048, 7, 7]           4,096
          Conv2d-149           [-1, 2048, 7, 7]       2,097,152
     BatchNorm2d-150           [-1, 2048, 7, 7]           4,096
            ReLU-151           [-1, 2048, 7, 7]               0
       ConvBlock-152           [-1, 2048, 7, 7]               0
          Conv2d-153            [-1, 512, 7, 7]       1,048,576
     BatchNorm2d-154            [-1, 512, 7, 7]           1,024
            ReLU-155            [-1, 512, 7, 7]               0
          Conv2d-156            [-1, 512, 7, 7]       2,359,296
     BatchNorm2d-157            [-1, 512, 7, 7]           1,024
            ReLU-158            [-1, 512, 7, 7]               0
          Conv2d-159           [-1, 2048, 7, 7]       1,048,576
     BatchNorm2d-160           [-1, 2048, 7, 7]           4,096
            ReLU-161           [-1, 2048, 7, 7]               0
   IdentityBlock-162           [-1, 2048, 7, 7]               0
          Conv2d-163            [-1, 512, 7, 7]       1,048,576
     BatchNorm2d-164            [-1, 512, 7, 7]           1,024
            ReLU-165            [-1, 512, 7, 7]               0
          Conv2d-166            [-1, 512, 7, 7]       2,359,296
     BatchNorm2d-167            [-1, 512, 7, 7]           1,024
            ReLU-168            [-1, 512, 7, 7]               0
          Conv2d-169           [-1, 2048, 7, 7]       1,048,576
     BatchNorm2d-170           [-1, 2048, 7, 7]           4,096
            ReLU-171           [-1, 2048, 7, 7]               0
   IdentityBlock-172           [-1, 2048, 7, 7]               0
       AvgPool2d-173           [-1, 2048, 1, 1]               0
          Linear-174                    [-1, 3]           6,147
================================================================
Total params: 23,514,179
Trainable params: 23,514,179
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.57
Forward/backward pass size (MB): 270.43
Params size (MB): 89.70
Estimated Total Size (MB): 360.70
----------------------------------------------------------------

```
- 代码解读：
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/646092dab4004996819c300a8775aa46.png)
## 5.7 训练模型（$\to$ `main.py`）

```python
import copy
from solve import train, test

# (1) 定义损失函数和优化器
loss_fn = nn.CrossEntropyLoss() # 作用：定义交叉熵损失函数
optimizer = torch.optim.Adam(resnet50.parameters(), lr=1e-4) # 作用：定义Adam优化器，学习率为1e-4

# (2) 训练模型
epochs = 20

train_loss = []
train_acc = []
test_loss = []
test_acc = []
best_acc = 0.0

for epoch in range(epochs):
    resnet50.train()
    epoch_train_loss, epoch_train_acc = train(train_loader, resnet50, loss_fn, optimizer, device)

    resnet50.eval()
    epoch_test_loss, epoch_test_acc = test(test_loader, resnet50, loss_fn, device)

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
```
- 运行结果：
```
Epoch:  1, Train Loss: 0.9100, Train Acc: 64.8%, Test Loss: 1.1026, Test Acc: 63.4%, LR: 1.00E-04
Epoch:  2, Train Loss: 0.7488, Train Acc: 71.9%, Test Loss: 0.6902, Test Acc: 76.3%, LR: 1.00E-04
Epoch:  3, Train Loss: 0.6327, Train Acc: 77.0%, Test Loss: 1.0810, Test Acc: 70.0%, LR: 1.00E-04
Epoch:  4, Train Loss: 0.5607, Train Acc: 79.6%, Test Loss: 0.5373, Test Acc: 83.2%, LR: 1.00E-04
Epoch:  5, Train Loss: 0.4685, Train Acc: 82.1%, Test Loss: 0.4189, Test Acc: 85.6%, LR: 1.00E-04
Epoch:  6, Train Loss: 0.4376, Train Acc: 84.3%, Test Loss: 0.3409, Test Acc: 87.7%, LR: 1.00E-04
Epoch:  7, Train Loss: 0.4178, Train Acc: 85.2%, Test Loss: 0.9594, Test Acc: 78.7%, LR: 1.00E-04
Epoch:  8, Train Loss: 0.3463, Train Acc: 86.7%, Test Loss: 0.4043, Test Acc: 87.4%, LR: 1.00E-04
Epoch:  9, Train Loss: 0.2725, Train Acc: 90.2%, Test Loss: 0.3031, Test Acc: 89.2%, LR: 1.00E-04
Epoch: 10, Train Loss: 0.2790, Train Acc: 90.0%, Test Loss: 0.3310, Test Acc: 87.1%, LR: 1.00E-04
Epoch: 11, Train Loss: 0.2495, Train Acc: 90.7%, Test Loss: 0.2652, Test Acc: 89.8%, LR: 1.00E-04
Epoch: 12, Train Loss: 0.2308, Train Acc: 91.4%, Test Loss: 0.4461, Test Acc: 85.9%, LR: 1.00E-04
Epoch: 13, Train Loss: 0.1705, Train Acc: 94.4%, Test Loss: 0.2586, Test Acc: 91.6%, LR: 1.00E-04
Epoch: 14, Train Loss: 0.1230, Train Acc: 95.2%, Test Loss: 0.2969, Test Acc: 90.4%, LR: 1.00E-04
Epoch: 15, Train Loss: 0.1606, Train Acc: 94.2%, Test Loss: 0.2728, Test Acc: 92.2%, LR: 1.00E-04
Epoch: 16, Train Loss: 0.1245, Train Acc: 95.6%, Test Loss: 0.4865, Test Acc: 85.6%, LR: 1.00E-04
Epoch: 17, Train Loss: 0.1526, Train Acc: 94.2%, Test Loss: 0.2836, Test Acc: 91.0%, LR: 1.00E-04
Epoch: 18, Train Loss: 0.0850, Train Acc: 96.9%, Test Loss: 0.2450, Test Acc: 89.8%, LR: 1.00E-04
Epoch: 19, Train Loss: 0.0915, Train Acc: 96.6%, Test Loss: 0.2321, Test Acc: 91.3%, LR: 1.00E-04
Epoch: 20, Train Loss: 0.0820, Train Acc: 97.1%, Test Loss: 0.4366, Test Acc: 91.3%, LR: 1.00E-04
Model saved at ./model/resnet50.pth
```
## 5.8 可视化Loss和Accuracy（$\to$ `main.py`）

```python
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
```
- 运行结果：
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/dd6f7bf124b74c689e6329eeed2c36b1.png)

## 5.9 测试模型（$\to$ `main.py`）

```python
best_model.load_state_dict(torch.load(PATH, map_location=device))
epoch_test_loss, epoch_test_acc = test(test_loader, best_model, loss_fn, device)
print(f"Test Accuracy: {epoch_test_acc*100:.1f}%, Test Loss: {epoch_test_loss:.4f}")
```
- 运行结果：
```
Test Accuracy: 92.2%, Test Loss: 0.2728
```

----
==微语录：时间像一条流动着的线，让我们有了过去，也有了未来。——《薄雾》==
