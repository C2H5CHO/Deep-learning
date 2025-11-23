import torch
import torch.nn as nn

# 1. 定义自动填充函数
def autopad(k, p=None):
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k] # 作用：计算填充大小
    return p

# 2. 定义残差块类
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

# 4. 定义训练函数
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

# 5. 定义测试函数
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

