from torchvision.datasets import CIFAR10
from torchvision.transforms import Compose, ToTensor
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torchsummary import summary
from torch.utils.data import DataLoader

# 1. 数据获取
train_dataset = CIFAR10(root='./data', train=True, transform=Compose([ToTensor()]), download=True)
test_dataset = CIFAR10(root='./data', train=False, transform=Compose([ToTensor()]), download=True)

print(f"训练数据集的维度为：{train_dataset.data.shape}")
print(f"测试数据集的维度为：{test_dataset.data.shape}")
print(f"训练数据集的类别为：{train_dataset.classes}")
print(f"训练数据集的类别索引为：{train_dataset.class_to_idx}")

plt.imshow(train_dataset.data[100])
print(train_dataset.targets[100])
plt.show()

print('--'*50)
# 2. 模型构建
class imgClassification(nn.Module):
    # 模型初始化
    def __init__(self):
        super(imgClassification, self).__init__()
        self.layer1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=3, stride=1, padding=0)
        self.pooling1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.layer2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=3, stride=1, padding=0)
        self.pooling2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.layer3 = nn.Linear(in_features=576, out_features=120)
        self.layer4 = nn.Linear(in_features=120, out_features=84)
        self.out = nn.Linear(in_features=84, out_features=10)

    # 模型前向传播
    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = self.pooling1(x)
        x = torch.relu(self.layer2(x))
        x = self.pooling2(x)
        x = torch.reshape(x, [x.shape[0], -1])
        x = torch.relu(self.layer3(x))
        x = torch.relu(self.layer4(x))
        out = self.out(x)

        return out

model = imgClassification()
summary(model, input_size=(3, 32, 32), batch_size=1, device='cpu')

print('--'*50)
# 3. 模型训练
def train():
    # 损失函数
    cri = nn.CrossEntropyLoss()
    # 优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.99))
    epochs = 10
    loss_mean = []
    for epoch in range(epochs):
        dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True)
        loss_sum = 0
        samples = 0.1
        for x, y in dataloader:
            y_pred = model(x)
            loss = cri(y_pred, y)
            loss_sum += loss.item()
            samples += 1
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # break
        loss_mean.append(loss_sum/samples)
        print(f"{loss_sum/samples}")
    print(f"{loss_mean}")

    torch.save(model.state_dict(), 'imgClassification_model.pth')

train()

print('--'*50)
# 4. 模型预测
def test():
    dataloader = DataLoader(test_dataset,batch_size=8,shuffle=False)
    # 加载模型
    model.load_state_dict(torch.load('imgClassification_model.pth'))

    # 遍历数据进行预测
    correct=0
    samples = 0
    for x,y in dataloader:
        y_predict = model(x)
        correct += (torch.argmax(y_predict,dim=-1)==y).sum()
        samples += len(y)
    acc = correct/(samples+0.000001)
    print(acc)

test()

