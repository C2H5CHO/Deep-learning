# 1. 工具包
import torch
from torch.utils.data import TensorDataset # 导入PyTorch的TensorDataset类，用于将数据封装成数据集
from torch.utils.data import DataLoader    # 导入PyTorch的DataLoader类，用于批量加载数据
import torch.nn as nn                      # 导入PyTorch的神经网络模块
import torch.optim as optim                # 导入PyTorch的优化器模块
from sklearn.datasets import make_regression # 导入sklearn生成回归数据集的函数
from sklearn.model_selection import train_test_split # 导入sklearn的数据集分割函数
import matplotlib.pyplot as plt            # 导入matplotlib绘图库
import numpy as np                         # 导入numpy数值计算库
import pandas as pd                        # 导入pandas数据分析库
import time                                # 导入时间处理模块

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

