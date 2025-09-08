import torch
import re
import jieba
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time

# 1. 构建词表
def build_vocab(file_path):
    # (1) 读取数据
    for line in open('./data/jaychou_lyrics.txt', 'r'):
        pass