import torch
import torch.nn as nn
import jieba

# 1. 分词
text = "北京东奥的进度条已经过半，不少外国运动员在完成自己的比赛后踏上归途。"
words = jieba.lcut(text)
print(words)

print('--'*50)
# 2. 去重
unique_words = list(set(words))
print(unique_words)
num_words = len(unique_words)
print(num_words)

print('--'*50)
# 3. 创建词嵌入层
embedding = nn.Embedding(num_embeddings=num_words, embedding_dim=3)
print(embedding)

for i, word in enumerate(unique_words):
    print(word)
    print(embedding(torch.tensor(i)))

