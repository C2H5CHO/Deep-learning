import numpy as np
import cv2
import matplotlib.pyplot as plt

# 1. 读取图片
img_1 = cv2.imread('pictures/blue-peacock.jpg')

# 2. 图片格式转换
img_1 = cv2.cvtColor(img_1, cv2.COLOR_BGR2RGB) # BGR转RGB

# 3. 显示图片
plt.figure(dpi=150) # 设置分辨率为150
plt.imshow(img_1)
plt.axis('off') # 关闭坐标轴
plt.show()
