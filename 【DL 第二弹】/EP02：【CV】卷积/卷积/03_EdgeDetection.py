import cv2
import numpy as np
from matplotlib import pyplot as plt

img2 = cv2.imread('pictures/edge detection.PNG', cv2.COLOR_BGR2GRAY) # 将原图片读取为灰度图

plt.figure(dpi=300)
plt.imshow(img2, cmap='gray') # 显示灰度图
plt.axis('off')
plt.show()

# 1. Sobel算子
sobelx = cv2.Sobel(img2, cv2.CV_64F, 1, 0, ksize=5) # x方向上的Sobel算子
plt.figure(dpi=300)
plt.imshow(sobelx, cmap='gray') # 显示x方向上的Sobel算子
plt.axis('off')
plt.show()

sobely = cv2.Sobel(img2, cv2.CV_64F, 0, 1, ksize=5) # y方向上的Sobel算子
plt.figure(dpi=300)
plt.imshow(sobely, cmap='gray') # 显示y方向上的Sobel算子
plt.axis('off')
plt.show()

# 2. Laplacian算子
laplacian = cv2.Laplacian(img2, cv2.CV_64F, ksize=5) # Laplacian算子
plt.figure(dpi=300)
plt.imshow(laplacian, cmap='gray') # 显示Laplacian算子
plt.axis('off')
plt.show()

# 3. Sharpness算子
sharpness = cv2.filter2D(img2, -1, np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])) # Sharpness算子
plt.figure(dpi=300)
plt.imshow(sharpness, cmap='gray') # 显示Sharpness算子
plt.axis('off')
plt.show()

# 4. box blur算子
box_blur = cv2.blur(img2, (5, 5)) # box blur算子
plt.figure(dpi=300)
plt.imshow(box_blur, cmap='gray') # 显示box blur算子
plt.axis('off')
plt.show()
