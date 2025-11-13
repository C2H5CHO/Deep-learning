import matplotlib.pyplot as plt
import numpy as np
import cv2

img_1 = cv2.imread('pictures/blue-peacock.jpg', cv2.IMREAD_GRAYSCALE)
# 图像的数值类型
print(f"img_1.dtype: {img_1.dtype}")
# 图像的数值范围
print(f"img_1.min: {img_1.min()}")
print(f"img_1.max: {img_1.max()}")

print('--'*50)
# unit8数值特点
unit8_test = np.array([0, 1, 255], dtype=np.uint8)
print(f"unit8_test: {unit8_test}")
print(f"unit8_test + 10: {unit8_test + 10}")
print(f"unit8_test - 10: {unit8_test - 10}")

print('--'*50)
# 图像的线性变换
# (1) 乘以1.0
img_1_1 = img_1*1.0
print(f"img_1_1: {img_1_1}")
print(f"img_1_1.dtype: {img_1_1.dtype}")
# (2) 除以255
img_1_2 = img_1/255
print(f"img_1_2: {img_1_2}")
print(f"img_1_2.dtype: {img_1_2.dtype}")

print('--'*50)
# 图像的亮度变换
img_1 = img_1/255
# (1) 增加亮度
img_1_light = np.clip(img_1 + 100/255, 0, 1)
plt.figure(dpi=150)
plt.imshow(img_1_light)
plt.axis('off')
plt.show()
# (2) 减少亮度
img_1_dark = np.clip(img_1 - 100/255, 0, 1)
plt.figure(dpi=150)
plt.imshow(img_1_dark)
plt.axis('off')
plt.show()
# (3) 亮度对比度增强
img_1_contrast = np.clip(img_1*1.5, 0, 1)
plt.figure(dpi=150)
plt.imshow(img_1_contrast)
plt.axis('off')
plt.show()

print('--'*50)
# 图像的hsv变换
img_1_hsv = cv2.imread('pictures/blue-peacock.jpg', cv2.COLOR_BGR2HSV)
print(f"img_1_hsv: {img_1_hsv}")
# (1) 增加色相
img_1_hsv_h = img_1_hsv.copy()
img_1_hsv_h[:, :, 0] = img_1_hsv_h[:, :, 0] + 100
img_1_hsv_h = cv2.cvtColor(img_1_hsv_h, cv2.COLOR_HSV2BGR)
plt.figure(dpi=150)
plt.imshow(img_1_hsv_h)
plt.axis('off')
plt.show()
# (2) 增加饱和度
img_1_hsv_s = img_1_hsv.copy()
img_1_hsv_s[:, :, 1] = img_1_hsv_s[:, :, 1] + 100
img_1_hsv_s = cv2.cvtColor(img_1_hsv_s, cv2.COLOR_HSV2BGR)
plt.figure(dpi=150)
plt.imshow(img_1_hsv_s)
plt.axis('off')
plt.show()
# (3) 增加亮度
img_1_hsv_v = img_1_hsv.copy()
img_1_hsv_v[:, :, 2] = img_1_hsv_v[:, :, 2] + 100
img_1_hsv_v = cv2.cvtColor(img_1_hsv_v, cv2.COLOR_HSV2BGR)
plt.figure(dpi=150)
plt.imshow(img_1_hsv_v)
plt.axis('off')
plt.show()

