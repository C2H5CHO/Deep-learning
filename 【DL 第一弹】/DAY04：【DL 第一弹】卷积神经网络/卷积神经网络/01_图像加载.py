import matplotlib.pyplot as plt
import numpy as np

img1 = np.zeros([200, 200, 3])
plt.imshow(img1)
plt.show()

img2 = np.full([200, 200, 3], 255)
plt.imshow(img2)
plt.show()

img3 = np.full([200, 200, 3], 128)
plt.imshow(img3)
plt.show()

img4 = plt.imread('img.jpg')
plt.imshow(img4)
plt.show()
print(img4.shape)
