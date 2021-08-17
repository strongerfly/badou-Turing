import cv2
import numpy as np
from matplotlib import pyplot as plt
img = cv2.imread('lenna.png')
img2 = img.copy()

'''
注意这里src和dst的输入并不是图像，而是图像对应的顶点坐标。
'''
src = np.float32([[256, 0],[256, 512], [512, 256],[0, 256]])
dst = np.float32([[0, 0], [362, 362], [362, 0], [0, 362]])
print(img.shape)
# 生成透视变换矩阵；进行透视变换
m = cv2.getPerspectiveTransform(src, dst)
print("warpMatrix:")
print(m)
result = cv2.warpPerspective(img2, m, (360, 360))
plt.subplot(121)
plt.imshow(img, "gray"), plt.title("src")
plt.subplot(122)
plt.imshow(result, "gray"), plt.title("dst")
plt.show()

