from skimage.color import rgb2gray
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
path = '/Users/snszz/PycharmProjects/CV/第二周/代码/lenna.png'
img = cv2.imread(path)
print(img)
print(img.shape)   # 512 * 512 * 3
print(type(img))  # numpy.ndarray

# method 1
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# 画图
plt.imshow(gray, cmap='gray')
plt.title('lenna_gray')
# cv2.imshow(gray)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# method2
gray = rgb2gray(img)
cv2.imwrite('lenna.png', gray)
# plt.imsave('lenna.png', gray)


