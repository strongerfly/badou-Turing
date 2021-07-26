import cv2
import numpy as np
import matplotlib.pyplot as plt

# plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']  # 用来正常显示中文标签 MAC下
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

img = cv2.imread('./img/lenna.png', 0)
print(img.shape)

# 获取图像高度、宽度
rows, cols = img.shape[:]
print(rows, cols)

# 图像二维像素转换为一维
data = img.reshape((rows * cols, 1))
data = np.float32(data)
print(data)

# 停止条件 TERM_CRITERIA_EPS满足精度条件停止、TERM_CRITERIA_MAX_ITER 迭代次数超过最大阈值停止
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

flags = cv2.KMEANS_RANDOM_CENTERS

comt, labels, centers = cv2.kmeans(data, 4, None, criteria, 10, flags)
print(comt, labels, centers)

dst = labels.reshape((img.shape[0], img.shape[1]))

# 显示图像
titles = [u'原始图像', u'聚类图像']
images = [img, dst]
for i in range(2):
    plt.subplot(1, 2, i + 1), plt.imshow(images[i], 'gray'),
    plt.title(titles[i])
    plt.xticks([]), plt.yticks([])
plt.show()
