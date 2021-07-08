import numpy as np
import cv2
# from cv2 import *
from matplotlib import pyplot as plt
img=cv2.imread("./lenna.png")
gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#cv2.imshow("",gray)
#cv2.waitKey()
# 灰度图像的直方图, 方法一
plt.figure()
plt.hist(gray.ravel(),256)
plt.show()


# 灰度图像的直方图, 方法二
hist = cv2.calcHist([gray],[0],None,[256],[0,256])
plt.figure()#新建一个图像
plt.title("Grayscale Histogram")
plt.xlabel("Bins")#X轴标签
plt.ylabel("# of Pixels")#Y轴标签
plt.plot(hist)
plt.xlim([0,256])#设置x坐标轴范围
plt.show()


# 彩色图像直方图
chans=cv2.split(img)
colors=("b","g","r")
plt.figure()
plt.title="flattened color histogram"
plt.xlabel("Bins")
plt.ylabel("# of Pixels")
for (chan,c) in zip(chans,colors):

    #  方法一
    plt.hist(np.array(chan).flatten(),256,color=c)
    plt.xlabel("Bins")
    plt.ylabel("y")

    # 方法二
    # hist = cv2.calcHist([chan], [0], None, [256], [0, 256])
    # plt.plot(hist, color=c)
    # plt.xlim([0, 256])
plt.show()