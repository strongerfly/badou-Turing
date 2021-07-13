import cv2
import numpy as np
from matplotlib import pyplot as plt
#flag默认为1，即读取为彩色图像，如果要读取为灰度图（单通道）则为0
img = cv2.imread('lenna.png')
h,w = img.shape[:2]
#flatten()将数组变为一维
#hist 每个灰度值的频数  256 表示横坐标的最大值为256，有256条柱 [0,256]表示数据显示范围
hist,bins=np.histogram(img.flatten(),256,[0,256])

#计算累计分布  cumsum;【1,2,3,4】—>【1,3,6,10】 累加
cdf =hist.cumsum()
#因为累加最后量过大，直方图中无法显示，因为做以下除法，保证在一张图上
#均衡化
cdf_normalized=cdf*256/(h*w)-1           #cdf*hist.max()/cdf.max()
plt.plot(cdf_normalized,color = 'b')
#plt.hist画直方图（不是累积直方图）
plt.hist(img.flatten(),256,[0,256],color='r')
plt.xlim([0,256])
plt.legend(('cdf','histogram'), loc = 'upper left')
# plt.legend(('cdf','histogram'), loc = 'upper left')
plt.show()



gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow("image_gray", gray)
# 灰度图像直方图均衡化
dst = cv2.equalizeHist(gray)
# 直方图
hist = cv2.calcHist([dst],[0],None,[256],[0,256])
plt.figure()
plt.hist(dst.ravel(), 256)
plt.show()



(b,g,r)= cv2.split(img)
bh = cv2.equalizeHist(b)
gh = cv2.equalizeHist(g)
rh = cv2.equalizeHist(r)
result = cv2.merge((bh,gh,rh))
res1 = np.hstack((img,result))  #水平拼接两幅图
cv2.imshow('dst',res1)  #给展示的图片命名为dst
cv2.waitKey()