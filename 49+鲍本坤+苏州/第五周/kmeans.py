# coding: utf-8
import cv2
import numpy as np
import matplotlib.pyplot as plt

'''
retval, bestLabels, centers = kmeans(data, K, bestLabels, criteria, attempts, flags, centers=None)
data:  需要分类数据，最好是np.float32的数据，每个特征放一列。

K:  聚类个数 

bestLabels：预设的分类标签或者None

criteria：迭代停止的模式选择，这是一个含有三个元素的元组型数。格式为（type, max_iter, epsilon） 其中，type有如下模式：

cv2.TERM_CRITERIA_EPS ：精确度（误差）满足epsilon，则停止。
cv2.TERM_CRITERIA_MAX_ITER：迭代次数超过max_iter，则停止。
cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER：两者结合，满足任意一个结束。
attempts：重复试验kmeans算法次数，将会返回最好的一次结果

flags：初始中心选择，可选以下两种：

v2.KMEANS_PP_CENTERS：使用kmeans++算法的中心初始化算法，即初始中心的选择使眼色相差最大.详细可查阅kmeans++算法。(Use kmeans++ center initialization by Arthur and Vassilvitskii)
cv2.KMEANS_RANDOM_CENTERS：每次随机选择初始中心（Select random initial centers in each attempt.）

返回值：

compactness：紧密度，返回每个点到相应重心的距离的平方和

labels：结果标记，每个成员被标记为分组的序号，如 0,1,2,3,4...等

centers：由聚类的中心组成的数组
'''

img = cv2.imread('ddk.jpg')
img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#图片太大，需要调小
img = cv2.resize(img,(800,500))
cv2.imshow('pic',img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
#获取图像高度、宽度
rows, cols = img.shape[:]
#二维图像转一维
data=img.reshape(rows*cols,1)
data=np.float32(data)


#停止条件，满足精度或者迭代次数均可
criteria = (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER,10,0.1)

#设置标签
flags= cv2.KMEANS_PP_CENTERS
#聚类
compactness,labels,centers = cv2.kmeans(data,4,None,criteria,10,flags)

dst = labels.reshape((img.shape[0],img.shape[1]))

plt.rcParams['font.sans-serif']=['SimHei']

titles=[u'原始图像',u'聚类图像']
images=[img,dst]
for i in range(2):
    plt.subplot(1,2,i+1),plt.imshow(images[i]),
    plt.title(titles[i])
    plt.xticks([]),plt.yticks([])
plt.show()



