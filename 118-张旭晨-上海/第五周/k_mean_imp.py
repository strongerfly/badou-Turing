# -*- coding: utf-8 -*-


import cv2
import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans


def opencvkmeans(img):
    # 获取图像高度、宽度
    rows, cols = img.shape[:]
    # 图像二维像素转换为一维
    data = img.reshape((rows * cols, 1))
    data = np.float32(data)
    # 停止条件 (type,max_iter,epsilon)
    criteria = (cv2.TERM_CRITERIA_EPS +
                cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    # 设置标签
    flags = cv2.KMEANS_RANDOM_CENTERS
    # K-Means聚类 聚集成4类
    compactness, labels, centers = cv2.kmeans(data, 4, None, criteria, 10, flags)
    # 生成最终图像
    return labels.reshape((img.shape[0], img.shape[1]))


def sklearnkmean():
    """
    第一部分：数据集
    X表示二维矩阵数据，篮球运动员比赛数据
    总共20行，每行两列数据
    第一列表示球员每分钟助攻数：assists_per_minute
    第二列表示球员每分钟得分数：points_per_minute
    """
    X = [[0.0888, 0.5885],
         [0.1399, 0.8291],
         [0.0747, 0.4974],
         [0.0983, 0.5772],
         [0.1276, 0.5703],
         [0.1671, 0.5835],
         [0.1306, 0.5276],
         [0.1061, 0.5523],
         [0.2446, 0.4007],
         [0.1670, 0.4770],
         [0.2485, 0.4313],
         [0.1227, 0.4909],
         [0.1240, 0.5668],
         [0.1461, 0.5113],
         [0.2315, 0.3788],
         [0.0494, 0.5590],
         [0.1107, 0.4799],
         [0.1121, 0.5735],
         [0.1007, 0.6318],
         [0.2567, 0.4326],
         [0.1956, 0.4280]
         ]
    """
    第二部分：KMeans聚类
    clf = KMeans(n_clusters=3) 表示类簇数为3，聚成3类数据，clf即赋值为KMeans
    y_pred = clf.fit_predict(X) 载入数据集X，并且将聚类的结果赋值给y_pred
    """

    clf = KMeans(n_clusters=3)
    y_pred = clf.fit_predict(X)
    # 输出完整Kmeans函数，包括很多省略参数
    # print(clf)
    """
    第三部分：可视化绘图
    """
    # 获取数据集的第一列和第二列数据 使用for循环获取 n[0]表示X第一列
    ax,ay,bx,by,cx,cy= [],[],[],[],[],[]
    for i in range(len(X)):
        if y_pred[i] == 0:
            ax.append(X[i][0])
            ay.append(X[i][1])
        if y_pred[i] == 1:
            bx.append(X[i][0])
            by.append(X[i][1])
        if y_pred[i] == 2:
            cx.append(X[i][0])
            cy.append(X[i][1])
    # print(x)
    # y = [n[1] for n in X]
    # print(y)
    ''' 
    绘制散点图 
    参数：x横轴; y纵轴; c=y_pred聚类预测结果; marker类型:o表示圆点,*表示星型,x表示点;
    '''
    plt.scatter(ax, ay, marker='x', color='r')
    plt.scatter(bx, by, marker='o', color='g')
    plt.scatter(cx, cy, marker='*', color='b')
    # 绘制标题
    plt.title("Kmeans-Basketball Data")
    # 绘制x轴和y轴坐标
    plt.xlabel("assists_per_minute")
    plt.ylabel("points_per_minute")

    # 设置右上角图例
    plt.legend(['A','B','C'])

    # 显示图形
    plt.show()


def rgbKmeans():
    img = cv2.imread('lenna.jpg')
    # 一维转化
    data = img.reshape((-1, 3))
    data = np.float32(data)
    # 停止条件
    criteria = (cv2.TERM_CRITERIA_EPS +
                cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    # 初选质心
    flags = cv2.KMEANS_RANDOM_CENTERS
    # 数据，K类，最好的标签，评价条件，最大循环次数，初选质心模式
    ompactness2, labels2, centers2 = cv2.kmeans(data, 2, None, criteria, 10, flags)
    ompactness4, labels4, centers4 = cv2.kmeans(data, 4, None, criteria, 10, flags)
    ompactness6, labels6, centers6 = cv2.kmeans(data, 8, None, criteria, 10, flags)
    ompactness8, labels8, centers8 = cv2.kmeans(data, 16, None, criteria, 10, flags)
    ompactness10, labels10, centers10 = cv2.kmeans(data, 64, None, criteria, 10, flags)

    centers2 = np.uint8(centers2)
    res2 = centers2[labels2.flatten()]
    kmeans2 = res2.reshape(img.shape)

    centers4 = np.uint8(centers4)
    res4 = centers4[labels4.flatten()]
    kmeans4 = res4.reshape(img.shape)

    centers6 = np.uint8(centers6)
    res6 = centers6[labels6.flatten()]
    kmeans6 = res6.reshape(img.shape)

    centers8 = np.uint8(centers8)
    res8 = centers8[labels8.flatten()]
    kmeans8 = res8.reshape(img.shape)

    centers10 = np.uint8(centers10)
    res10 = centers10[labels10.flatten()]
    kmeans10 = res10.reshape(img.shape)

    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    dst2 = cv2.cvtColor(kmeans2, cv2.COLOR_RGB2BGR)
    dst4 = cv2.cvtColor(kmeans4, cv2.COLOR_RGB2BGR)
    dst6 = cv2.cvtColor(kmeans6, cv2.COLOR_RGB2BGR)
    dst8 = cv2.cvtColor(kmeans8, cv2.COLOR_RGB2BGR)
    dst10 = cv2.cvtColor(kmeans10, cv2.COLOR_RGB2BGR)
    plt.rcParams['font.sans-serif'] = ['SimHei']
    titles = [u'原始图像', u'聚类图像 K=2', u'聚类图像 K=4', u'聚类图像 K=6', u'聚类图像 K=8',  u'聚类图像 K=10']
    images = [img, dst2, dst4, dst6, dst8, dst10]
    for i in range(5):
        plt.subplot(2, 3, i + 1), plt.imshow(images[i], 'gray'),
        plt.title(titles[i])
        plt.xticks([]), plt.yticks([])
    plt.show()


if __name__ == '__main__':
    # 读取原始图像灰度颜色
    # img = cv2.imread('lenna.jpg', 0)
    # dst = opencvkmeans(img)
    # # 用来正常显示中文标签
    # plt.rcParams['font.sans-serif'] = ['SimHei']
    #
    # # 显示图像
    # titles = [u'原始图像', u'聚类图像']
    # images = [img, dst]
    # for i in range(2):
    #     plt.subplot(1, 2, i + 1), plt.imshow(images[i], 'gray'),
    #     plt.title(titles[i])
    #     plt.xticks([]), plt.yticks([])
    # plt.show()
    # sklearnkmean()
    rgbKmeans()