import random
import math
import matplotlib.pyplot as plt


class KMeans:
    def __init__(self, n_clusters):
        self.n_clusters = n_clusters
        self.centroid_list = []
        self.predict = []

    def get_rand_centroid(self, X):
        # 随机取质心
        centroid_list = []
        while len(centroid_list) < self.n_clusters:
            d = int(random.random() * len(X))
            if X[d] not in centroid_list:
                centroid_list.append(X[d])
        return centroid_list

    @staticmethod
    def get_distance(point, C):
        # 计算两点间距离(欧式距离)
        return math.sqrt((point[0]-C[0])**2 + (point[1]-C[1])**2)

    def get_distributed(self, X):
        # 计算每个点距离最近的质心，并将该点划入该质心所在的簇
        dis_list = [[] for k in range(self.n_clusters)]
        for point in X:
            distance_list = []
            for C in self.centroid_list:
                distance_list.append(self.get_distance(point, C))
            min_index = distance_list.index(min(distance_list))
            dis_list[min_index].append(point)
        return dis_list

    @staticmethod
    def get_virtual_centroid(distributed):
        # 计算每个簇所有点的坐标的算数平均，作为虚拟质心
        v_centroid_list = []
        for distribution in distributed:
            x = []
            y = []
            for point in distribution:
                x.append(point[0])
                y.append(point[1])
            v_centroid_list.append([sum(x)/len(x), sum(y)/len(y)])
        return v_centroid_list

    def fit_predict(self, X):
        self.centroid_list = self.get_rand_centroid(X)
        while True:
            # 聚类
            distributed = self.get_distributed(X)
            # 计算虚拟质心
            v_centroid_list = self.get_virtual_centroid(distributed)
            # 如果两次质心相同，说明聚类结果已定
            if sorted(v_centroid_list) == sorted(self.centroid_list):
                break
            # 否则继续训练
            self.centroid_list = v_centroid_list
        # 对结果按照数据集顺序进行分类
        predict = []
        for point in X:
            i = 0
            for dis in distributed:
                if point in dis:
                    predict.append(i)
                i += 1
        self.predict = predict
        return predict

    def plot_clustering(self, X):
        x = []
        y = []
        for point in X:
            x.append(point[0])
            y.append(point[1])
        plt.scatter(x, y, c=self.predict, marker='x')
        plt.show()


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

# 初始化kmeans分类器
km = KMeans(3)
# 预测
predict = km.fit_predict(X)
print(predict)
# 绘图
km.plot_clustering(X)
