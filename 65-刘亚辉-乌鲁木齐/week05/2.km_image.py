import random
import numpy as np
import matplotlib.pyplot as plt
import cv2


class KMImage:
    def __init__(self, n_clusters):
        self.n_clusters = n_clusters
        self.centroid_list = []
        self.distributed = ''
        self.km_image = ''

    def get_rand_centroid(self, image):
        centroid_list = []
        while len(centroid_list) < self.n_clusters:
            [x, y] = [int(random.random() * image.shape[0]), int(random.random() * image.shape[1])]
            if [x, y] not in centroid_list:
                centroid_list.append([x, y])
        return centroid_list

    @staticmethod
    def get_instance(image, pixel, C):
        # 将距离定义为像素值差的绝对值
        row, col = pixel
        c_row, c_col = C
        return abs(image[row, col] - image[c_row, c_col])

    def get_distributed(self, image):
        dis_list = [[] for k in range(self.n_clusters)]
        for row in range(image.shape[0]):
            for col in range(image.shape[1]):
                distance_list = []
                for C in self.centroid_list:
                    distance_list.append(self.get_instance(image, [row, col], C))
                min_index = distance_list.index(min(distance_list))
                dis_list[min_index].append([row, col])
        return dis_list

    def get_virtual_centroid(self, distributed):
        # 如果有空集，则取其他两个质心的坐标均值
        if [] in distributed:
            index = distributed.index([])
            v_centroid_list = self.centroid_list.copy()
            # 去除空集对应的质心
            v_centroid_list.pop(index)
            # 计算其余两个质心的坐标均值
            x = []
            y = []
            for C in v_centroid_list:
                x.append(C[0])
                y.append(C[1])
            v_centroid_list.insert(index, [round(sum(x)/len(x)), round(sum(y)/len(y))])
        else:
            v_centroid_list = []
            for dis in distributed:
                x = []
                y = []
                for pixel in dis:
                    x.append(pixel[0])
                    y.append(pixel[1])
                v_centroid_list.append([int(sum(x)/len(x)), int(sum(y)/len(y))])
        return v_centroid_list

    def plot_image(self, image, j):
        # 分类
        km_image = image.copy()

        for i in range(len(self.distributed)):
            c_row, c_col = self.centroid_list[i]
            for pixel in self.distributed[i]:
                km_image[pixel[0], pixel[1]] = image[c_row, c_col]
        # 用排过序的质心坐标命名图像
        centroid_name = ''
        for row, col in sorted(self.centroid_list):
            centroid_name += str(row) + '_' + str(col) + '_'
        print(centroid_name)
        cv2.imwrite('images/{}_{}.png'.format(str(j), centroid_name[-1]), km_image)
        return km_image

    def fit_predict(self, image, r=255):
        # 默认训练255轮
        self.centroid_list = self.get_rand_centroid(image)
        j = 0
        while j < r:
            # 聚类
            distributed = self.get_distributed(image)
            self.distributed = distributed
            # 重新划分质心
            v_centroid_list = self.get_virtual_centroid(distributed)
            if sorted(v_centroid_list) == sorted(self.centroid_list):
                self.distributed = distributed
                break
            self.centroid_list = v_centroid_list
            j += 1
            # 每10轮输出一副图像
            if j % 10 == 0:
                self.plot_image(image, j)
        # 训练结束输出一张
        self.plot_image(image, 'final')


image = cv2.imread('lenna.png')
image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
km = KMImage(4)
km.fit_predict(image_gray)
