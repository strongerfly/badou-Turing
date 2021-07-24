# coding: utf-8
import numpy as np
import cv2
import random
import  copy
'''
该文件实现kmeans聚类，质心随机选取
'''
class KMeans(object):
    def __init__(self,k=2,max_iter=20,epsilon=1e-3):
        self.K=k#聚成多少类
        self.max_iter=max_iter#最大的迭代次数
        self.epsilon=epsilon#停止迭代条件之一的精度
        self.centers={}#存储聚类中心，每类一行
        self.labels=[]#聚类标签的索引（0,1,2...）

    def Dokmeans(self,data):
        #1第一步随机选择K个初始化质心
        centersIdx=random.sample(range(1,data.shape[0]),self.K)#先随机选取k个索引
        self.centers={k:data[centersIdx[k]] for k in range(self.K)}

        #2.进行迭代,
        for  iter in  range(self.max_iter):
            Nclass={}
            self.labels=[]
            #2.1计算每个样本到各质心向量u的距离，并进行分类
            for j in range(data.shape[0]):
                mindis=10000000
                index_=-1
                for i in range(self.K):#
                    dis=np.linalg.norm(data[j]-self.centers[i])
                    if dis<mindis:
                        mindis=dis
                        index_=i#记录距离最小的标记属于哪个类簇，这里的标记也可以直接先把dis放到列表里，然后用列表的index函数求得索引
                #将数据归属到满足条件的类别
                if index_ not in Nclass and index_!=-1:
                    Nclass[index_]=[]
                    Nclass[index_].append(data[j])
                    self.labels.append(index_)#将该点的标签记录
                elif(index_!=-1):
                    Nclass[index_].append(data[j])
                    self.labels.append(index_)
                else:
                    print("index is -1,data may be no correct")
                    pass
            #2.2在得到样本的归类后，需要更新质心
            old_centers=self.centers.copy()#作拷贝，不能直接赋值，否则值会跟着改变
            for i in range(self.K):
                self.centers[i]=np.array(Nclass[i]).mean(0)
            bstop=True
            old_centers_flatten=np.array([old_centers[i] for i in range(self.K)]).flatten()
            new_centers_flatten=np.array([self.centers[i] for i in range(self.K)]).flatten()

            if abs(old_centers_flatten-new_centers_flatten).sum()>self.epsilon:
                print("iter:{},diff={}".format(iter,abs(old_centers_flatten-new_centers_flatten).sum()))
                bstop=False
            if bstop==True:
                break

if __name__=='__main__':
    img=cv2.imread('lenna.png')
    h,w,c=img.shape
    K=2#划分2类
    kmeans=KMeans(k=K,max_iter=20,epsilon=1e-2)
    data=img.reshape((-1,3))
    data = np.float32(data)
    kmeans.Dokmeans(data)
    labels=np.array(kmeans.labels)#返回每个样本的标签索引
    centers=np.array([kmeans.centers[i] for i in range(K)]).astype(np.uint8)#集群中心矩阵
    print("centers.shape:",centers.shape)
    res = centers[labels.flatten()]
    print("res:", res.shape)  # shape:[262144,3]
    dst = res.reshape((img.shape))
    showimg = np.zeros((h, 2 * w, c), dtype=np.uint8)
    showimg[:, 0:w] = img
    showimg[:, w:2 * w] = dst
   # cv2.namedWindow("result", cv2.WINDOW_NORMAL)
    cv2.imshow("result", showimg)
    cv2.waitKey()











