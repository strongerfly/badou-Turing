#!/usr/bin/env python
# encoding=gbk

import matplotlib.pyplot as plt
import sklearn.decomposition as dp
from sklearn.datasets.base import load_iris
import numpy as np

def iris_demo():
    x,y=load_iris(return_X_y=True) #�������ݣ�x��ʾ���ݼ��е��������ݣ�y��ʾ���ݱ�ǩ
    print('����������',x.shape,'��ǩ��',y.shape)
    pca=dp.PCA(n_components=2) #����pca�㷨�����ý�ά�����ɷ���ĿΪ2
    reduced_x=pca.fit_transform(x) #��ԭʼ���ݽ��н�ά��������reduced_x��
    print(reduced_x[:4, :])
    print('���������:',pca.explained_variance_ratio_)  # ���������,ʲô���壿

    red_x,red_y=[],[]
    blue_x,blue_y=[],[]
    green_x,green_y=[],[]
    for i in range(len(reduced_x)): #���β������𽫽�ά������ݵ㱣���ڲ�ͬ�ı���
        if y[i]==0:
            red_x.append(reduced_x[i][0])
            red_y.append(reduced_x[i][1])
        elif y[i]==1:
            blue_x.append(reduced_x[i][0])
            blue_y.append(reduced_x[i][1])
        else:
            green_x.append(reduced_x[i][0])
            green_y.append(reduced_x[i][1])
    plt.scatter(red_x,red_y,c='r',marker='x')
    plt.scatter(blue_x,blue_y,c='b',marker='D')
    plt.scatter(green_x,green_y,c='g',marker='.')
    plt.show()

class PCA_NP():
    def __init__(self, n_components):
        self.n_components = n_components

    def fit_transform(self, X):
        self.n_features_ = X.shape[1] #�����ռ�ά��
        self.m = X.shape[0] #��������
        # ��Э�������
        X = X - X.mean(axis=0) #���Ļ�
        self.covariance = np.dot(X.T, X) / self.m  # 0��ֵ�����Э�������ʽ
        # ��Э������������ֵ����������
        eig_vals, eig_vectors = np.linalg.eig(self.covariance)
        print(eig_vals)
        print(eig_vectors)
        # ��ý�����������ֵ�����
        idx = np.argsort(-eig_vals) #�����ǶԳ�������������ֵһ��Ϊ������
        # ��ά����
        self.components_ = eig_vectors[:, idx[:self.n_components]]
        print(self.components_)
        # ��X���н�ά
        #print(X.shape,self.components_.shape)
        return np.dot(X, self.components_)

def my_iris_demo():
    x, y = load_iris(return_X_y=True)  # �������ݣ�x��ʾ���ݼ��е��������ݣ�y��ʾ���ݱ�ǩ
    pca =PCA_NP(n_components=2)
    reduced_x = pca.fit_transform(x)
    print(reduced_x[:4,:])#�ڶ�ά�����iris_demo�������������ģ�

    red_x, red_y = [], []
    blue_x, blue_y = [], []
    green_x, green_y = [], []
    for i in range(len(reduced_x)):  # ���β������𽫽�ά������ݵ㱣���ڲ�ͬ�ı���
        if y[i] == 0:
            red_x.append(reduced_x[i][0])
            red_y.append(reduced_x[i][1])
        elif y[i] == 1:
            blue_x.append(reduced_x[i][0])
            blue_y.append(reduced_x[i][1])
        else:
            green_x.append(reduced_x[i][0])
            green_y.append(reduced_x[i][1])
    plt.scatter(red_x, red_y, c='r', marker='x')
    plt.scatter(blue_x, blue_y, c='b', marker='D')
    plt.scatter(green_x, green_y, c='g', marker='.')
    plt.show()

if __name__ == "__main__":
    #iris_demo()
    my_iris_demo()