# -*- coding: utf-8 -*-
"""

@author: wuming
"""
# -*- coding: utf-8 -*-

import numpy as np

class PCA():
    def __init__(self,input_X,out_n):
        self.input_X=input_X
        self.out_n=out_n
        self.centX=[]#输入的样本矩阵
        self.Cov=[]#协方差矩阵
        self.outK=[]#降维后的矩阵
        self.centX=self._centralized()
        self.Cov=self._cov()
        self.outK=self._out_array()


    def _centralized(self):#中心化
        centX=self.input_X-np.mean(self.input_X,axis=0)
        return centX
    def _cov(self):#协方差函数
        Cov=np.dot(self.centX.T,self.centX)/(self.centX.shape[0]-1)
        return Cov
    def _out_array(self):#降维矩阵
        eig_vals,eig_vectors=np.linalg.eig(self.Cov)
        idx=np.argsort(-eig_vals)
        W=eig_vectors[:,idx[:self.out_n]]
        outK=np.dot(self.centX,W)#中心化后的矩阵用于计算
        print(outK)
        return  outK

if __name__=='__main__':
    # X = np.array([[10, 15, 29],
    #               [15, 46, 13],
    #               [23, 21, 30],
    #               [11, 9, 35],
    #               [42, 45, 11],
    #               [9, 48, 5],
    #               [11, 21, 14],
    #               [8, 5, 15],
    #               [11, 12, 21],
    #               [21, 20, 25]])
    X = np.array([[-1, 2, 66, -1], [-2, 6, 58, -1], [-3, 8, 45, -2], [1, 9, 36, 1], [2, 10, 62, 1], [3, 5, 83, 2]])
    K = np.shape(X)[1] - 2
    pca=PCA(X,K)
