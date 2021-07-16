#-*- coding: utf-8 -*-
import numpy as np
import math


'''
pca实现：
1.对原始数据进行零均值化处理
2.求协方差矩阵
3.对协方差矩阵求特征向量和特征值
4.对特征值从大到小排序，选择前k个最大值对应的特征向量构成矩阵P
5.将数据用P矩阵转换到新空间：Y=PX

'''
class PCA(object):
    def __init__(self,K=2,ratio=0):
        '''
        样本矩阵：data，
        :k 保留前K阶对应的特征,
        :ratio 根据保留方差的百分比计算得出K,K和ratio指定其中一个，如果ratio>0,则使用ratio计算出的K
        '''
        self.K=K
        self.ratio=ratio#通过方差百分比进行确定保留多少维度

    def ratio2k(self,eigvals):
        '''
        根据百分比确定K
        '''
        sorteigvals=np.sort(eigvals)[::-1]
        vals_sum=np.sum(sorteigvals)#特征值的和
        k,tmp,topk=0,0,0
        for j in sorteigvals:
            tmp+=j
            k+=1
            if tmp>=vals_sum*self.ratio:#大于一定百分比返回k
                topk=k
                return topk
        return topk

    def norm_feature(self,data):
        '''
        样本中心化处理
        '''
        Mean=np.mean(data,axis=0)#按列处理
        Var=np.std(data,axis=0)
        data_norm=(data-Mean)/Var
        return  data_norm

    def transform(self,data):
        #1.对数据进行零均值化处理
        data_norm=self.norm_feature(data)#[4,5]
        #2.求协方差矩阵cov=1/m*ZTZ
        cov=np.dot(data_norm.T,data_norm)/(data_norm.shape[0]-1)
        #3求协方差矩阵的特征向量(eigvectors)和特征值(eigvalues)，貌似np.linalg.eigh已经对特征值做了从小到大的排序输出
        eigvalues,eigvectors=np.linalg.eigh(cov)
        topK=0
        if self.ratio>0:
            topK=self.ratio2k(eigvalues.copy())
            self.K=topK
        #对特征值进行排序
        idx=eigvalues.argsort()[::-1]
        # 降维矩阵,取前k个特征值对应的特征向量
        P_mat = eigvectors[:, idx[:self.K]]#[5,3]
        # 对原始数据映射到新的空间
        transformed_data=data_norm.dot(P_mat)
        return transformed_data

if __name__=='__main__':
    data=np.array([[1,-1,0,2,1],[2,0,0,-1,-1],[2,0,1,1,0],[1,1,0,-1,1]])
    pca = PCA(ratio=0.9)#用比例的方法在高维度的时候比直接保留k阶可能比较合理，
    new_data=pca.transform(data)
    print("new_data:",new_data)









