# -*- coding: utf-8 -*-
"""

@author: wuming
descri: 射影变化
"""
# -*- coding: utf-8 -*-
import numpy as np

def WarpPerspectiveMatrix(src,dst):
    assert src.shape[0]==dst.shape[0] and src.shape[0]>=4
    nums=src.shape[0]
    A=np.zeros((2*nums,8))
    B=np.zeros((2*nums,1))
    for i in range(nums):
        A_i=src[i,:]
        B_i=dst[i,:]
        A[2*i,:]=[A_i[0],A_i[1],1,0,0,0,-A_i[0]*B_i[0],-A_i[1]*B_i[0]]
        A[2*i+1,:]=[0,0,0,A_i[0],A_i[1],1,-A_i[0]*B_i[1],-A_i[1]*B_i[1]]
        B[2*i]=B_i[0]
        B[2 * i+1] = B_i[1]
    A=np.mat(A)
    warpMatrix=A.I*B  #A.I为A的逆矩阵
    #注意：array可以用于1到n维而mat只能是二维
    warpMatrix=np.array(warpMatrix).T[0]#先将举证转化为多维向量对矩阵进行转置T[0]直接将向量的维度由（1，8）转为（8，）并去掉外面中括号
    warpMatrix=np.insert(warpMatrix, warpMatrix.shape[0], values=1.0, axis=0) #参数2是插入的位置
    warpMatrix=warpMatrix.reshape((3,3))
    return warpMatrix



if __name__ == '__main__':
    print('warpMatrix')
    src = [[10.0, 457.0], [395.0, 291.0], [624.0, 291.0], [1000.0, 457.0]]
    src = np.array(src)
    dst = [[46.0, 920.0], [46.0, 100.0], [600.0, 100.0], [600.0, 920.0]]
    dst = np.array(dst)
    warpMatrix = WarpPerspectiveMatrix(src, dst)
    print(warpMatrix)