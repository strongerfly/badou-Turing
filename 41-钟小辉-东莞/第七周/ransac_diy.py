import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as sl
import scipy as sp

class LinearLeastsquareModel:
    # 最小二乘求线性解,用于RANSAC的输入模型
    def __init__(self,input_columns,output_columns,debug = False):
        self.input_columns = input_columns
        self.output_columns = output_columns
        self.debug = debug

    # 将输入x，y进行按a0 + b0x = y 进行堆叠，写成[1，x] *[a,b]^T = [y]矩阵格式
    # np.vstack按垂直方向（行顺序）堆叠数组构成一个新的数组
    def model_fit(self,data):
        A = np.vstack([data[:,i] for i in self.input_columns]).T
        B = np.vstack([data[:,i] for i in self.output_columns]).T
        x, resids, rank, s = sl.linalg.lstsq(A,B) #residues:残差和，x：返回最小二乘法的k/b值
        return x #返回最小平方和向量

    def get_error(self,data,model):
        A = np.vstack([data[:, i] for i in self.input_columns]).T
        B = np.vstack([data[:, i] for i in self.output_columns]).T
        # 计算的y值,B_fit = model.k*A + model.b
        B_fit = sp.dot(A,model)
        err_per_point =np.sum((B-B_fit)**2,axis=1)
        return err_per_point


def ransac_model(k):

    iterations = 0
    best_model = None
    best_consensus_set = None
    best_error = np.inf
    while(iterations < k):
        h = 0

def test():
    jj = 0


if __name__ == "__main__":
    test()

"""
#         RANSAC算法的输入是一组观测数据，一个可以解释或者适应于观测数据的参数化模型，一些可信的参数。
#         RANSAC通过反复选择数据中的一组随机子集来达成目标。被选取的子集被假设为局内点，并用下述方法进行验证：
#         1.
#         首先我们先随机假设一小组局内点为初始值。然后用此局内点拟合一个模型，此模型适应于假设的局内点，所有的未知参数都能从假设的局内点计算得出。
#         2.
#         用1中得到的模型去测试所有的其它数据，如果某个点适用于估计的模型，认为它也是局内点，将局内点扩充。
#         3.
#         如果有足够多的点被归类为假设的局内点，那么估计的模型就足够合理。
#         4.
#         然后，用所有假设的局内点去重新估计模型，因为此模型仅仅是在初始的假设的局内点估计的，后续有扩充后，需要更新。
#         5.
#         最后，通过估计局内点与模型的错误率来评估模型。  整个这个过程为迭代一次，此过程被重复执行固定的次数，每次产生的模型有两个结局：  1、要么因为局内点太少，还不如上一次的模型，而被舍弃，  2、要么因为比现有的模型更好而被选用。
#

# 输入：
# data —— 一组观测数据
# model —— 适应于数据的模型
# n —— 适用于模型的最少数据个数
# k —— 算法的迭代次数
# t —— 用于决定数据是否适应于模型的阀值
# d —— 判定模型是否适用于数据集的数据数目
# 输出：
# best_model —— 跟数据最匹配的模型参数（如果没有找到好的模型，返回null）
# best_consensus_set —— 估计出模型的数据点
# best_error —— 跟数据相关的估计出的模型错误
# ————————————————
   
maybe_inliers =    #从数据集中随机选择n个点
maybe_model = #适合于maybe_inliers的模型参数
consensus_set = maybe_inliers
for ( #每个数据集中不属于maybe_inliers的点 ） 
if ( 如果点适合于maybe_model，且错误小于t ）
将点添加到consensus_set
if （ consensus_set中的元素数目大于d ）
已经找到了好的模型，现在测试该模型到底有多好
better_model = 适合于consensus_set中所有点的模型参数
this_error = better_model究竟如何适合这些点的度量
if ( this_error < best_error )
我们发现了比以前好的模型，保存该模型直到更好的模型出现
best_model =  better_model
best_consensus_set = consensus_set
best_error =  this_error
增加迭代次数
返回 best_model, best_consensus_set, best_error
"""