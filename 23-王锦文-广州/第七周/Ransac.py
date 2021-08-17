#coding=utf-8
import numpy as np
import random
from scipy.optimize import  leastsq
import pylab

'''
该文件实现Ransac算法
'''
class LinearLeastSquareModel(object):
    '''
    该类实现最小二乘法
    '''
    def __init__(self,input_columns, output_columns,paramnum):
        '''
        初始化
        input_columns:输入的变量个数
        output_columns:输出的变量个数
        paramnum:模型参数数量，比如a*x^2+b*x+c=y，参数为（a,b,c）共三个参数
        '''
        self.input_columns=input_columns
        self.output_columns=output_columns
        self.paramnum=paramnum#该模型有多少个参数
        self.fitparam = np.random.randn(self.paramnum)  # 第一次迭代的时候随机初始化多项式的参数,后续更新
    def fit_Fun(self,param,data):
        '''
        这里是定义一个要拟合的目标函数，这里我们假设拟合一个多项式函数
        param:多项式的参数
        data：数据
        '''
        func=np.poly1d(param)
        return func(data)

    def get_error(self,param,x,y):
        '''
        获取模型预测结果与真实值间的误差，用于传入优化器中，计算得到最优解
        param：模型的参数
        data：需要拟合的数据
        '''
        error=self.fit_Fun(param,x)-y
        return error
    def get_best_params(self,data):
        '''
        获取模型的拟合参数
        '''
        x=data[:,:self.input_columns]
        y=data[:,self.output_columns:]
        x = np.squeeze(x)
        y=np.squeeze(y)
        fit_param=leastsq(self.get_error,self.fitparam,args=(x,y))#拟合曲线
        self.fitparam=fit_param[0]
        return  fit_param[0]

    def get_test_error(self,param,data):
        '''
        返回测试样本的误差
        param：拟合得到的曲线参数
        data：测试的数据
        return：返回预测值与真实值的误差
        '''
        x = data[:, :self.input_columns]
        y = data[:, self.output_columns:]
        y_fit = self.fit_Fun(param,x)  # 计算预测的y值
        err= np.sum((y - y_fit) ** 2, axis=1)
        return err

def random_samples(samples,n):
    '''
    对样本进行随机采样
    samples:样本
    n：需要至少抽取n个样本点作为内群点拟合
    return 返回抽取的内群点和其他待验证点
    '''
    ids=np.arange(len(samples))
    np.random.shuffle(ids)
    intlier_samples=samples[ids[:n],:]
    val_samples=samples[ids[n:],:]
    return intlier_samples,val_samples

def Ransac(data, model, n, k, t, d):
    """
    输入:
        data - 样本点
        model - 假设模型:已知
        n - 生成模型所需的最少样本点
        k - 最大迭代次数
        t - 阈值:作为判断点满足模型的条件
        d - 拟合ok,需要的样本点最少的个数
    输出:
        bestfit - 最优拟合解参数
    """
    iterations = 0
    bestfit = None
    besterr = np.inf  #
    best_inlier_data= None
    #fitparam=np.random.randn(model.paramnum)#第一次迭代的时候随机初始化多项式的参数
    for _ in range(k):
        inliers,outliers=random_samples(data,n)#随机选择样本作为内群点inliters
        #先进行模型的拟合,fitparam返回拟合的参数
        fitparam=model.get_best_params(inliers)
        test_err=model.get_test_error(fitparam,outliers)#获取该内群点拟合参数下，验证点的预测值与真实值的误差
        inner_data=outliers[test_err<t]#从outliters样本中选出误差小于阈值的点
        if len(inner_data)>d:
            betterdata=np.vstack((inliers,inner_data))#合并内群数据
            bettermodelparam=model.get_best_params(betterdata)#重新用新的数据获取模型最优参数
            better_errs=model.get_test_error(bettermodelparam,betterdata)#获取新参数后的内群点预测值与真实值误差
            newerr=np.mean(better_errs)#求平均
            if newerr<besterr:#小于误差，则更新参数
                bestfit=bettermodelparam
                besterr=newerr
                best_inlier_data=betterdata

    if bestfit is None:
        raise ValueError("===fit error，param not found")
    return bestfit,betterdata

if __name__=="__main__":
    # 生成数据,
    n_samples = 600  # 样本个数
    n_inputs = 1  # 输入变量个数,这里我们假设拟合一个多项式（y=a1*x^2+a2*x+a3）
    n_outputs = 1  # 输出变量个数
    x_data=np.linspace(-30,30,n_samples).reshape(-1, 1)
    src_param=[1,-1.5,-3]#这是原始曲线的参数
    real_y = np.poly1d(src_param)(x_data).reshape(-1,1)  # 真实的标签
    #对原始数据先加入高斯噪声，再加入离群点
    gasuss_noise=np.random.normal(loc=0,scale=0.5,size=x_data.shape)
    x_data_noise=x_data+gasuss_noise
    real_y_noise=real_y+gasuss_noise
    #加入离群点
    n_outliers=180
    #随机抽取100个数据id
    outlier_idxs=random.sample(np.arange(x_data_noise.shape[0]).tolist(),n_outliers)#采样的索引
    x_data_noise[outlier_idxs]=30 * np.random.random((n_outliers, n_inputs))
    real_y_noise[outlier_idxs] = 40* np.random.normal(loc=1.0,scale=4.0,size=(n_outliers, n_outputs))  # 加入噪声和离群点
    all_data=np.hstack((x_data_noise,real_y_noise))
    model = LinearLeastSquareModel(1,1,3) #最新二乘拟合类实例化
    # RANSAC 算法
    ransac_fit, ransac_data = Ransac(all_data, model, 100, 1000, 5e3, 300)
    #绘图相关，这里参考提供的画图代码
    sort_idxs = np.argsort(x_data_noise[:, 0])
    x_data_sorted = x_data_noise[sort_idxs]  # 排序，x从小到大绘图
    pylab.plot(x_data_noise[:, 0], real_y_noise[:, 0], 'k.', label='data')  # 散点图
    pylab.plot(ransac_data[:,0], ransac_data[:,1], 'gx', label="RANSAC data")
    y_fit=np.poly1d(ransac_fit)(x_data_sorted)
    y_perfit=np.poly1d(src_param)(x_data_sorted)
    pylab.plot(x_data_sorted[:, 0],y_perfit,'r',label='real curve')
    pylab.plot(x_data_sorted[:, 0],y_fit[:, 0],'b',label='RANSAC fit')
    pylab.legend()
    pylab.show()



