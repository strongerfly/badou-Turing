'''
RANSAC是一个算法的思想， 可以将很多算法进行优化

the code come from the teacher, but I have understand all the meanings of it,

'''
import numpy as np
import scipy as sp
import scipy.linalg as sl


def random_partition(n, n_data):
    '''
    :param n: 选择前n个数据, int
    :param n_data: 需要打乱的数据总数, int
    :return: 打乱后的两部分数据的索引
    idxs1:
    idxs2:
    '''
    all_index = np.arange(n_data)    # 获取所有的索引
    np.random.shuffle(all_index)     # 打乱下标索引
    idxs1 = all_index[:n]            # 取前n个打乱的数据
    idxs2 = all_index[n:]            # 剩余数据的索引
    return idxs1, idxs2



def RandomSampleConsensus(data, model, n, k, t, d, debug = False, return_all = False):
    """
    输入:
        data - 样本点
        model - 假设模型:事先自己确定
        n - 生成模型所需的最少样本点
        k - 最大迭代次数
        t - 阈值:作为判断点满足模型的条件
        d - 拟合较好时,需要的样本点最少的个数,当做阈值看待
    输出:
        bestfit - 最优拟合解（返回nil,如果未找到）

    iterations = 0
    bestfit = nil #后面更新
    besterr = something really large #后期更新besterr = thiserr
    while iterations < k
    {
        maybeinliers = 从样本中随机选取n个,不一定全是局内点,甚至全部为局外点
        maybemodel = n个maybeinliers 拟合出来的可能符合要求的模型
        alsoinliers = emptyset #满足误差要求的样本点,开始置空
        for (每一个不是maybeinliers的样本点)
        {
            if 满足maybemodel即error < t
                将点加入alsoinliers
        }
        if (alsoinliers样本点数目 > d)
        {
            %有了较好的模型,测试模型符合度
            bettermodel = 利用所有的maybeinliers 和 alsoinliers 重新生成更好的模型
            thiserr = 所有的maybeinliers 和 alsoinliers 样本点的误差度量
            if thiserr < besterr
            {
                bestfit = bettermodel
                besterr = thiserr
            }
        }
        iterations++
    }
    return bestfit
    """
    # ransac迭代参数的初始化
    iterations = 0
    bestfit = None    # 需要拟合的模型
    besterr = np.inf   # 初始的误差值设置为无穷大， 只有这样对很大的计算误差，也可以继续更新
    best_inlier_idxs = None   # 定义为内群点
    # 对于迭代次数，未确定的， 使用while, iterations < k（最大迭代次数）
    while iterations < k:
        #首先要对内群点初始化， 同时对于剩余点保存，且这样的划分是随机的
        maybe_idxs, test_idxs = random_partition(n, data.shape[0])
        print ('test_idxs = ', test_idxs)
        # 获取随机选择的数据
        maybe_inliers = data[maybe_idxs, :]  # 等价于maybe_inliers = data[maybe_idxs]
        test_points = data[test_idxs, :]     # 等价于：test_points = data[test_idxs]
        # 用随机选择的内群点进行模型拟合
        maybymodel = model.fit(maybe_inliers)
        test_err = model.get_error(test_points, maybymodel)  # 计算误差：least square sum
        print('test_err = ', test_err < t)
        also_idxs = test_idxs[test_err < t]
        print('also_idxs', also_idxs)
        also_inliers = data[also_idxs, :]
        if debug:
            print ('test_err.min()',test_err.min())
            print ('test_err.max()',test_err.max())
            print ('numpy.mean(test_err)',np.mean(test_err))
            print ('iteration %d:len(alsoinliers) = %d' %(iterations, len(also_inliers)) )
        # if (len(also_inliers) > d)
        print('d = ', d )
        if (len(also_inliers) > d):
            betterdata = np.concatenate( (maybe_inliers, also_inliers))   # 样本的连接
            bettermodel = model.fit(betterdata)
            betterdata_errs = model.get_error(betterdata, bettermodel)
            thiserr = np.mean(betterdata_errs)   #平均误差作为新的误差
            if thiserr < besterr:
                bestfit = bettermodel
                besterr = thiserr
                best_inlier_idxs = np.concatenate( (maybe_idxs, also_idxs))  #更新局内点,将新点加入
        iterations += 1
    if bestfit is None:
        raise ValueError(" did't meet fit acceptance criteria ")
    if return_all:
        return bestfit, {'inliers':best_inlier_idxs}
    else:
        bestfit

class LinearLeastSquareModel:
    #最小二乘求线性解,用于RANSAC的输入模型
    def __init__(self, input_columns, output_columns, debug = False):
        self.input_columns = input_columns
        self.output_columns = output_columns
        self.debug = debug

    def fit(self, data):
        #np.vstack按垂直方向（行顺序）堆叠数组构成一个新的数组
        A = np.vstack( [data[:,i] for i in self.input_columns] ).T #第一列Xi-->行Xi
        B = np.vstack( [data[:,i] for i in self.output_columns] ).T #第二列Yi-->行Yi
        x, resids, rank, s = sl.lstsq(A, B)  #residues:残差和
        return x #返回最小平方和向量, 拟合的系数

    def get_error(self, data, model):
        '''
        :param data: 原始数据
        :param model: 拟合系数
        :return:
        '''
        A = np.vstack( [data[:,i] for i in self.input_columns] ).T #第一列Xi-->行Xi  data[:,i] = 向量 = 一行 = 一列
        B = np.vstack( [data[:,i] for i in self.output_columns] ).T #第二列Yi-->行Yi
        B_fit = sp.dot(A, model) #计算的y值,B_fit = model.k*A + model.b
        err_per_point = np.sum( (B - B_fit) ** 2, axis = 1) #sum squared error per row
        return err_per_point

def test():
    n_samples = 600
    n_inputs = 1
    n_outputs = 1
    # 生成自变量
    A_exact = 20 * np.random.random(size = (n_samples, n_inputs))
    perfect_fit = 60 * np.random.rand(n_inputs, n_outputs)        # 斜率
    B_exact = np.dot(A_exact, perfect_fit)

    #加入高斯噪声,最小二乘能很好的处理
    A_noisy = A_exact + np.random.normal(size = A_exact.shape)
    B_noisy = B_exact + np.random.normal(size = B_exact.shape)

    # 添加异常点
    if 1:
        n_outliers = int(round( n_samples * 0.3 ))
        # 获取A_noisy的索引0 - 599
        all_idxs = np.arange(A_noisy.shape[0])
        # 将all_idxs打乱
        np.random.shuffle(all_idxs)
        # 随机选择前n_outliers个索引点，作为异常点的索引
        outliers_idxs = all_idxs[:n_outliers]
        # 对含有噪声的数据，将outliers_idxs对应的索引位置换成噪声和局外点Xi, Yi
        A_noisy[outliers_idxs] = 20 * np.random.random( (n_outliers, n_inputs) )
        B_noisy[outliers_idxs] = 60 * np.random.random( (n_outliers, n_outputs) )

        # 对于模型的设置
        all_data = np.hstack( (A_noisy, B_noisy) )  # 形式([Xi,Yi]....) shape:(600,2)600行2列
        input_columns = range(n_inputs)  # 数组的第一列x:0
        # 获得输出变量的列索引
        output_columns = [n_inputs + i for i in range(n_outputs)]
        debug = False

        # 带入模型获得基础的解
        # 类的实例化:用最小二乘生成已知模型
        model = LinearLeastSquareModel(input_columns, output_columns, debug = debug)
        linear_fit, resids, rank, s = sp.linalg.lstsq(all_data[:, input_columns], all_data[:, output_columns])
        # run ransac 算法
        ransac_fit, ransac_data = RandomSampleConsensus(data = all_data, model = model, n = 50, k = 1000, t = 7e3,
                                                       d = 300, debug = debug, return_all = True)

        if 1:
            import pylab
            # value:min--->max indices ---Returns the indices that would sort an array.
            # 对于输入的原始数据按从小到大的索引排列
            sort_idxs = np.argsort(A_exact[:, 0])
            A_col0_sorted = A_exact[sort_idxs]    # 小到大  # 秩为2的数组  排序之后直线

        if 1:
            pylab.plot( A_noisy[:, 0], B_noisy[:, 0], 'k.', label = 'data')   # 散点图
            pylab.plot( A_noisy[ransac_data['inliers'], 0], B_noisy[ransac_data['inliers'], 0], 'bx',
                        label = "RANSAC data")
        else:
            pylab.plot( A_noisy[non_outlier_idxs, 0], B_noisy[non_outlier_idxs, 0], 'k.',
                        label = "noisy data")
            pylab.plot( A_noisy[outlier_idxs, 0], B_noisy[outlier_idxs, 0], 'r.',
                        label = "outlier data")

        pylab.plot( A_col0_sorted[:, 0], np.dot(A_col0_sorted, ransac_fit)[:, 0], 'r', label = "RANSAC fit" )
        pylab.plot( A_col0_sorted[:, 0], np.dot(A_col0_sorted, perfect_fit)[:, 0], 'g', label = "exact system" )
        pylab.plot( A_col0_sorted[:, 0], np.dot(A_col0_sorted, linear_fit)[:, 0] , 'orange', label = "linear fit")

        pylab.legend()
        pylab.show()
if __name__ == '__main__':
    test()



