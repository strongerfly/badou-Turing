import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as sl
import scipy as sp

"""原理
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


"""

class LinearLeastsquareModel:
    # 最小二乘求线性解,用于RANSAC的输入模型
    def __init__(self,input_columns,output_columns,debug = False):
        self.input_columns = input_columns
        self.output_columns = output_columns
        self.debug = debug

    # 将输入x，y进行按a0 + b0x = y 进行堆叠，写成[1，x] *[a,b]^T = [y]矩阵格式
    # np.vstack按垂直方向（行顺序）堆叠数组构成一个新的数组
    def fit(self,data):
        A = np.vstack([data[:,i] for i in self.input_columns]).T
        B = np.vstack([data[:,i] for i in self.output_columns]).T
        x, resids, rank, s = sp.linalg.lstsq(A,B) #residues:残差和，x：返回最小二乘法的k/b值
        return x #返回最小平方和向量

    def get_error(self,data,model):
        A = np.vstack([data[:, i] for i in self.input_columns]).T
        B = np.vstack([data[:, i] for i in self.output_columns]).T
        # 计算的y值,B_fit = model.k*A + model.b
        B_fit = sp.dot(A,model)
        err_per_point =np.sum((B-B_fit)**2,axis=1)
        return err_per_point


def ransac_model(data,model,n,k,t,d,debug= False,return_all = False):
    # 输入:
    #     data - 样本点
    #     model - 假设模型:事先自己确定
    #     n - 生成模型所需的最少样本点
    #     k - 最大迭代次数
    #     t - 阈值:作为判断点满足模型的条件
    #     d - 拟合较好时,需要的样本点最少的个数,当做阈值看待
    # 输出:
    #     bestfit - 最优拟合解（返回nil,如果未找到）
    # 输出：
    # best_model —— 跟数据最匹配的模型参数（如果没有找到好的模型，返回null）
    # best_consensus_set —— 估计出模型的数据点
    # best_error —— 跟数据相关的估计出的模型错误
    # ————————————————

    # maybe_inliers =  # 从数据集中随机选择n个点
    # maybe_model =  # 适合于maybe_inliers的模型参数
    # consensus_set = maybe_inliers
    # for (  # 每个数据集中不属于maybe_inliers的点 ）
    #         if ( 如果点适合于maybe_model，且错误小于t ）
    #         将点添加到consensus_set
    #         if （ consensus_set中的元素数目大于d ）
    #         已经找到了好的模型，现在测试该模型到底有多好
    #         better_model = 适合于consensus_set中所有点的模型参数
    #         this_error = better_model究竟如何适合这些点的度量
    #         if ( this_error < best_error)
    # 我们发现了比以前好的模型，保存该模型直到更好的模型出现
    # best_model =  better_model
    # best_consensus_set = consensus_set
    # best_error =  this_error
    # 增加迭代次数
    # 返回 best_model, best_consensus_set, best_error
    iterations = 0
    bestfit = None
    best_inlier_idxs = None #内群
    better_point_set = None #

    betterr = np.inf
    while(iterations < k):
        #从数据中随机选择n个点作为内群点maybe_idxs,其他测试点test_idxs
        #这里注意，需要每次都将数据打乱
        if n < data.shape[0]:
            all_idxs = np.arange(data.shape[0]) #生成特定补偿的排列
            np.random.shuffle(all_idxs)
            maybe_idxs = all_idxs[:n]
            test_idxs = all_idxs[n:]

        else:
            print(f"设定的内群点数目大于输入点数目 {n} > {data.shape[0]}" )
            break
        #内群随机点
        maybe_inliners = data[maybe_idxs,:]
        test_points = data[test_idxs,:]

        # 适合于内群点maybe_inliers的模型参数k/b
        maybe_model = model.fit(maybe_inliners)
        #计算其他点误差的最小平方和
        test_err = model.get_error(test_points,maybe_model)

        # 如果点适合于maybe_model，且错误小于t，这里consensus_set/also_inliers是等同的，这里只是学习
        consensus_set = test_points[test_err<t]
        also_idxs =  test_idxs[test_err<t]
        also_inliers = data[also_idxs, :]
        # print(also_inliners)

        if debug:
            print ('test_err.min()',test_err.min())
            print ('test_err.max()',test_err.max())
            print ('numpy.mean(test_err)',np.mean(test_err))
            print ('iteration %d:len(alsoinliers) = %d' %(iterations, len(also_inliers)) )

        #         将点添加到consensus_set
        if (len(also_inliers) > d):
            # betterdata = np.concatenate((maybe_inliners,consensus_set))
            betterdata = np.concatenate((maybe_inliners, also_inliers))
            better_model = model.fit(betterdata)
            better_err = model.get_error(betterdata,better_model)
            thiserr = np.mean(better_err)
            if thiserr < betterr :
                bestfit = better_model
                betterr = thiserr
                best_point_set = betterdata
                best_inlier_idxs = np.concatenate((maybe_idxs, also_idxs))  # 更新局内点,将新点加入

        iterations += 1
    if bestfit is None:
        raise ValueError("did't meet fit acceptance criteria")
    if return_all:
        return bestfit,{'inliers':best_inlier_idxs},best_point_set
    else:
        return bestfit







def test():
    # 生成理想数据
    n_samples = 500  # 样本个数
    n_inputs = 1  # 输入变量个数
    n_outputs = 1  # 输出变量个数

    # 随机生成0-20之间的500个数据:行向量x
    A_exact = 20 * np.random.random((n_samples,n_inputs))
    # print(f"%s A_exact.shape:",A_exact.shape) #500*1

    #生成斜率K
    perfect_fit =20 * np.random.normal(size=(n_inputs,n_outputs))

    #由y = kx生成直线上的点 500*1
    B_exact = sp.dot(A_exact,perfect_fit)

    #生成局外点和噪声点,这里是噪声 500 * 1行向量,代表Xi/Yi
    A_noisy = A_exact + np.random.normal(size=A_exact.shape)
    B_noisy = B_exact + np.random.normal(size=B_exact.shape)
    # print(f"%s A_noisy:", A_noisy[range(2), :])
    # print(f"%s A_noisy:", B_noisy[range(2), :])

    if 1:
        #局外干扰点,打乱原数组顺序，添加局外干扰点
        n_outliners = 100
        all_idxs = np.arange(A_noisy.shape[0])
        np.random.shuffle(all_idxs) #将all_idxs打乱
        outliner_idxs = all_idxs[:n_outliners]#取前100个
        A_noisy[outliner_idxs] = 20 * np.random.normal(size=(n_outliners,n_outputs))
        B_noisy[outliner_idxs] = 30 * np.random.normal(size=(n_outliners, n_outputs))
        #print("%s A_noise.shape:",A_noise.shape)


    #setup-model
    all_data = np.hstack((A_noisy,B_noisy)) #形式([Xi,Yi]....) shape:(500,2)
    # print(f"%s all_data:", all_data[range(2), :])
    input_columns = range(n_inputs)
    output_columns = [n_inputs+i for i in range(n_outputs)]
    debug = False

    #使用最小二乘法生成模型
    model = LinearLeastsquareModel(input_columns,output_columns,debug=debug) #类的实例化:用最小二乘生成已知模型

    #生成最小二乘法的model, 将x，y代入y =ax+b中求得a，b，都存储到第一个返回值里 linear_fit，这是最小二乘法的拟合K/b
    linear_fit,residues,rank,s = sp.linalg.lstsq(all_data[:,input_columns],all_data[:,output_columns])

    #由Ransac算法计算拟合的K值
    ransac_fit, ransac_data, ransac_point =   ransac_model(all_data, model, 50, 3000, 5e3, 200,debug = debug, return_all = True)


    if 1:
        import pylab

        sort_idxs = np.argsort(A_exact[:, 0])
        A_col0_sorted = A_exact[sort_idxs]  # 秩为2的数组

        #没有看懂为什么从小到大排序，有啥区别
        # A_col0_sorted = A_exact

        if 1:
            pylab.plot(A_noisy[:, 0], B_noisy[:, 0], 'k.', label='data')  # 散点图

            #展示方式1
            pylab.plot( A_noisy[ransac_data['inliers'], 0], B_noisy[ransac_data['inliers'], 0], 'bx',
                       label="RANSAC data")

            # 展示方式2
            # pylab.plot(ransac_point[:, 0], ransac_point[:, 1], 'x',
            #            label="RANSAC another data")


        else:
            pylab.plot(A_noisy[non_outlier_idxs, 0], B_noisy[non_outlier_idxs, 0], 'k.', label='noisy data')
            pylab.plot(A_noisy[outlier_idxs, 0], B_noisy[outlier_idxs, 0], 'r.', label='outlier data')

        pylab.plot(A_col0_sorted[:, 0],
                   np.dot(A_col0_sorted, ransac_fit)[:, 0],
                   color="red", label='RANSAC fit')
        pylab.plot(A_col0_sorted[:, 0],
                   np.dot(A_col0_sorted, perfect_fit)[:, 0],
                   color="yellow", label='exact system')
        pylab.plot(A_col0_sorted[:, 0],
                   np.dot(A_col0_sorted, linear_fit)[:, 0],
                   label='linear fit')
        pylab.legend()
        pylab.show()






if __name__ == "__main__":
    test()

