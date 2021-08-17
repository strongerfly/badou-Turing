'''
根据最小二乘的定义写出最小二乘算法
'''
import random
import matplotlib.pyplot as plt

def LeastSquare(X1, X2):
    '''
    :param X: 是一个(n, 2)的list, or narray
    :return: 最优的线性拟合的系数
    '''
    N = len(X1)
    sum_x1x2 = 0
    sum_x1 = 0
    sum_x2 = 0
    sum_x1_square = 0
    for i in range(N):
        sum_x1x2 += X1[i] * X2[i]
        sum_x1 += X1[i]
        sum_x2 += X2[i]
        sum_x1_square += X1[i] ** 2
    k = (N * sum_x1x2 - sum_x1 * sum_x2) / (N * sum_x1_square - sum_x1 ** 2)
    b = (sum_x2  - k * sum_x1) / N
    return round(k, 2), round(b, 2)



if __name__ == '__main__':
    random.seed(222)
    X1 = [i for i in range(100)]
    X2 = [round(4 * i + 4 + random.randint(-20, 20), 2) for i in X1]
    # plt
    plt.figure(figsize=(8, 16))
    plt.subplot(2,1,1)
    plt.scatter(X1,X2)
    plt.title('relationship of X1 and X2')
    plt.xlabel('X1'), plt.ylabel('x2')
    plt.xlim(0, 102), plt.ylim(0, 420)   # x,y 的范围
    plt.xticks(range(0, 102, 20), [str(i) for i in range(0, 102, 20)])
    # 拟合数据
    k, b = LeastSquare(X1, X2)
    print('拟合结果的系数', k, b)

    # 画出拟合的线
    Y = [k * i + b for i in X1]
    plt.plot(X1, Y, color = 'red')

    plt.subplot(2, 1, 2)
    X1_add = [random.randint(0, 400) for i in range(50)]
    X2_add = [random.randint(0, 400) for i in range(50)]
    X1.extend(X1_add)
    X2.extend(X2_add)
    plt.scatter(X1,X2)
    plt.title('relationship of X1 and X2')
    plt.xlabel('X1'), plt.ylabel('x2')
    plt.xlim(0, 102), plt.ylim(0, 420)   # x,y 的范围
    plt.xticks(range(0, 102, 20), [str(i) for i in range(0, 102, 20)])
    # 拟合数据
    k1, b1 = LeastSquare(X1, X2)
    print('加入噪声后的拟合系数', k1, b1)

    # 画出拟合的线
    Y = [k1 * i + b1 for i in X1]
    plt.plot(X1, Y, color = 'red')
    plt.show()


