import cv2
import numpy as np


def peper(snp):
    '''
    :param snp: 信噪比
    :return:
    '''
    h, w = gray.shape
    arr = np.random.choice([0, 255], int(round(h * w * snp)))  # [0, 255, 255, 0, 0, 255] 随机出现
    # 将这些点随机加入到图像中
    # 随机选择i,j
    x = np.random.choice(h, int(round(h * w * snp)))
    y = np.random.choice(w, int(round(h * w * snp)))
    for i, j, k in zip(x, y, arr):
        gray[i,j] = k
    cv2.imshow('pepper salt noise demo', gray.astype(np.uint8))
    cv2.waitKey(0)

snp = 0.1
img = cv2.imread('lenna.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 转换彩色图像为灰度图
peper(0.1)






'''
numpy.random.choice(a, size=None, replace=True, p=None)
Parameters:	
a : 1-D array-like or int
If an ndarray, a random sample is generated from its elements. If an int, the random sample is generated as if a were np.arange(a)

size : int or tuple of ints, optional
Output shape. If the given shape is, e.g., (m, n, k), then m * n * k samples are drawn. Default is None, in which case a single value is returned.

replace : boolean, optional
Whether the sample is with or without replacement

p : 1-D array-like, optional
The probabilities associated with each entry in a. If not given the sample assumes a uniform distribution over all entries in a.

#从a(只要是ndarray都可以，但必须是一维的)中随机抽取数字，并组成指定大小(size)的数组
#replace:True表示可以取相同数字，False表示不可以取相同数字
#数组p：与数组a相对应，表示取数组a中每个元素的概率，默认为选取每个元素的概率相同。
'''
'''
c = np.random.choice(5,3)   # range(5) ,随机选择3个，  默认可以可重复，
print(c, 'ccc')

d = np.random.choice([0, 255], 20)
print(d, 'dd')

arr = np.random.choice([0, 255], int(2.0))
print(arr, 'dd')

np = 0.3
h, w = gray.shape
print(int(h * w * np), 'np')
'''
