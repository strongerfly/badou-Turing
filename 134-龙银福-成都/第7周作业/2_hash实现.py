import cv2
import numpy as np

""" 均值哈希算法 aHash """
def aHash(image):
    # 1-缩放：图片缩放为8x8, 保留结构, 除去细节
    image = cv2.resize(image, (8, 8), interpolation=cv2.INTER_CUBIC) # INTER_CUBIC: 4x4像素邻域的双三次插值
    # 2-灰度化
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 3-求平均值：计算灰度图所有像素的平均值
    pixels_mean = 0
    for i in range(8):
        for j in range(8):
            pixels_mean += gray[i, j]
    pixels_mean = pixels_mean / 64
    # 4-比较：像素值大于平均值记作1, 相反记作0, 总共64位
    # 5-生成hash：将上述步骤生成的1和0按顺序组合起来既是图像的指纹(hash)
    hash_value = ''
    for i in range(8):
        for j in range(8):
            if gray[i, j] > pixels_mean:
                hash_value = hash_value + '1'
            else:
                hash_value = hash_value + '0'
    return hash_value

""" 差值哈希算法 dHash """
def dHash(image):
    # 1-缩放：图像缩放为8x9, 保留结构, 除去细节
    image = cv2.resize(image, (9, 8), interpolation=cv2.INTER_CUBIC)
    # 2-灰度化
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 3-比较：像素值大于后一个像素值记做1, 相反记作0, 本行不与下一行对比
    # 4-生成hash：将上述步骤生成的1和0按顺序组合起来既是图片的指纹(hash)
    hash_value = ''
    for i in range(8):
        for j in range(8):
            if gray[i, j] > gray[i, j+1]:
                hash_value = hash_value + '1'
            else:
                hash_value = hash_value + '0'
    return hash_value

""" hash值对比(对比指纹) --> 将两幅图的指纹对比,计算汉明距离,即两个64位的hash值
    有多少位是不一样的,不相同位数越少,图片越相似 """
def cmpHash(hash1, hash2):
    if len(hash1) != len(hash2): # 传入的参数长度不等, 返回-1代表传参出错
        return -1
    # 遍历判断
    n = 0 # 相似度
    for i in range(len(hash1)):
        if hash1[i] != hash2[i]:
            n += 1
    return n

if __name__ == '__main__':
    image1 = cv2.imread("lenna.png", 1)
    image2 = cv2.imread("lenna_noise.png", 1)

    ahash1 = aHash(image1)
    ahash2 = aHash(image2)
    print("lenna图像的均值hash: ", ahash1)
    print("加噪lenna图像的均值hash: ", ahash2)
    n1 = cmpHash(ahash1, ahash2)
    print("均值hash算法相似度: ", n1)

    dhash1 = dHash(image1)
    dhash2 = dHash(image2)
    print("lenna图像的差值hash: ", dhash1)
    print("加噪lenna图像的差值hash: ", dhash2)
    n2 = cmpHash(dhash1, dhash2)
    print("插值hash算法相似度: ", n2)