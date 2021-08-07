import cv2
import numpy as np
def meanhash(img):
    # 1. 缩放：图片缩放为8*8，保留结构，除去细节。
    img = cv2.resize(img, (8,8), interpolation=cv2.INTER_LINEAR)
    # 2. 灰度化：转换为灰度图。
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 3. 求平均值：计算灰度图所有像素的平均值。
    pixel_mean = np.mean(gray)
    # 4. 比较：像素值大于平均值记作1，相反记作0，总共64位。
    #5. 生成hash：将上述步骤生成的1和0按顺序组合起来既是图片的指纹（hash）。
    hashstr = ''
    for i in range(8):
        for j in range(8):
            if gray[i,j] > pixel_mean:
                hashstr += '1'
            else:
                hashstr += '0'
    return hashstr

def diffhash(img):
    # 1. 缩放：图片缩放为8*9，保留结构，除去细节。
    img = cv2.resize(img, (9, 8), interpolation=cv2.INTER_LINEAR)
    # 2. 灰度化：转换为灰度图。
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 3. 求平均值：计算灰度图所有像素的平均值。 ---这步没有，只是为了与均值哈希做对比
    # 4. 比较：像素值大于后一个像素值记作1，相反记作0。本行不与下一行对比，每行9个像素，八个差值，有8行，总共64位
    # 5. 生成hash：将上述步骤生成的1和0按顺序组合起来既是图片的指纹（hash）。
    hashstr = ''
    i = 0
    while i < 8:
        for j in range(8):
            if gray[i, j] > gray[i,j+1]:
                hashstr += '1'
            else:
                hashstr += '0'
        i += 1
    return hashstr

# 6. 对比指纹：将两幅图的指纹对比，计算汉明距离，即两个64位的hash值有多少位是不一样的，不
# 相同位数越少，图片越相似
def cmphash(hashstr1, hashstr2):
    n = 0
    if len(hashstr1) != len(hashstr2):
        return -1
    for i, j in zip(hashstr1, hashstr2):
        if i != j:
            n += 1
    return n


if __name__ == '__main__':
    original = cv2.imread('lenna.png')
    noise = cv2.imread('lenna_noise.png')
    # meanhash
    original_meanhash = meanhash(original)
    noise_meanhash = meanhash(noise)
    print(original_meanhash, 'original_meanhash')
    print(noise_meanhash, 'noise_meanhash')
    mean_cmp = cmphash(original_meanhash, noise_meanhash)
    print(mean_cmp, 'mean_cmp')

    original_diffhash = diffhash(original)
    noise_diffhash = diffhash(noise)
    print(original_diffhash, 'original_diffhash')
    print(noise_diffhash, 'noise_diffhash')
    diff_cmp = cmphash(original_diffhash, noise_diffhash)
    print(diff_cmp, 'diff_cmp')

