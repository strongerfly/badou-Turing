import numpy as np
import cv2
'''
均值hash和差值hash实现
'''

def ahash(img):
    resize_img = cv2.resize(img, (8, 8))
    g_img = cv2.cvtColor(resize_img, cv2.COLOR_BGR2GRAY)

    img_mean = g_img.mean()
    print(img_mean)
    hash_list = []
    for i in range(8):
        for j in range(8):
            if g_img[i, j] > img_mean:
                hash_list.append(1)
            else:
                hash_list.append(0)
    return hash_list

def dhash(img):
    resize_img = cv2.resize(img, (9, 8))
    g_img = cv2.cvtColor(resize_img, cv2.COLOR_BGR2GRAY)
    print(g_img.shape)
    dhash_list = []
    for i in range(8):
        for j in range(8):
            if g_img[i, j] > g_img[i, j+1]:
                dhash_list.append(1)
            else:
                dhash_list.append(0)
    return dhash_list

def comparehash(hash1, hash2):
    n = 0
    #判断长度是否相等
    if len(hash1) != len(hash2):
        return -1
    for i in range(len(hash1)):
        if hash1[i] != hash2[i]:
            n += 1
    return n

img = cv2.imread("./lenna.png")
img1 = cv2.imread("./lenna_noise.png")
img_ahash = ahash(img)
img1_ahash = ahash(img1)
n = comparehash(img_ahash, img1_ahash)
print("均值hash的汉明距离为", n)

img_dhash = dhash(img)
img1_dhash = dhash(img1)
n1 = comparehash(img_dhash, img1_dhash)
print("差值hash的汉明距离为", n1)
