
import numpy as np
import matplotlib.pyplot as plt
import cv2


def ahash(image1):
    img1= cv2.resize(image1,(8,8),interpolation=cv2.INTER_LINEAR)
    gray1 = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
    h,w = gray1.shape
    _mean = np.sum(gray1)/(h*w)
    print(_mean)
    str =''
    for ii in range(8):
        for jj in range(8):
            if _mean < gray1[ii,jj]:
                str = str +"1"
            else:
                str = str + "0"

    return str

def bhash(image1):
    img1= cv2.resize(image1,(9,8),interpolation=cv2.INTER_LINEAR)
    gray1 = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)

    str =''
    for ii in range(8):
        for jj in range(8):
            if gray1[ii,jj+1] < gray1[ii,jj]:
                str = str +"1"
            else:
                str = str + "0"

    return str


def comp(str1,str2):
    n = 0
    if len(str1) !=len(str2):
        return -1
    for i in range(len(str1)-1):
        if str1[i] !=str2[i]:
            n += 1
    return n

img1 = cv2.imread("lenna.png")
str1 = ahash(img1)
img2 = cv2.imread("lenna_noise.png")
str2 = ahash(img2)
# print(str2)
# print(str1)

print(comp(str1,str2))

str11 = bhash(img1)
str22 = bhash(img2)
print(comp(str11,str22))








