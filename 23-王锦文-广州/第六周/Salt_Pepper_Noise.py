# coding: utf-8
import cv2
import numpy as np
import random
'''
该文件实现椒盐噪声
'''
def salt_pepper_noise(img,ratio=0.1):
    '''
    img:输入的图像
    ratio：加入噪声像素点占图像总像素点的比例
    '''
    h,w,c=img.shape
    noise_num=int(h*w*ratio)
    #先取出需要加噪声的图像索引
    X=np.random.choice(range(w),size=noise_num)
    Y=np.random.choice(range(h),size=noise_num)
    dst_img=img.copy()
    #对取出需要加噪声的图像索引对应像素操作
    for x,y in zip(list(X),list(Y)):
        if random.random()<=0.5:
            dst_img[y,x,:]=0
        else:
            dst_img[y,x,:]=255
    return  dst_img
if __name__=='__main__':
    img=cv2.imread('lenna.png')
    h,w,c=img.shape
    dstimg=salt_pepper_noise(img)
    showimg = np.zeros((h, 2 * w, c), dtype=np.uint8)
    showimg[:, 0:w] = img
    showimg[:, w:2 * w] = dstimg
    cv2.imshow('result',showimg)
    cv2.waitKey()

