# coding: utf-8
import cv2
import numpy as np
import random
'''
该文件实现高斯噪声
'''
def GasussNoise(img,mean=0.1,var=0.05,ratio=0.8):
    '''
    img:输入的图像
    mean：高斯均值
    var：高斯方差
    ratio：加入高斯噪声像素点占图像总像素点的比例
    '''
    h,w,c=img.shape
    noise_num=int(h*w*ratio)
    #随机取图像的索引点
    X=np.random.choice(range(w),size=noise_num)
    Y=np.random.choice(range(h),size=noise_num)
    #归一化0-1之间在操作
    img=img/255.0
    #高斯分布
    noise = np.random.normal(mean, var, noise_num)
    dst_img=img.copy()
    for x,y,n in zip(list(X),list(Y),list(noise)):
        dst_img[y,x,:]=img[y,x,:]+n
    dst_img = np.clip(dst_img, 0, 1)
    dst_img = np.uint8(dst_img * 255)
    return  dst_img
if __name__=='__main__':
    img=cv2.imread('lenna.png')
    h,w,c=img.shape
    dstimg=GasussNoise(img)
    showimg = np.zeros((h, 2 * w, c), dtype=np.uint8)
    showimg[:, 0:w] = img
    showimg[:, w:2 * w] = dstimg
    cv2.imshow('result',showimg)
    cv2.waitKey()

