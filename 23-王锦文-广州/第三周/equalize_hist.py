#-*- coding: utf-8 -*-
import cv2
import numpy as np
import math

'''
equalize_hist.py 直方图均衡化实现：以灰度图为例,彩色图只是先将通道分离分别进行直方图均衡化后再merge
步骤：1.统计图像每个灰度级下像素点个数
2.计算每个灰度级像素点占总像素点比例（累计密度）
3.根据映射函数计算每个灰度级下新的灰度值(均衡化处理)
4.用新的灰度值表更新图像灰度值
'''
def equalize_hist(img):
    h,w=img.shape
    total_num=h*w
    equalize_img = np.zeros(img.shape, dtype=img.dtype)
    pixratio=0.
    for i in range(256):
        #1.统计图像每个灰度级下像素点个数
        #pixindex=np.sum(np.where(img==i,1,0))
        pixindex=np.where(img==i)#返回的是索引
        #2.计算每个灰度级像素点占比,,累计密度
        pxinum=len(img[pixindex])
        pixratio+=pxinum/total_num
        #3.均衡化处理,+0.5是为了取整
        new_val=255*pixratio+0.5
        #用新的值更新原图对应的灰度值
        equalize_img[pixindex]=new_val
    return  equalize_img

if __name__=='__main__':
    img=cv2.imread('lenna.png',cv2.IMREAD_GRAYSCALE)
    h,w=img.shape
    showimg = np.zeros((h,2*w), dtype=img.dtype)
    equalize_img=equalize_hist(img)
    showimg[:, 0:w] = img
    showimg[:, w:2 * w] = equalize_img
    cv2.imshow("result", showimg)
    cv2.waitKey()

        


    
