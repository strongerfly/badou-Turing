# -*- coding: utf-8 -*-
"""
Created on Sun Jul  4 22:25:50 2021

@author: wuming
"""

import cv2
import numpy as np

def bilinear_interp(scr_img,out_dim=(800,800)):
    height,width,channels=scr_img.shape
    dst_h,dst_w=out_dim[0],out_dim[1]
    if dst_h==height and dst_w==width:
        return scr_img.copy()
    scale_h,scale_w=dst_h/height,dst_w/width
    dst_img=np.zeros((dst_h,dst_w,channels),dtype=np.uint8)
    for i in range(channels):
        for dst_y in range(dst_h):
            for dst_x in range (dst_w):
#                几何中心相等
                scr_x=(dst_x+0.5)/scale_w-0.5
                scr_y=(dst_y+0.5)/scale_h-0.5
#                相邻电点的坐标
                scr_x0=int(np.floor(scr_x))
                scr_x1=min(scr_x0+1,width-1)
                scr_y0=int(np.floor(scr_y))
                scr_y1=min(scr_y0+1,height-1)
#                插值
                temp0 = (scr_x1 - scr_x) * scr_img[scr_y0,scr_x0,i] + (scr_x - scr_x0) * scr_img[scr_y0,scr_x1,i]
                temp1 = (scr_x1 - scr_x) * scr_img[scr_y1,scr_x0,i] + (scr_x - scr_x0) * scr_img[scr_y1,scr_x1,i]
                dst_img[dst_y,dst_x,i] = int((scr_y1 - scr_y) * temp0 + (scr_y - scr_y0) * temp1)
    return dst_img

if __name__=='__main__':
    
    img = cv2.imread('lenna.png')
    dst = bilinear_interp(img,(600,600))
    cv2.imshow('bilinear interp',dst)
    cv2.waitKey(0)        
    