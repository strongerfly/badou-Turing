#-*- coding: utf-8 -*-
import cv2
import numpy as np
import math
import time
def bilinear_resize(img,dstw=1920,dsth=1080):
    '''
    func:双线性插值
    img：输入的图像（BGR格式）
    dstw，dsth：缩放后的图像尺寸w，h
    '''
    h,w,c=img.shape
    dst_img=np.zeros((dsth,dstw,c),dtype=np.uint8)
    #缩放因子
    scale_x = h / dsth
    scale_y = w / dstw
    start_time=time.time()
    for j in range(dsth):
        for i in range(dstw):
            #1.1先找到目标图像相对原图的位置
            #src_x=(dst_x+0.5)*scale_x-0.5
            srcx=(i+0.5)*scale_x-0.5
            srcy=(j+0.5)*scale_y-0.5
            #防止越界
            if srcx<0:
                srcx=0
            if srcx>=w-1:
                srcx=srcx-2
            if srcy < 0:
                srcy = 0
            if srcy >= h - 1:
                srcy = srcy - 2

            #1.2找（src_x和src_y）附近的四个点进行插值，这里直接使用公式了，方便，u，v为小数部分,f(i,j)为原图像像素
            # f(i + u, j + v) = (1 - u)(1 - v)
            # f(i, j) + (1 - u)
            # vf(i, j + 1) + u(1 - v)
            # f(i + 1, j) + uvf(i + 1, j + 1)
            #浮点运行比较耗时，我们使用整数方法计算会节省时间，将其乘以2048转换为整数计算，貌似这里不省多少时间，
            src_left,src_top,src_right,src_bot=int(srcx),int(srcy),int(srcx)+1,int(srcy)+1
           # print("src_bot:",src_bot)
           #  u,v=srcx-src_left,srcy-src_top
           #  for c in range(3):
           #      dst_img[j,i,c]=(1-u)*(1-v)*img[src_top,src_left,c]+(1-u)*v*img[src_bot,src_left,c]\
           #                 +u*(1-v)*img[src_top,src_right,c]+u*v*img[src_bot,src_right,c]
            u=(srcx-src_left)*2048
            v=(srcy-src_top)*2048
            u_u=2048-u
            v_v=2048-v
            for c_ in range(c):
                 dst_img[j,i,c_]=(u_u*v_v*img[src_top,src_left,c_]+u_u*v*img[src_bot,src_left,c_]\
                            +u*v_v*img[src_top,src_right,c_]+u*v*img[src_bot,src_right,c_])/4194304
    print("cost time:{} ms".format((time.time()-start_time)*1000))#21268ms 4k resize 720p
    cv2.imshow("resize",dst_img)
    #cv2.imwrite("resize1.jpg",dst_img)
    cv2.waitKey()
if __name__=='__main__':
    img=cv2.imread("resize.jpg")
    bilinear_resize(img)
