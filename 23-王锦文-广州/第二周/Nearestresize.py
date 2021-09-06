#-*- coding: utf-8 -*-
import cv2
import numpy as np
import math
import time
def nearest_resize(img,dstw=1920,dsth=1080):
    '''
    func:最近邻插值
    img：BGR图像
    dstw，dsth：缩放后的尺寸w，h
    '''
    h,w,c=img.shape
    dstimg=np.zeros((dsth,dstw,c),dtype=np.uint8)
    scalex=dstw/w
    scaley=dsth/h

    for j in range(dsth):
        for i in range(dstw):
            #找到目标图相对原图的坐标,并防止越界
            u=min(round(i/scalex),w-1)
            v=min(round(j/scaley),h-1)
            dstimg[j, i] = img[v, u]
            #貌似不用for循环通道也可以的，因为numpy具备这样的机制
            # for c_ in range(c):
            #     dstimg[j,i,c]=img[v,u,c]
    cv2.imshow("dstimg",dstimg)
    cv2.waitKey()
if __name__=='__main__':
    img=cv2.imread("resize.jpg")
    nearest_resize(img)
