#-*- coding: utf-8 -*-
import cv2
import numpy as np
import math
import time
def RGB2Gray(img):
    '''
    func:RGB图像转灰度图
    img：输入的图像（BGR格式）
    return ：None
    这里只实现2种方式
    gray=R*0.3+G*0.59+B*0.11
    Gray=（R*76+G*151+B*26）>>8
    '''
    h,w,_=img.shape
    print(img.shape)
    dstimg=np.zeros((h,w,1),dtype=np.uint8)


    #方式1
    start_time=time.time()
    dstimg[:, :, 0] = img[:, :, 0] * 0.11 + img[:, :, 1] * 0.59 + img[:, :, 2] *0.3
    end_time=time.time()

    #方式2
    # #for 循环比较耗时
    # for j in range(h):
    #     for i in range(w):
    #         dstimg[j,i,0]=(img[j,i,0]*26+img[j,i,1]*151+img[j,i,2]*76)>>8
    dstimg[:, :, 0] = np.sum(img * np.array([26, 151, 76]), axis=2) >> 8
    #17ms和21ms
    print("float cost time:{}ms,displacement cost time:{}ms".format((end_time-start_time)*1000,(time.time()-end_time)*1000))

    cv2.imshow("gray",dstimg)
    cv2.waitKey()
if __name__=='__main__':
    img=cv2.imread("resize.jpg")
    RGB2Gray(img)