# -*- coding: utf-8 -*-
"""
Created on Sun Jul  4 21:17:53 2021

@author: wuming
"""

import cv2
import numpy as np
def nearest_interp(image,nH,nW):
    height,width,channels=image.shape
    zeroImage=np.zeros((nH,nW,channels),np.uint8)
    sh=height/nH
    sw=width/nW
    for i in range(nH):
        for j in range(nW):
            zeroImage[i,j]=image[int(i*sh),int(j*sw)]
    return zeroImage

img=cv2.imread("lenna.png")
result_image=nearest_interp(img,800,800)
cv2.imshow("result_image",result_image)
cv2.waitKey(0)