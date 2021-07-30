import numpy as np
import cv2
import random

def function(src,sigma,mean,percetage):
    w,h=src.shape[0:2]
    dst=src
    NoiseNum=int(percetage*w*h)
    for i in range (NoiseNum):
        x=random.randint(0,w-1)
        y=random.randint(0,h-1)
        dst[x,y]=dst[x,y]+random.gauss(mean,sigma)
        # 若灰度值小于0则强制为0，若灰度值大于255则强制为255
        if dst[x,y]<0:
            dst[x, y]=0
        elif dst[x,y]>255:
            dst[x,y]=255
    return  dst


    return


if __name__=="__main__":
   img=cv2.imread("lenna.png",0)
   dst=function(img,2,4,0.8)

   img = cv2.imread("lenna.png", 1)
   dst2 = function(img, 2, 4, 0.8)
   img = cv2.imread('lenna.png')
   img2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
   cv2.imshow("src",img2)
   cv2.imshow("dst", dst)
   cv2.imshow("dst2", dst2)
   cv2.waitKey(0)