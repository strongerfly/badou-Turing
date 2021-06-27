import cv2
import numpy as np
# 最邻近插值The nearest interpolation
def function(img):
    h,w=img.shape[:2]
    dst=np.zeros((700,700,3),np.uint8)
    for i in range(700):
        for j in range(700):
            x=round(i/700*w)
            y=round(j/700*h)
            # 防止下标溢出
            if(x==h):
                x=h-1
            if y==w:
                y=w-1
            dst[i][j]=img[x][y]
    return dst

if __name__=="__main__":
    img = cv2.imread('./lenna.png')
    newImg=function(img)
    cv2.imshow('old', img)
    cv2.imshow('nearest', newImg)
    cv2.waitKey()
