
import cv2
import numpy as np
def function(img, dstW, dstH):
    height,width,channels = img.shape
    emptyImage=np.zeros((destW,dstH,channels),np.uint8)
    sh=dstH/height #高度缩放比例
    sw=dstW/width  #宽度缩放比例
    for i in range(dstH):
        for j in range(dstW):
            x=int(i/sh)
            y=int(j/sw)
            emptyImage[i,j]=img[x,y]
    return emptyImage

img=cv2.imread("lenna.png")
zoom=function(img,800,800)
print(zoom)
print(zoom.shape)
cv2.imshow("image",img)
cv2.imshow("nearest interp",zoom)
cv2.waitKey(0)

