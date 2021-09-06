import numpy as np
import cv2

'''***************************代码实现rgb2gray**************************************'''
img=cv2.imread('lenna.png')

b,g,r=cv2.split(img)
source=False
if source:
    img_gray=np.round(0.2*b+0.5*g+0.3*r).astype(img.dtype)

else:
    img_gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)


cv2.imshow('img_gray',img_gray)
cv2.imshow('source_img',img)
cv2.waitKey()
cv2.destroyAllWindows()





