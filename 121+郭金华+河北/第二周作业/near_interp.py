import numpy as np
import cv2
def Near_inter(img):
    img_shape=img.shape
    img_interp=np.zeros((500,500,img_shape[2]),img.dtype)
    temp_sw=500/img_shape[1]
    temp_sh=500/img_shape[0]
    for i in range(500):
        for j in range(500):
            x=int(i/temp_sh)
            y=int(j/temp_sw)
            img_interp[i,j]=img[x,y]
    return img_interp

img=cv2.imread('lenna.png')
img_interp=Near_inter(img)
print(img_interp.shape)
cv2.imshow('interp_img',img_interp)
cv2.imshow('img',img)
cv2.waitKey()
cv2.destroyAllWindows()