from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import math
import cv2
def BiLinear_interpolation(img,dstH,dstW):
    scrH,scrW,channel=img.shape
    print("src_h, src_w = ", scrH,scrW)
    img=np.pad(img,((0,1),(0,1),(0,0)),'constant')  #nupy数组边缘填充
    retimg=np.zeros((dstH,dstW,3),dtype=np.uint8)
    for i in range(dstH):
        for j in range(dstW):
            scrx=(i+0.5)*(scrH/dstH)-0.5
            scry=(j+0.5)*(scrW/dstW)-0.5
            x=math.floor(scrx)  #取整
            y=math.floor(scry)  #取整
            u=scrx-x
            v=scry-y

            retimg[i,j]=(1-u)*(1-v)*img[x,y]+u*(1-v)*img[x+1,y]+(1-u)*v*img[x,y+1]+u*v*img[x+1,y+1]
    return retimg
im_path='lenna.png'
image=np.array(Image.open(im_path))
img =cv2.imread("lenna.png")

image2=BiLinear_interpolation(image,image.shape[0]*2,image.shape[1]*2)   #image.shape[1]*2
image2=Image.fromarray(image2.astype('uint8')).convert('RGB')
image2.save('out.png')
img = cv2.imread("out.png")
cv2.imshow("image show ",img)
cv2.waitKey()