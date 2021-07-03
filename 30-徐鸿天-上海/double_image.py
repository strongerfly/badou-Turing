from PIL import Image
# import matplotlib.pyplot as plt
import numpy as np
import math

def BiLinear_interpolation(img,dstH,dstW):
     scrH,scrW,_=img.shape
     img=np.pad(img,((0,1),(0,1),(0,0)),'constant')
     retimg=np.zeros((dstH,dstW,3),dtype=np.uint8)
     for i in range(dstH):
         for j in range(dstW):
             scrx=(i+0.5)*(scrH/dstH)-0.5
             scry=(j+0.5)*(scrW/dstW)-0.5
             x=math.floor(scrx)
             y=math.floor(scry)
             u=scrx-x
             v=scry-y
             retimg[i,j]=(1-u)*(1-v)*img[x,y]+u*(1-v)*img[x+1,y]+(1-u)*v*img[x,y+1]+u*v*img[x+1,y+1]
     return retimg

im_path= '../../bd_ai/30-徐鸿天-上海/1.jpg'
image=np.array(Image.open(im_path))
image2=BiLinear_interpolation(image,image.shape[0]*2,image.shape[1]*2)
image2=Image.fromarray(image2.astype('uint8')).convert('RGB')
image2.save('out2.png')