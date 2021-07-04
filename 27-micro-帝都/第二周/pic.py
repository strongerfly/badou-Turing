import cv2
import numpy as np

def function(img,output):
    height,width,channels = img.shape
    outputImage = np.zeros((output[0],output[1],channels),np.uint8)
    sh = output[0]/height
    sw = output[1]/width
    for i in range(output[0]):
        for j in range(output[1]):
            x = int(i/sh)
            y = int(j/sw)
            outputImage[i,j]=img[x,y]
    return outputImage

img = cv2.imread("lenna.png")
out = function(img,(200,200))
print(out)
print(out.shape)
cv2.imshow("ori",img)
cv2.imshow("out",out)
cv2.waitKey()