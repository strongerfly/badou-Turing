
import numpy as np
import cv2
import matplotlib.pyplot as plt

def nearest_interp(orgImagPath,destHeight,destWidth):
    orgImage = cv2.imread(orgImagPath)
    orgHeight,orgWidth,orgChannels = orgImage.shape
    destImage = np.zeros((destHeight,destWidth,orgChannels),np.uint8)
    sh = destHeight/orgHeight
    sw = destWidth/orgWidth
    for i in range(destHeight):
        for j in range(destWidth):
            x = int(i/sh)
            y = int(j/sw)
            destImage[i,j] = orgImage[x,y]
    return  destImage

testImagePath = "lenna.png"
orgImage = plt.imread(testImagePath)
tempImage800 = nearest_interp(testImagePath,800,800)
tempImage400 = nearest_interp(testImagePath,400,400)
plt.subplot(131)
plt.imshow(orgImage)
plt.subplot(132)
plt.imshow(tempImage800)
plt.subplot(133)
plt.imshow(tempImage400)
plt.savefig("nearest_interp")
plt.show()
