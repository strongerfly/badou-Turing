import cv2
import numpy as np

def nearest_interpolation(img):
    h,w,channel = img.shape
    emptyIamge = np.zeros((800,800,channel),np.uint8)
    sh = 800/h
    sw = 800/w
    for i in range(800):
        for j in range(800):
            x= int(i/sh)
            y = int(j/sw)
            emptyIamge[i,j] = img[x,y]
    return emptyIamge


img = cv2.imread('lenna.png')
nir = nearest_interpolation(img)
print(nir)
print(nir.shape)
cv2.imshow('nearest interpolation',nir)
cv2.imshow('original',img)
cv2.waitKey(0)