import cv2
import numpy as np
import matplotlib.pyplot as plt

img=cv2.imread('lenna.png',0)
data=img.reshape((-1,1))
data=np.float32(data)
compactness,labels,centers=cv2.kmeans(data,4,None,(cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER,10,1.0),10,cv2.KMEANS_RANDOM_CENTERS)

center=np.uint8(centers)

res=center[labels]

res=res.reshape((img.shape))

cv2.imshow('res',res)
cv2.waitKey()


