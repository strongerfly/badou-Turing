import  cv2
import numpy as np
import math

def function(sh,sw,img):
    l,w,d = img.shape
    L = math.floor(l * sh)
    W = math.floor(w * sw)
    zerosImg = np.zeros((L, W, d), img.dtype)
    for i in range(L):
        for j in range(W):
            il = math.floor(i / sh)
            jw = math.floor(j / sw)
            zerosImg[i, j] = img[il, jw]
    return zerosImg


lenna = cv2.imread("lenna.png")
zoom = function(0.5, 0.5, lenna)

cv2.imshow("nearest interp", zoom)
cv2.imshow("image", lenna)
cv2.waitKey(0)




