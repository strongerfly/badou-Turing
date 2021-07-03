import cv2
import matplotlib.pyplot as plt
import numpy as np


def cv_show(name,img):
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


img  = cv2.imread("lenna.png")
img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# cv_show("image",img)
print(img_gray[0:5,0:5])

# retval,img_gray2=cv2.threshold(img_gray,127,255,cv2.THRESH_BINARY)
# cv_show("image",img_gray2)

h,w = img_gray.shape
for ii in range(h):
    for jj in range(w):
        if img_gray[ii,jj] <=127:
            img_gray[ii, jj] = 0
        else:
            img_gray[ii, jj] = 255

cv_show("image",img_gray)