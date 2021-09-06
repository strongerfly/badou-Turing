import cv2
import numpy as np
from matplotlib import pyplot as plt
'''
img = cv2.imread(r'..\Second_Week\lenna.png')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

dst = cv2.equalizeHist(gray)

hist = cv2.calcHist([dst],[0],None,[256],[0,255])

plt.figure()
plt.hist(dst.ravel(),256)
plt.show()

cv2.imshow('Histogram Equalization',np.hstack([gray,dst]))
cv2.waitKey(0)'''

img = cv2.imread(r'..\Second_Week\lenna.png')
(b,g,r) = cv2.split(img)
bh = cv2.equalizeHist(b)
gh = cv2.equalizeHist(g)
rh = cv2.equalizeHist(r)

results = cv2.merge((bh,gh,rh))
cv2.imshow('color equalization',results)
cv2.waitKey(0)