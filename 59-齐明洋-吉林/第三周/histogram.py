import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread(r'..\Second_Week\lenna.png',1)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#cv2.imshow('gray',gray)
#cv2.waitKey(0)
plt.figure(figsize=(8,8),facecolor='blue')
plt.hist(gray.ravel(),256)
plt.show()
#####
