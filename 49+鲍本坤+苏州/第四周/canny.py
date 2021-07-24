import cv2
import numpy as np
from matplotlib import pyplot as plt


img = cv2.imread('ddk.jpg',1)
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
result = cv2.Canny(gray,100,500)
# plt.subplot(131),plt.imshow(result,"gray")
# plt.subplot(132),plt.imshow(img,"gray")
# plt.subplot(133),plt.imshow(gray,"gray")
# plt.show()
img = cv2.resize(img,(800,500))
result = cv2.resize(result,(800,500))
# hmerge = np.hstack((img,result))
cv2.imshow("src",img)
cv2.imshow("result",result)
cv2.waitKey()
cv2.destroyAllWindows()
