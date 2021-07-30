import cv2
import numpy as np

img = cv2.imread('zzhang.png', 1)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
result = cv2.Canny(gray, 10, 150)
cv2.imshow("canny", result)
k = cv2.waitKey(0)
if k == 27:
    cv2.destroyAllWindows()
elif k == ord('s'):
    cv2.imwrite('zzhang1.png', result)
cv2.waitKey(0)