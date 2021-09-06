# canny实现
import cv2

img = cv2.imread('img/lenna.png', 1)
cv2.imshow('img', img)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
can = cv2.Canny(gray, 200, 300)
cv2.imshow('canny', can)

cv2.waitKey(0)
cv2.destroyAllWindows()
