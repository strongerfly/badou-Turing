#调用opencv中的canny算子
import cv2

img = cv2.imread("./lenna.png")
cv2.imshow('test', img)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray_canny = cv2.Canny(gray, 100, 300)
cv2.imshow("canny", gray_canny)
cv2.waitKey(0)