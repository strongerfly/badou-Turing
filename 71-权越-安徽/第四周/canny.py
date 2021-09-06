import cv2 as cv

img=cv.imread("./lenna.png")
# 三通道转单通道
X_train_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
# 200 阈值下限，300 阈值上线
# canny
# 图片转灰度
# 高斯滤波
# nms 过滤重复边缘
# 双界阈值确定强弱边缘，并对孤立弱边缘进行已知，非孤立点与强边缘进行连接
canny_img1=cv.Canny(X_train_gray,200,300)
cv.imshow('canny1', canny_img1)

# 先高斯滤波处理在交给canny
canny_img2=cv.Canny(cv.GaussianBlur(img,(3,3),0),200,300)
cv.imshow('canny2', canny_img2)

cv.waitKey()


