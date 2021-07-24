import numpy as np
import cv2 as cv


'''
src：输入图像的三个点坐标
dst：输出图像的三个点坐标
三个点分别对应左上角、右上角、左下角
'''
img = cv.imread('ddk.jpg', 1)
img = cv.resize(img,(800,500))
img2 = img.copy()
rows, cols, channels = img.shape
p1 = np.float32([[0,0], [cols-1,0], [0,rows-1]])
p2 = np.float32([[0,rows*0.3], [cols*0.8,rows*0.2], [cols*0.15,rows*0.7]])
M = cv.getAffineTransform(p1, p2)  #从p1到p2
dst = cv.warpAffine(img, M, (cols,rows))
cv.imshow('original', img)
cv.imshow('text', img2)
cv.imshow('result', dst)
cv.waitKey(0)
cv.destroyAllWindows()

'''
dst：透视后的输出图像，dsize决定输出图像大小
src：输入图像
M：3*3变换矩阵
flags：插值方法，默认为INTER_LINEAR
borderMode：边类型，默认为BORDER_CONSTANT
borderValue：边界值，默认为0
'''
# p1 = np.float32([[0,0], [cols-1,0], [0,rows-1], [rows-1,cols-1]])
# p2 = np.float32([[0,rows*0.3], [cols*0.8,rows*0.2], [cols*0.15,rows*0.7], [cols*0.8,rows*0.8]])
# M = cv.getPerspectiveTransform(p1,p2)
# dst = cv.warpPerspective(img, M, (cols, rows))
# cv.imshow('original', img)
# cv.imshow('result', dst)
# cv.waitKey(0)
# cv.destroyAllWindows()