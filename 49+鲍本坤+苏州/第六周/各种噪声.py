import cv2 as cv
import numpy as py
#Image模块是在Python PIL图像处理中常见的模块，对图像进行基
# 础操作的功能基本都包含于此模块内。如open、save、conver、show…等功能
from PIL import Image

from skimage import util


img = cv.imread('lenna.png',0)
noise_gs_img=util.random_noise(img,mode='poisson')
cv.imshow('src',img)
cv.imshow('dst',noise_gs_img)
cv.waitKey(0)
cv.destroyAllWindows()