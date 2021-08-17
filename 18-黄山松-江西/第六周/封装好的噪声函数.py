import cv2
import numpy as np
from PIL import Image
from skimage import util

img = cv2.imread('zzhang.png')
# noise_gs_img = util.random_noise(img, mode='poisson')
noise_gs_img = util.random_noise(img, mode='s&p', amount=0.2)

cv2.imshow('source', img)
cv2.imshow('lenna', noise_gs_img)
cv2.imwrite('New_pic1.png', noise_gs_img)
cv2.waitKey(0)
cv2.destroyAllWindows()