import cv2
import numpy as np
from PIL import Image
from skimage import util

img = cv2.imread('lenna.png')
noise_gs_img = util.random_noise(img,mode='poisson')
cv2.imshow('sourse',img)
cv2.imshow('lenna_noise',noise_gs_img)
cv2.waitKey(0)
cv2.destroyAllWindows()