import matplotlib.pyplot as plt
import numpy as np
import math
import cv2
import random
def cv_imshow(img,name):
    cv2.imshow(name,img)
    cv2.waitKey()
    cv2.destroyAllWindows()

def gasuss_noise(image, mean=0, var=0.001):

    '''

        添加高斯噪声

        mean : 均值

        var : 方差

    '''

    image = np.array(image/255, dtype=float)

    noise = np.random.normal(mean, var ** 0.5, image.shape)

    out = image + noise

    if out.min() < 0:

        low_clip = -1.

    else:

        low_clip = 0.

    out = np.clip(out, low_clip, 1.0)

    out = np.uint8(out*255)

    #cv.imshow("gasuss", out)

    return out

# Read image

img = cv2.imread("lena.png")
gasuss_img = gasuss_noise(img, mean=0, var=0.02)

cv_imshow(gasuss_img,"name")