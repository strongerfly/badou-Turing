import matplotlib.pyplot as plt
import numpy as np
import math
import cv2
import random
def cv_imshow(img,name):
    cv2.imshow(name,img)
    cv2.waitKey()
    cv2.destroyAllWindows()

def sapn_noise(image,prob):

    '''

    添加椒盐噪声

    prob:噪声比例

    '''

    output = np.zeros(image.shape,np.uint8)

    thres = 1 - prob

    for i in range(image.shape[0]):

        for j in range(image.shape[1]):

            rdn = random.random()

            if rdn < prob:

                output[i][j] = 0

            elif rdn > thres:

                output[i][j] = 255

            else:

                output[i][j] = image[i][j]

    return output

img = cv2.imread("lena.png")
sp_img = sapn_noise(img,prob=0.1)
cv_imshow(sp_img,"name")