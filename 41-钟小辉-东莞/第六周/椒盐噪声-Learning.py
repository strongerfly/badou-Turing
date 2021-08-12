import random

import cv2
import numpy as np

def GaussNoise(src,percentage):
    h,w = src.shape
    img_noise =src.copy()
    noise_number = int(percentage*h*w)
    for i in range(noise_number):
        random_x = random.randint(0,h-1)
        random_y = random.randint(0,w-1)
        if random.random() < 0.5:
            img_noise[random_x,random_y] = 0
        else:
            img_noise[random_x,random_y] = 255

    return    img_noise

img = cv2.imread("lenna.png",0)
img2 = GaussNoise(img,0.1)
cv2.imshow("image",img)
cv2.imshow("image2",img2)
cv2.waitKey(0)
cv2.destroyAllWindows()
