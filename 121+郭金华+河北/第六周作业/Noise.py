import cv2
import random

def gauss_noise(img):
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            gauss=random.gauss(0,0.5)
            img[i][j]=img[i][j]+gauss
            if img[i][j]>255:
                img[i][j]=255
            elif img[i][j]<0:
                img[i][j]=0
    return img

def salt(img,per):
    for k in range(int(per * img.shape[0] * img.shape[1])):
        i = random.randint(0, img.shape[0] - 1)
        j = random.randint(0, img.shape[1] - 1)
        if random.randint(0, 1) > 0.5:
            img[i][j] = 0
        else:
            img[i][j] = 255
    return img

img=cv2.imread('lenna.png',0)
cv2.imshow('source_img',img)
img_gauss=gauss_noise(img)
cv2.imshow('gaus_noise',img_gauss)
img_salt=salt(img,0.1)
cv2.imshow('img_noise',img_salt)
cv2.waitKey()









