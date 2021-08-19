import cv2
import random

img = cv2.imread('../img/lenna.png', 1)
mu = 0
sigma = 4


def show(img, name):
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


show(img, 'origin')


def jiaoyanNoise(img, percentage):
    noise = img
    number = int(noise.shape[2] * img.shape[0] * img.shape[1] * percentage)
    for i in range(number):
        randomChannel = random.randint(0, img.shape[2] -1)
        randomW = random.randint(0, img.shape[0] -1)
        randomH = random.randint(0, img.shape[1] -1)
        if random.random() <= 0.5:
            noise[randomW, randomH, randomChannel] = 255
        else:
            noise[randomW, randomH, randomChannel] = 0
    return noise


show(jiaoyanNoise(img, 0.2), 'jiaoyan')