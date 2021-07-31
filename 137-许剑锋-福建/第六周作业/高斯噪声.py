import cv2
import random

img = cv2.imread('../img/lenna.png', 1)
mu = 0
sigma = 4


def show(img, name):
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def guassion():

    return random.gauss(mu, sigma)


show(img, 'origin')


def guassionNoise(img):
    noise = img
    for i in range(noise.shape[2]):
        channel = noise[:, :, i]
        for w in range(channel.shape[0]):
            for h in range(channel.shape[1]):
                channel[w, h] = channel[w, h] + guassion()
                if channel[w, h] > 255:
                    channel[w, h] = 255
                if channel[w, h] < 0:
                    channel[w, h] = 0
    return noise


show(guassionNoise(img), 'guassion')