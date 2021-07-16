#    @author Created by Genius_Tian

#    @Date 2021/7/4

#    @Description 图像直方图
import cv2
from matplotlib import pyplot as plt
import numpy as np

"""
画直方图
"""


def gray_histogram1(img):
    plt.hist(img.ravel(), 256)
    plt.show()


def gray_histogram2(img):
    hist = cv2.calcHist(img, [0], None, [256], [0, 256])
    plt.plot(hist, color='gray')
    plt.show()


def rgb_histogram(img):
    channels = cv2.split(img)
    color = ['b', 'g', 'r']
    for chan, c in zip(channels, color):
        hist = cv2.calcHist([chan], [0], None, [256], [0, 256])
        plt.plot(hist, color=c)
        plt.xlim([0, 256])
    plt.show()


if __name__ == '__main__':
    image = cv2.imread(r"../resources/lenna.png", 0)
    gray_histogram1(image)
    gray_histogram2(image)
    rgb_histogram(image)
