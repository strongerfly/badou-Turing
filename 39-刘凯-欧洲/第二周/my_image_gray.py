import cv2
import numpy as np

def my_2gray(input_image):
    '''
    convert the image to gray
    :param input_image: input image as numpy array
    :return: grayscale image
    '''

    height, width, _ = input_image.shape
    grayImage = np.zeros((height, width, 1), np.uint8)
    for i in range(height):
        for j in range(width):
            m = input_image[i, j]
            grayImage[i, j] = int(m[0] * 0.11 + m[1] * 0.59 + m[2] * 0.3)
    return grayImage

img = cv2.imread("lenna.png")

gray_img = my_2gray(img)

cv2.imshow("image gray", gray_img)
cv2.imshow("image orignal", img)

cv2.waitKey(0)