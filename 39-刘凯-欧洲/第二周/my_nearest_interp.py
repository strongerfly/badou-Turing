import cv2
import numpy as np

def my_nearest_interp(input_img, target_size):
    '''
    nearest interpolation
    :param input_img: input image as numpy array
    :param target_size: the target size for interpolation
    :return: image after nearest interpolation
    '''
    height, width, channels = img.shape
    emptyImage = np.zeros((target_size, target_size, channels), np.uint8)
    sh = target_size / height
    sw = target_size / width
    for i in range(target_size):
        for j in range(target_size):
            x = int(i / sh)
            y = int(j / sw)
            emptyImage[i, j] = img[x, y]
    return emptyImage

img = cv2.imread("lenna.png")

zoom_small = my_nearest_interp(img, 320)
zoom_large = my_nearest_interp(img, 1024)

cv2.imshow("nearest interp zoom small", zoom_small)
cv2.imshow("nearest interp zoom large", zoom_large)
cv2.imshow("image orignal", img)

cv2.waitKey(0)