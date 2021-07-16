from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import math


def NN_interpolation(img, dstH, dstW):
    scrH, scrW, chanel = img.shape
    retimg = np.zeros((dstH, dstW, chanel), dtype=np.uint8)
    for i in range(dstH-1):
        for j in range(dstW-1):
            scrx = round((i ) * (scrH / dstH))
            scry = round((j ) * (scrW / dstW))
            retimg[i, j] = img[scrx,scry]
    return retimg





d