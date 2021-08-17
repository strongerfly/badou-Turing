import cv2
import numpy as np
import matplotlib.pyplot as plt

def histogram(image):
    hist = np.zeros(256)
    height, width = image.shape
    for i in range(height):
        for j in range(width):
            hist[image[i, j]] += 1

    return hist

image = cv2.imread("lenna.png")
cv2.imshow('Original Image', image)
cv2.waitKey()
cv2.destroyAllWindows()

channels = cv2.split(image)
colors = ('b', 'g', 'r')
plt.figure()
plt.title('3 Channels Histogram')
plt.xlabel('Bins')
plt.ylabel('# of Pixels')

for (channel, color) in zip(channels, colors):
    # Histogram = cv2.calcHist([channel], [0], None, [256], [0, 256])
    # plt.plot(Histogram, color=color)
    hist = histogram(channel)
    plt.plot(hist, color=color)
    plt.xlim([0, 256])

plt.show()