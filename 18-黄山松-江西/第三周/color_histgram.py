import cv2
from matplotlib import pyplot as plt

img = cv2.imread("lenna.png")
cv2.imshow("Original", img)
chans = cv2.split(img)
colors = ('b', 'g', 'r')
for (chan, color) in zip(chans, colors):
    hist = cv2.calcHist([chan], [0], None, [256], [0, 256])
    plt.plot(hist, color=color)
plt.title('Flattened color histogram')
plt.xlabel('Bins')
plt.xlim([0, 270])
plt.ylabel('Number of Pixels')
plt.ylim([0, 5000])
plt.show()
