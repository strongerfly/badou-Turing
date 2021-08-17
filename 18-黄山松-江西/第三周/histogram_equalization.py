import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('lenna.png', 1) # cv2默认BGR
(b, g, r) = cv2.split(img)
bH = cv2.equalizeHist(b)
gH = cv2.equalizeHist(g)
rH = cv2.equalizeHist(r)
result = cv2.merge((rH, gH, bH))
result_1 = cv2.merge((bH, gH, rH))
img_1 = cv2.merge((r, g, b))
cv2.imshow('Histogram Equalization', result_1)    # cv2默认BGR

plt.subplot(221)
plt.imshow(img_1)    # 默认RGB

plt.subplot(222)
plt.imshow(result)

colors = ('b', 'g', 'r')
plt.subplot(223)
for (chan, color) in zip((b, g, r), colors):
    hist = cv2.calcHist([chan], [0], None, [256], [0, 256])
    plt.plot(hist, color=color)
plt.xlim([0, 256])
plt.xlabel('Bins')
plt.ylabel('Number of Pixels')
plt.title('src')

plt.subplot(224)
for (chan, color) in zip((bH, gH, rH), colors):
    hist = cv2.calcHist([chan], [0], None, [256], [0, 256])
    plt.plot(hist, color=color)
plt.xlim([0, 256])
plt.xlabel('Bins')
plt.ylabel('Number of Pixels')
plt.title('dst')


plt.show()

k = cv2.waitKey(0)
if k == 27:
    cv2.destroyAllWindows()
elif k == ord('s'):
    cv2.imwrite('zhangzhangla.png', result_1)
cv2.waitKey(0)

