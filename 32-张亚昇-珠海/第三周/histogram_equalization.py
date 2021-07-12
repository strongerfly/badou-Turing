import numpy as np
import cv2
import matplotlib.pyplot as plt
img = cv2.imread('./lenna.png')
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
print(img_gray.shape)

#hist = cv2.calcHist([img_gray],[0],None,[256],[0,256])
def hist_equ(img):
    hist = cv2.calcHist([img],[0],None,[256],[0,256])
    print(hist.shape)
    p = np.cumsum((hist)/(img.shape[0] * img.shape[1])) * 256 -1
    p = p.astype('uint8')
    print(p.shape)
    img_equ = np.zeros(img.shape, dtype='uint8')
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            img_equ[i,j] = p[img[i,j]]
    return img_equ
img_eq=hist_equ(img_gray)
fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.set_title("Raw Image")
ax1.imshow(img_gray, 'gray')
ax1.set_xticks([]), ax1.set_yticks([])

ax2.set_title("Histogram Equalized Image")
ax2.imshow(img_eq, 'gray')
ax2.set_xticks([]), ax2.set_yticks([])

fig.tight_layout()
plt.show()

#print(hist.shape)
def hist(img):
    plt.hist(img.ravel(), 256, [0, 256])
    plt.show()

hist(img_gray)
hist(img_eq)
#for i in range(img_gray.shape[0]):
    #for j in range(img_gray.shape[1]):
