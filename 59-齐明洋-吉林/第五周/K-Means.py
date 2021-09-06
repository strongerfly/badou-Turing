import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('lenna.png',0)
print(img.shape)

h,w = img.shape[:]
data = img.reshape((h*w,1))
data = np.float32(data)

criteria = (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER,10,1.0)
flags = cv2.KMEANS_RANDOM_CENTERS
compactness,labels,centers = cv2.kmeans(data,6,None,criteria,10,flags)
print(centers)

dst = labels.reshape((img.shape[0],img.shape[1]))

plt.rcParams['font.sans-serif'] = ['SimHei']
titles = [u'原始',u'聚类']
images = [img,dst]
for i in range(2):
    plt.subplot(1,2,i+1),plt.imshow(images[i],'gray')
    plt.title(titles[i])
    plt.xticks([]),plt.yticks([])
plt.show()