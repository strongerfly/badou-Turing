import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('lenna.png')
print(img.shape)

data = img.reshape((-1,3))
data = np.float32(data)

criteria = (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER,10,1.0)
flags = cv2.KMEANS_RANDOM_CENTERS
compactness,labels2,centers2 = cv2.kmeans(data,2,None,criteria,10,flags)
compactness,labels4,centers4 = cv2.kmeans(data,4,None,criteria,10,flags)
compactness,labels8,centers8 = cv2.kmeans(data,8,None,criteria,10,flags)
compactness,labels16,centers16 = cv2.kmeans(data,16,None,criteria,10,flags)
compactness,labels32,centers32 = cv2.kmeans(data,32,None,criteria,10,flags)

centers2 = np.uint8(centers2)
res = centers2[labels2.flatten()]
dst2 = res.reshape((img.shape))

centers4 = np.uint8(centers4)
res = centers4[labels4.flatten()]
dst4 = res.reshape((img.shape))

centers8 = np.uint8(centers8)
res = centers8[labels8.flatten()]
dst8 = res.reshape((img.shape))

centers16 = np.uint8(centers16)
res = centers16[labels16.flatten()]
dst16 = res.reshape((img.shape))

centers32 = np.uint8(centers32)
res = centers32[labels32.flatten()]
dst32 = res.reshape((img.shape))

centers2 = np.uint8(centers2)
res = centers2[labels2.flatten()]
dst = res.reshape((img.shape))

img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
dst2 = cv2.cvtColor(dst2,cv2.COLOR_BGR2RGB)
dst4 = cv2.cvtColor(dst4,cv2.COLOR_BGR2RGB)
dst8 = cv2.cvtColor(dst8,cv2.COLOR_BGR2RGB)
dst16 = cv2.cvtColor(dst16,cv2.COLOR_BGR2RGB)
dst32 = cv2.cvtColor(dst32,cv2.COLOR_BGR2RGB)

plt.rcParams['font.sans-serif'] = ['SimHei']

titles = [u'原始',u'聚2类',u'聚4类',u'聚8类',u'聚16类',u'聚32类']
images = [img,dst2,dst4,dst8,dst16,dst32]
for i in range(6):
    plt.subplot(2,3,i+1),plt.imshow(images[i],'gray'),plt.title(titles[i])
    plt.xticks([]),plt.yticks([])
plt.show()