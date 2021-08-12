import  cv2
import numpy as np

img = cv2.imread("lenna.png")
gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
# print(cv2.__version__)

#创建sift模型
sift = cv2.xfeatures2d.SIFT_create()
#得到关键点
kp = sift.detect(gray,None)
# print(kp)
#将关键点描绘出来
# img = cv2.drawKeypoints(gray,kp,img)
img_kp = img.copy()
# img_kp = cv2.drawKeypoints(gray,kp,img_kp)
img_kp = cv2.drawKeypoints(img_kp,kp,img_kp)

#描绘关键点
cv2.imshow('drawKeypoints', img_kp)
cv2.waitKey(0)
cv2.destroyAllWindows()


#计算特征向量
keypoints, descriptor = sift.compute(gray,kp)
print(np.array(keypoints).shape)
print(descriptor.shape)

#描绘特征向量
img_des = img.copy()
# img_des = cv2.drawKeypoints(img_des,keypoints,img_des,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
#                         color=(51, 163, 236))
img_des = cv2.drawKeypoints(img_des,keypoints,img_des,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv2.imshow('sift_keypoints', img_des)
cv2.waitKey(0)
cv2.destroyAllWindows()

