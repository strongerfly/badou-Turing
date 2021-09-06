import matplotlib.pyplot as plt
import numpy as np
import cv2

def cv_show(name,image):
    cv2.imshow(name,image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

#使用灰度图
img1 = cv2.imread("iphone1.png",0)
img2 = cv2.imread("iphone2.png",0)
# cv_show("image",img1)
# cv_show("image",img2)

"""
# cv2.drawMatches(imageA, kpsA, imageB, kpsB, matches[:10], None, flags=2)  # 对两个图像关键点进行连线操作
# 
# 参数说明：imageA和imageB表示图片，kpsA和kpsB表示关键点， matches表示进过cv2.BFMatcher获得的匹配的索引值，也有距离， flags表示有几个图像
# 
# 书籍的SIFT特征点连接：
# 
#    第一步:使用sift.detectAndComputer找出关键点和sift特征向量
# 
#    第二步：构建BFMatcher()蛮力匹配器，bf.match匹配sift特征向量，使用的是欧式距离
# 
#    第三步：根据匹配结果matches.distance对matches按照距离进行排序
# 
#    第四步：进行画图操作，使用cv2.drawMatches进行画图操作

"""

# 第一步：构造sift，求解出特征点和sift特征向量
sift = cv2.xfeatures2d_SIFT.create()

#用SIFT找到关键点和特征向量
kp1,des1 = sift.detectAndCompute(img1,None)
kp2,des2 = sift.detectAndCompute(img2,None)

# crossCheck表示两个特征点要互相匹，例如A中的第i个特征点与B中的第j个特征点最近的，并且B中的第j个特征点到A中的第i个特征点也是
#NORM_L2: 归一化数组的(欧几里德距离)，如果其他特征计算方法需要考虑不同的匹配计算方式

# #方法1： 第二步：构造BFMatcher()蛮力匹配，匹配sift特征向量距离最近对应组分
# # 获得匹配的结果
# bf = cv2.BFMatcher(cv2.NORM_L2,crossCheck =True)
# matches =  bf.match(des1,des2)
# #第三步：对匹配的结果按照距离进行排序操作
# mathces=sorted(matches,key=lambda x:x.distance) #据距离来排序
# 第四步：使用cv2.drawMacthes进行画图操作
# img3 = cv2.drawMatches(img1,kp1,img2,kp2,matches[:10],None,flags=2)

#方法2：第二步：构造BFMatcher(),k对最佳匹配:k=2
bf = cv2.BFMatcher(cv2.NORM_L2)
matches = bf.knnMatch(des1, des2, k=2)
goodMatch = []
for m, n in matches:
    if m.distance < 0.50 * n.distance:
        goodMatch.append(m)

# 第四步：使用cv2.drawMacthes进行画图操作
img3 = cv2.drawMatches(img1,kp1,img2,kp2,goodMatch[:20],None,flags=2)
cv_show("image",img3)