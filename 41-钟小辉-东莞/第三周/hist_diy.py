import  numpy as np
import matplotlib.pyplot as plt
import  cv2

def cv_show(name,image):
    cv2.imshow(name, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

#这里只接受单通道：DIY
def histeq(img,nbr_bins = 256):
    #1.像素的存储
    h,w = img.shape
    outimg =np.zeros((h,w),img.dtype) #输出图像
    _nums = np.zeros(256) #存储0-255灰度值的数目
    Pi = np.zeros(256)  # 存储0-255灰度值在h*W中的比例
    sumPi = np.zeros(256)  # 累加比例和
    q= np.zeros(256)  # 最后的各像素均衡化后的转换值

    #计算各像素个数
    for i in range(h):
        for j in range(w):
            index = img[i,j]
            _nums[index] =_nums[index]+1
    # print(_nums)
    # print(_nums.shape)
    # print(sum(_nums))

    #计算各像素个数在h，w中的比例系数、累加和
    for ii in range(256):
        Pi[ii] = (_nums[ii]*1.0)/(h*w)
        sumPi[ii] = sum(Pi)
        q[ii] =max(0,round(sumPi[ii]*256-1))

    # print(sum(Pi))
    # print(sumPi)
    # print(q)
    # 最后的各像素均衡化后的转换值
    for ii in range(256):
        outimg[img==ii] = q[ii]
    # print(outimg)
    return outimg


# 注意histe函数，这里只支持输入单通道的图片进行均衡化
img = cv2.imread(r"C:\Users\ZhongXH2\Desktop\zuoye\lenna.png")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
b,g,r = cv2.split(img)
# cv_show("image",gray)
new = histeq(gray)
cv_show("image",new)

#直方图
plt.figure()
plt.hist(gray.ravel(),bins=256)
plt.show()



