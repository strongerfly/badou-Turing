#coding=utf=8
import cv2
import numpy as np
'''
该文件实现canny算法
'''

def gaussian2D(shape, sigma=0.5):
    '''
    获取二维高斯的密度函数
    shape是核的大小（2k+1）
    sigma:方差
    '''
    m, n = [(ss - 1.) / 2. for ss in shape]#高斯
    y, x = np.ogrid[-m:m+1,-n:n+1] #生成网格信息，

    h = (1/(2.0*np.pi*sigma**2))*np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    h=h/(np.sum(h)+1e-8)
   # print("H",h,h.shape)
    return h

def DoBlur(img,radius,sigma=1.2):
    '''
    高斯平滑操作
    radius:高斯平滑核半径
    '''
    h,w=img.shape
    print(h,w)
    diameter = 2 *radius  + 1
    gaussian = gaussian2D((diameter, diameter), sigma=sigma)
    #与图像做高斯平滑gaoaa
    #先进行图像padding
    pad=diameter//2
    #新建图像
    out_img=np.zeros(img.shape)
    img_pad=np.pad(img,((pad,pad),(pad,pad)),'constant')
    #做高斯平滑
    for j in range(h):
        for i in range(w):
            out_img[j,i]=np.sum(np.multiply(gaussian,img_pad[j:j+diameter,i:i+diameter]))
    out_img=out_img.astype(np.uint8)
   # cv2.imshow("blur",out_img)
  #  cv2.imshow("src",img)
    return  out_img
   # cv2.waitKey()
def DosobelAndGetEdge(img):
    '''
    对图像做水平和垂直方向的滤波，得到水平和垂直方向梯度，就是计算每一个点的梯度幅值与方向
    :return: 
    '''
    Kx=np.array([[1,0,-1],[2,0,-2],[1,0,-1]],dtype=np.float32)
    Ky=np.array([[1,2,1],[0,0,0],[-1,-2,-1]])
    #进行pad
    k_size=Kx.shape[0]
    pad=Kx.shape[0]//2
    H,W =img.shape
    img_pad=np.pad(img,((pad,pad),(pad,pad)),'constant')
    out_x=np.zeros(img.shape)
    out_y=np.zeros(img.shape)
    #先进行sobel运算
    for j in range(H):
        for i in range(W):
            out_x[j,i]=np.sum(np.multiply(Kx,img_pad[j:j+k_size,i:i+k_size]))
            out_y[j, i] = np.sum(np.multiply(Ky, img_pad[j:j + k_size, i:i + k_size]))
    #求幅值
    Grad=np.hypot(out_x,out_y)
    #方向
    theta=np.arctan2(out_y,out_x+1e-8)#返回的值范围【-PI/2,PI/2】

    return Grad,theta
def Do_nms_suppression(grad,theta):
    '''
    极大值抑制，该点梯度值与周围8邻域相比梯度值最大，则保留，可分为4个方向，
    方向1：【0,22.5】-【157.5,180】；
    方向2【22.5，67.5】；
    方向3【67.5，112.5】；
    方向4 【112.5,157.5】
    这样划分，我们使用的线性插值是最近邻插值
    :param grad: 梯度幅值
    :param theta: 梯度方向
    :return:
    '''
    H,W=grad.shape
    output=np.zeros(grad.shape,dtype=np.uint8)
    theta=theta*180/np.pi
    theta[theta<0]+=180#对称性，我们只看一，二象限
    for j in range(1,H-1):#注意边缘
        for i in range(1,W-1):
            p1,p2=0,0
            #落在方向1
            if (0<=theta[j,i]<22.5) or(157.5<=theta[j,i]<=180):
                p1=grad[j,i-1] #这里其实是用了最近邻插值，取最靠近的点作为目标点
                p2=grad[j,i+1]
            #落在方向2
            elif(22.5<=theta[j,i]<67.5):
                p1=grad[j+1,i-1]
                p2=grad[j-1,i+1]
            elif(67.5<=theta[j,i]<112.5):
                p1=grad[j-1,i]
                p2=grad[j+1,i]
            elif(112.5<theta[j,i]<157.5):
                p1=grad[j-1,i-1]
                p2=grad[j+1,i-1]
            if grad[j,i]>p1 and grad[j,i]>p2:
                output[j,i]=grad[j,i]
    return output

def Do_doublethresh(img,lowthres=10,highthres=80):
    '''
    :param img: 极大值抑制的图像
    :param lowthres:
    :param highthres:
    :return:
    '''
    H,W=img.shape
    output=np.zeros(img.shape,dtype=np.uint8)
    largeindex=np.where(img>highthres)
    lowindex=np.where(img<lowthres)
    midindex=np.where((img<=highthres)&(img>=lowthres))
    output[largeindex]=255
    output[midindex]=0
   # output[lowindex]=img[lowindex]
    #对在双阈值范围内的点作进一步处理,在弱边缘点周围8邻域范围内有一个点是强边缘点，则将该点置位为强边缘点
  #  midindex=list(midindex)
   # print("midindex:",midindex)

    # for j in range(1,H-1):
    #     for i in range(1,W-1):
    #         if (output[j,i]==25):
    #             if (output[j - 1, i - 1] == 255) or (output[j - 1, i] == 255) or (output[j - 1, i + 1] == 255) or \
    #                     (output[j, i - 1] == 255) or (output[j, i + 1] == 255) or (output[j + 1, i - 1] == 255) or \
    #                     (output[j + 1, i] == 255) or (output[j + 1, i + 1] == 255):
    #                 output[j, i] = 255
    #             else:
    #                 output[j, i] = 0
    for j,i in zip(midindex[0],midindex[1]):
        if(output[j-1,i-1]==255) or(output[j-1,i]==255) or(output[j-1,i+1]==255) or \
          (output[j,i-1]==255) or (output[j,i+1]==255) or(output[j+1,i-1]==255) or \
          (output[j+1,i]==255) or(output[j+1,i+1]==255):
            output[j,i]=255
        else:
            output[j,i]=0
    return output


if __name__=='__main__':
    img=cv2.imread("lenna.png")
    #1.灰度化
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    #2.高斯平滑
    gray=DoBlur(gray,3)
    #梯度幅值和方向
    grad,theta=DosobelAndGetEdge(gray)
    #极大值抑制
    nms_=Do_nms_suppression(grad,theta)
    #双阈值连接
    result_canny=Do_doublethresh(nms_)
    cv2.imshow("canny",result_canny)
    cv2.waitKey()



    # img=cv2.GaussianBlur(gray,(7,7),1.0)# cv实现高斯滤波
    # cv2.imshow("cv",img)
    # cv2.waitKey()


