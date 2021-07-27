import cv2
import numpy as np


'''
第四周作业：
1）canny实现
2）相机模型推导
3）透视变换实现
'''

#点击鼠标，获得图像坐标
def on_EVENT_LBUTTONDOWN(event, x, y, flags, param):
    '''
    :param event: 鼠标事件
    :param x: 点击点的横坐标
    :param y: #点击点的纵坐标
    :param flags:
    :param param:
    :return:
    '''
    a = []
    b = []
    if event == cv2.EVENT_LBUTTONDOWN:
        xy = "%d,%d" % (x, y)  # 获得鼠标点击的点的坐标
        a.append(x)  # 保存点击点的横坐标
        b.append(y)  # 保存点击点的纵坐标
        # 以鼠标点击点为圆心进行圆的绘制
        cv2.circle(img,  # 绘制图像的图布
                   (x, y),  # 圆心
                   1,  # 半径
                   (0, 0, 255),  # 绘制颜色
                   thickness=-1  # 轮廓的厚度，-1表示填充
                   )
        # 在点击点附近进行文本的标注
        cv2.putText(img,
                    (xy),  # 需要标注上去的文本内容
                    (x, y),  # 文本框的位置，一般指的是文本框的左下角点的坐标
                    cv2.FONT_HERSHEY_PLAIN,  # fontface
                    1.0,  # fontScale
                    (255, 0, 0),  # 文本颜色
                    thickness=2)  # 文本粗细

        cv2.imshow("image", img)
        print(x, y)

if __name__ == '__main__':
    # canny实现
    img = cv2.imread('lenna.png')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imshow("canny", cv2.Canny(gray, 200, 300))
    cv2.namedWindow("image")
    cv2.setMouseCallback("image", on_EVENT_LBUTTONDOWN)  # 获得在图像上的鼠标响应函数
    cv2.imshow("image", img)
    result3 = img.copy()
    #透视变换
    src = np.float32([[19, 57], [216, 3], [121, 403], [435, 155]])
    dst = np.float32([[0, 0], [337, 0], [0, 488], [337, 488]])
    print(img.shape)
    m = cv2.getPerspectiveTransform(src, dst)
    print("warpMatrix:")
    print(m)
    result = cv2.warpPerspective(result3, m, (337, 488))
    cv2.imshow("src", img)
    cv2.imshow("result", result)
    cv2.waitKey(0)
