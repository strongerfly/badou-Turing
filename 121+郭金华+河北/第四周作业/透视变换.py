import cv2
import numpy as np

img=cv2.imread('fisheyes.jpg')
result=img.copy()

source=np.float32([[235,482], [624, 569], [785,639], [875,513]])
dest=np.float32([[226,521], [633,572], [824,668], [875,569]])

warpmatrix=cv2.getPerspectiveTransform(source,dest)
print(warpmatrix)
result=cv2.warpPerspective(result, warpmatrix, (1024,2048))
cv2.imshow('source', img)
cv2.waitKey()
cv2.imshow('des', result)
cv2.waitKey()
'''*************************************在显示的图像上面显示像素坐标***************************************************************'''
# def on_EVENT_LBUTTONDOWN(event, x, y, flags, param):
#     if event == cv2.EVENT_LBUTTONDOWN:
#         xy = "%d,%d" % (x, y)
#         cv2.putText(img, xy, (x, y), cv2.FONT_HERSHEY_PLAIN,
#                     1.0, (0, 0, 0), thickness=1)
#         cv2.imshow("image", img)
#
#
# cv2.namedWindow("image")
# cv2.setMouseCallback("image", on_EVENT_LBUTTONDOWN)
# cv2.imshow("image", img)
#
# while (True):
#     try:
#         cv2.waitKey(100)
#     except Exception:
#         cv2.destroyAllWindows()
#         break
#
# cv2.waitKey(0)
# cv2.destroyAllWindows()
'''******************************************************************************************************************'''
