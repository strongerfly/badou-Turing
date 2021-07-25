import cv2
import numpy as np
if __name__ == "__main__":
    img = cv2.imread("tower.jpeg")
    # cv2.imshow("input", img)
    # cv2.waitKey(0)
    w, h, _ = img.shape
    src = np.float32([[180,94],[377,39],[242,312],[436,265]])
    dis = np.float32([[0,0],[w,0],[0,h],[w,h]])
    matrix = cv2.getPerspectiveTransform(src,dis)
    result = cv2.warpPerspective(img,matrix,(w,h))
    cv2.imshow("input",img)
    cv2.imshow("output",result)
    cv2.waitKey(0)