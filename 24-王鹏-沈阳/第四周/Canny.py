import cv2

if __name__ == "__main__":
    img = cv2.imread("lenna.png",cv2.IMREAD_GRAYSCALE)
    result = cv2.Canny(img,200,300)
    cv2.imshow("canny",result)
    cv2.waitKey(0)