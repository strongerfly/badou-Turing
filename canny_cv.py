import  cv2

img = cv2.imread(r"C:\Users\ZhongXH2\Desktop\zuoye\lenna.png",1)
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
cv2.imshow("image",cv2.Canny(gray,100,200))
cv2.waitKey(0)
cv2.destroyAllWindows()
