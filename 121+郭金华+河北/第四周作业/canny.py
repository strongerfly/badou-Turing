
import cv2


path=r'C:\Users\xiaoguo\Desktop\lenna.png'
img=cv2.imread(path,0)
dst=cv2.Canny(img,50,200)
cv2.imshow('lenna',dst)
cv2.waitKey()
cv2.destroyAllWindows()