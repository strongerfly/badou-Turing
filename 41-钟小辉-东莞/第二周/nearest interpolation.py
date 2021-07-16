import cv2
import matplotlib.pyplot as plt
import numpy as np

def cv_show(name,img):
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



#放大或缩小，假设放大或缩小k倍
def nearest_interpolation(img,k1,k2):

    height,width,channels = img.shape
    new_image = np.zeros((int(k1 * height),int(k2 * width),channels),np.uint8)

    for i in range(int(k1 * height)):
        for j in range(int(k2 * width)):
            x = int(i/k1)
            y = int(j/k2)
            new_image[i,j] = img[x,y]

    return new_image

img  = cv2.imread("lenna.png")
# cv_show("image",img)
print(img.shape)
new_img = nearest_interpolation(img,1.1,1.1)
# cv_show("image",new_img)
print(new_img.shape)
cv2.imshow("image2",img)
cv2.imshow("image",new_img)
cv2.waitKey(0)

