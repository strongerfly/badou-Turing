import cv2
import numpy as np

img = cv2.imread("photo1.jpg")
cv2.imshow("image",img)
cv2.waitKey(0)
cv2.destroyAllWindows()

src = np.float32([[207, 151], [517, 285], [17, 601], [343, 731]])
dst = np.float32([[0,0],[500,0],[0,400],[500,400]])
_mat = cv2.getPerspectiveTransform(src,dst)

print(f"warpMatrix:\n{_mat}")

res = img.copy()
result = cv2.warpPerspective(res,_mat,(800,800))
cv2.imshow("image",result)
cv2.waitKey(0)
cv2.destroyAllWindows()