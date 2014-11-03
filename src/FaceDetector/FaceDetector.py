import cv2
import cv

image = cv2.imread("../../images/face01.jpg")
image_gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
kernal = [[0,-1,0],[-1,5,-1],[0,-1,0]]
image_gray = image_gray*kernal
cv2.imshow("test", image_gray)
cv2.waitKey()
