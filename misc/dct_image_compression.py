import cv2
import numpy as np

img = cv2.imread("D:\installer\opencv\sources\samples\data\lena.jpg")
o_img = img
yuv_img = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
y,u,v = cv2.split(yuv_img)
#dct
block_size = 8
y = np.float32(y)
u = np.float32(u)
v = np.float32(v)
dct_y = cv2.dct(y)
dct_u = cv2.dct(u)
dct_v = cv2.dct(v)
#idct
y = cv2.idct(dct_y)
u = cv2.idct(dct_u)
v = cv2.idct(dct_v)
y = np.uint8(y)
u = np.uint8(u)
v = np.uint8(v)

yuv_img = cv2.merge([y,u,v])
img = cv2.cvtColor(yuv_img, cv2.COLOR_YUV2BGR)
cv2.imshow("lena", img-o_img)
cv2.waitKey(0)
