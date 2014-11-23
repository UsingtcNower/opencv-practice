import numpy as np
import scipy.ndimage as ndimage
import cv2
import Image

if __name__ == "__main__":
    #kernel = np.array([[0.25,0,-0.25],[0,0,0],[-0.25,0,0.25]])
    kernel = np.array([[0,0.125,0],[0.125,-0.5,0.125],[0,0.125,0]])
    print kernel

    #image = cv2.imread("../images/face01.jpg")
    im = Image.open("../images/face01.jpg")
    image = np.array(im,dtype=float)
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image_gray = cv2.equalizeHist(image_gray)
    print image_gray
    image_gray_cv = image_gray

    image_gray = np.array(image_gray)
    image_gray = ndimage.filters.convolve(image_gray, kernel, mode="constant",cval=0)
    print image_gray
    #image_gray_cv = cv2.filter2D(image_gray_cv, -1, kernel, borderType=cv2.BORDER_CONSTANT)
    #print image_gray_cv
    
    cv2.imshow("test", image_gray)
    cv2.waitKey()
