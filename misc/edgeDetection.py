import numpy as np
import scipy.ndimage as ndimage
import cv2

if __name__ == "__main__":
    kernel = np.mat([[0.25,0,-0.25],[0,0,0],[-0.25,0,0.25]])
    #print kernel

    image = cv2.imread("../images/face01.jpg")
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image_gray = cv2.equalizeHist(image_gray)
    #print image_gray
    
    image_gray = ndimage.filters.convolve(image_gray, kernel, mode="constant", cval=0.0)
    #image_gray = cv2.filter2D(image_gray, cv2.CV_8U, kernel)
    print image_gray
    
    cv2.imshow("test", image_gray)
    cv2.waitKey()
