#!/usr/bin/end python

import numpy as np
import cv2
import cv

if __name__ == '__main__':
	m = np.asmatrix([[1,2],[1,2]], np.float32)
	print np.cov(m)
	
	covmat,mean = cv2.calcCovarMatrix(m, cv2.cv.CV_COVAR_NORMAL | cv2.cv.CV_COVAR_COLS, ctype=cv2.CV_32F)
	print covmat
	print mean
