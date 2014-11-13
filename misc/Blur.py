#!/usr/bin/env python

import sys
import numpy as np
import cv2
import cv2.cv as cv


if __name__ == '__main__':
	kernel = np.ones((3,3), np.float64)
	print kernel

	mat = np.array([[1,2,3],[4,5,6],[7,8,9]], np.float64)
	print mat
	print "blur result"
	print cv2.blur(mat, (3,3), borderType=cv2.BORDER_CONSTANT)
	
	kernel = kernel/9
	print "filter2D result"
	print cv2.filter2D(mat, -1, kernel, borderType=cv2.BORDER_CONSTANT)
