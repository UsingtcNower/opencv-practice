#1/usr/bin env python

import sys
import numpy as np
import cv2

if __name__ == '__main__':
	img = cv2.imread('../images/line.jpg')
	#print img
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	gray = cv2.blur(gray, (3,3))
	cv2.imshow('raw', gray)
	edge = cv2.Canny(gray, 50, 150)
	cv2.imshow('canny', edge)
	lines = cv2.HoughLines(edge, 1, np.pi/180, 50)
	for rho, theta in lines[0]:
		cosv = np.cos(theta)
		sinv = np.sin(theta)
		x0 = rho*cosv
		y0 = rho*sinv
		x1 = int(x0 - 1000*sinv)
		y1 = int(y0 - 1000*cosv)
		x2 = int(x0 + 1000*sinv)
		y2 = int(y0 + 1000*cosv)
		
		cv2.line(img, (x1,y1), (x2,y2), (0,0,255), 2)
	
	cv2.imshow('test', img)
	cv2.waitKey()
