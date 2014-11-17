#!/usr/bin/env python

'''
Morphology operations.

Usage:
  morphology.py [<image>]

Keys:
  1   - change operation
  2   - change structure element shape
  ESC - exit
'''

import numpy as np
import cv2


if __name__ == '__main__':
    print __doc__

    import sys
    from itertools import cycle
    from common import draw_str

    try: fn = sys.argv[1]
    except: fn = '../images/black01.jpg'
    img = cv2.imread(fn)
    
    cv2.imshow('original', img)
    def update(dummy=None):
        #beta  = cv2.getTrackbarPos('beta', 'skin beauty')
       # beta = 3
	#res = np.log10(img*(beta-1)+1)/np.log10(beta)
	res = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
	channels = cv2.split(res)
	channels[0] = cv2.equalizeHist(channels[0])
	res = cv2.merge(channels)
	res = cv2.cvtColor(res, cv2.COLOR_YCR_CB2BGR)
	cv2.imshow('skin beauty', res)

    cv2.namedWindow('skin beauty')
    cv2.createTrackbar('beta', 'skin beauty', 2, 10, update)
    update()
    cv2.waitKey()
    cv2.destroyAllWindows()
