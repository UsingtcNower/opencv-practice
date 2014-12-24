#!/usr/bin/env python
import numpy as np
import cv2


if __name__ == '__main__':
    print __doc__

    import sys
    from itertools import cycle
    from common import draw_str

    try: fn = sys.argv[1]
    except: fn = '../images/beauty01.jpg'
    img = cv2.imread(fn)
    cv2.imshow("original", img)

    def update(dummy=None):
        contrast = cv2.getTrackbarPos('contrast', 'Basic Linear Transform')
        brightness = cv2.getTrackbarPos('brightness', 'Basic Linear Transform')
	res = np.dot(np.dot(img,contrast), 0.01)+brightness
	cv2.imshow('Basic Linear Transform', res)
	cv2.waitKey()

    cv2.namedWindow('Basic Linear Transform')
    cv2.createTrackbar('contrast', 'Basic Linear Transform', 100, 300, update)
    cv2.createTrackbar('brightness', 'Basic Linear Transform', 0, 100, update)
    cv2.setTrackbarPos('contrast', 'Basic Linear Transform', 100)
    cv2.setTrackbarPos('brightness', 'Basic Linear Transform', 0)
    update()
    cv2.destroyAllWindows()
