#!/usr/bin/env python

import sys
import numpy as np
import cv2
import cv2.cv as cv

help_message = '''
USAGE: FaceDetector.py [--cascade <cascade_fn>] [--nested-cascade <cascade_fn>] [<image_path>]
'''

def detect(img, cascade):
    rects = cascade.detectMultiScale(img, scaleFactor=1.3, minNeighbors=4, minSize=(30, 30), flags = cv.CV_HAAR_SCALE_IMAGE)
    if len(rects) == 0:
        return []
    rects[:,2:] += rects[:,:2]
    return rects

def draw_rects(img, rects, color):
    for x1, y1, x2, y2 in rects:
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

if __name__ == '__main__':
    import sys, getopt
    print help_message

    args, image_src = getopt.getopt(sys.argv[1:], '', ['cascade=', 'nested-cascade='])
    try: image_src = image_src[0]
    except: image_src = 0
    args = dict(args)
    cascade_fn = args.get('--cascade', "../../data/haarcascades/haarcascade_frontalface_alt.xml")
    nested_fn  = args.get('--nested-cascade', "../../data/haarcascades/haarcascade_eye.xml")

    cascade = cv2.CascadeClassifier(cascade_fn)
    nested = cv2.CascadeClassifier(nested_fn)

    image = cv2.imread(image_src)
    if image is None:
	print 'imread error'
	sys.exit(-1)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    rects = detect(gray, cascade)
    draw_rects(image, rects, (0, 255, 0))
    cv2.imshow("FaceDetector", image)
    cv2.waitKey()
    cv2.destroyAllWindows()
