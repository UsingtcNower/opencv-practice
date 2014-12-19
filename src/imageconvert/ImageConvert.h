#ifndef CVUTIL_H
#define CVUTIL_H
#include "opencv/cxcore.h"
#include <QImage>

QImage *IplImage2QImage_s(IplImage *src);
IplImage *QImage2IplImage_s(QImage *src);

#endif // CVUTIL_H
