#include "ImageConvert.h"
#include "opencv/cvaux.h"

/*
 * Note: use shared buffer for src and dst.
 */
IplImage *QImage2IplImage_s(QImage *src) {
    IplImage *dst = 0;
    if(src) {
        dst = cvCreateImageHeader(cvSize(src->width(), src->height()), IPL_DEPTH_8U, 4);
        dst->imageData = (char *)src->bits();
    }
    return dst;
}

QImage *IplImage2QImage_s(IplImage *src) {
    QImage *dst = 0;
    if(src) {
        // Note, in opencv, the channels is BGR, so need change color firstly
        cvCvtColor(src, src, CV_BGR2RGB);
        dst = new QImage((uchar *)src->imageData, src->width, src->height, src->widthStep, QImage::Format_RGB888);
    }
    return dst;
}
