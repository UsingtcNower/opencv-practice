#include "imageprocess.h"
#include <QImage>

IplImage *imageprocess::QImage2IplImage(QImage *qimage) {
    IplImage *iplImage =
            cvCreateImageHeader(cvSize(qimage->width(), qimage->height()), 8, 3);
    iplImage->imageData = (char *)qimage->bits();
    return iplImage;
}
