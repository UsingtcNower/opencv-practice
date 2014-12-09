#ifndef IMAGEPROCESSOR_H
#define IMAGEPROCESSOR_H
#include "opencv2/imgproc/imgproc.hpp"

class QImage;

using namespace cv;

namespace imageprocess
{
void faceBeauty(Mat img);
IplImage *QImage2IplImage(QImage *qimage);
}

#endif // IMAGEPROCESSOR_H
