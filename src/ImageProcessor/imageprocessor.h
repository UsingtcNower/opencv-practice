#ifndef IMAGEPROCESSOR_H
#define IMAGEPROCESSOR_H
#include "opencv2/imgproc/imgproc.hpp"

using namespace cv;

class ImageProcessor
{
public:
    static void faceBeauty(Mat img);

private:
    ImageProcessor();
    ImageProcessor(const ImageProcessor&);
    void operator= (const ImageProcessor&);
};

#endif // IMAGEPROCESSOR_H
