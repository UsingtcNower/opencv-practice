#include <vector>
#include "imageprocessor.h"

using namespace std;

ImageProcessor::ImageProcessor()
{
}

ImageProcessor::ImageProcessor(const ImageProcessor &) {

}

void ImageProcessor::operator =(const ImageProcessor&) {

}

void ImageProcessor::faceBeauty(Mat img) {
    cvtColor(img, img, COLOR_BGR2YCrCb);
    vector<Mat> channels;
    split(img, channels);
    equalizeHist(channels[0], channels[0]);
    merge(channels, img);
    cvtColor(img, img, COLOR_YCrCb2BGR);
}
