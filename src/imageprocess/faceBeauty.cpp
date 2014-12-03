#include <vector>
#include "imageprocess.h"

void imageprocess::faceBeauty(Mat img) {
    cvtColor(img, img, COLOR_BGR2YCrCb);
    vector<Mat> channels;
    split(img, channels);
    equalizeHist(channels[0], channels[0]);
    merge(channels, img);
    cvtColor(img, img, COLOR_YCrCb2BGR);
}
