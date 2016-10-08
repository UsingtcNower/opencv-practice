#include <helper_cuda.h>
#include <opencv2/opencv.hpp>

#define SHARED_MEMORY_BANKS 16

cv::Mat histImg;

__global__ void hist64binKernel()
{}

__global__ void hist64binMerge()
{}

int main(int argc, char** argv)
{
    int histSize = 64;
    cv::Mat img0 = cv::imread("lena.png");
    if (img0.empty()) {
        printf("error imread.\n");
        return -1;
    }
    assert(img0.channels() == 3);
    cv::Mat img(img0.cols, img0.rows, CV_8UC1);
    cv::cvtColor(img0, img, CV_RGB2GRAY);

    // init
    int devId = findCudaDevice(argc, (const char**)argv);
    int* drData = NULL;
    checkCudaErrors(cudaMalloc((void**)&drData, img.cols*img.rows*sizeof(int)));
    int* hist = NULL;
    checkCudaErrors(cudaMalloc((void**)&hist, histSize*sizeof(int)));

    // calculate


    // draw
    int hist_w = 512;
    int hist_h = 400;
    histImg = cv::Mat(hist_w, hist_h, CV_8UC3, cv::Scalar(0,0,0));
    int bin_w = cvRound((double)hist_w/histSize);

    for (int i = 0; i < histSize; ++i)
    {
        hist[i] = hist[i] * hist_h / 256;
    }
    for (int i = 1; i < histSize; ++i) {
        cv::line(histImg, Point((i-1)*bin_w, hist_h-hist[i-1]),
                          Point(i*bin_w, hist_h-hist[i]),
                          cv::Scalar(255, 0, 0), 2, 8, 0);
    }
    cv::imwrite(histImg, "histImg.png");
    return 0;
}