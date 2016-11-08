#include <opencv2/opencv.hpp>

int main()
{
    cv::Mat img0 = cv::imread("C:\\ProgramData\\NVIDIA Corporation\\CUDA Samples\\v7.5\\Debug\\lena.png");
    if (img0.empty()) {
        printf("error imread.\n");
        return -1;
    }

    cv::Mat img(img0.cols, img0.rows, CV_8UC1);
    cv::Mat dctMat;
    cv::dct(img, dctMat);
}