#include <opencv2/opencv.hpp>

int main()
{
    cv::Mat img0 = cv::imread("C:\\ProgramData\\NVIDIA Corporation\\CUDA Samples\\v7.5\\Debug\\lena.png");
    if (img0.empty()) {
        printf("error imread.\n");
        return -1;
    }

    cv::Mat img(img0.cols, img0.rows, CV_8UC1);
    cv::cvtColor(img0, img, CV_RGB2GRAY);
    cv::imshow("o", img);
    img = cv::Mat_<double>(img);
    cv::Mat dctMat;
    cv::dct(img, dctMat);

    int nChannel = dctMat.channels();

    cv::normalize(dctMat, dctMat, 0, 255, cv::NORM_MINMAX);
    //for (int y = 0; y < 200; y+=5) {
    //    for (int x = 0; x < 200; x+=5)
    //        printf("%.0f ", dctMat.at<double>(x, y));
    //    printf("\n");
    //}
    //cv::Mat img1 = cv::Mat_<char>(dctMat);


    cv::Mat img1;
    cv::idct(dctMat, img1);
    //cv::dct(dctMat, img1, cv::DCT_INVERSE);
    //cv::normalize(img1, img1, 0, 255, cv::NORM_MINMAX);
    //img1 = cv::Mat_<char>(img1);
    cv::imshow("idct", img1);
    cv::waitKey();
    
    getchar();
}