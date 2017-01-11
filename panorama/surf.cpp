#include<opencv2/opencv.hpp>
#include<opencv2/xfeatures2d.hpp>
#include<vector>

using namespace cv;
using namespace cv::xfeatures2d;
using namespace std;

int main()
{
    Mat img1 = imread("../data/shanghai-00.png");
    Mat img2 = imread("../data/shanghai-01.png");
    imshow("ԭͼ1", img1);
    imshow("ԭͼ2", img2);

    //���������
    vector<KeyPoint> keys1;
    vector<KeyPoint> keys2;
    Ptr<SURF> surf = SURF::create();
    surf->detect(img1, keys1);
    surf->detect(img2, keys2);

    //����������
    Mat featureMat1, featureMat2;
    surf->compute(img1, keys1, featureMat1);
    surf->compute(img2, keys2, featureMat2);

    //��ʼƥ�䣬�ȼ���ƥ������
    FlannBasedMatcher flann;
    vector<DMatch> matches;
    flann.match(featureMat1, featureMat2, matches);

    Mat dstImg;
    drawMatches(img1, keys1, img2, keys2, matches, dstImg, 
        Scalar(theRNG().uniform(0,255), theRNG().uniform(0,255), theRNG().uniform(0,255)),
        Scalar(theRNG().uniform(0,255), theRNG().uniform(0,255), theRNG().uniform(0,255)),
        Mat(), 2);
    imshow("����ƥ��", dstImg);

    waitKey();
}