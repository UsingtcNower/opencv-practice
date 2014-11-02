#ifndef FACEDETECTOR_H
#define FACEDETECTOR_H
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

using namespace cv;

class FaceDetector
{
public:
    FaceDetector();
    void exec(const std::string& imagePath);

private:
    std::vector<Rect> detect(Mat image);
    void display(Mat image, const std::vector<Rect>& faces);
};

#endif // FACEDETECTOR_H
