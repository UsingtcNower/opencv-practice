#include "src/FaceDetector/facedetector.h"
#include <QDebug>

const char *face_cascade_file = "TrainingSet/haarcascade_frontalface_alt.xml";

FaceDetector::FaceDetector()
{
}

void FaceDetector::exec(const std::string &imagePath) {
    Mat image = imread(imagePath);
    if(image.data == NULL) {
        qDebug() << "failed to read " << QString::fromStdString(imagePath);
        return ;
    }
    std::vector<Rect> faces = detect(image);
    display(image, faces);
}

std::vector<Rect> FaceDetector::detect(Mat image) {
    if(image.data == NULL) {
        qDebug() << "null Mat";
        return std::vector<Rect>();
    }
    Mat image_gray;
    cvtColor(image, image_gray, CV_BGR2GRAY);
    display(image_gray, std::vector<Rect>());
    equalizeHist(image_gray, image_gray);
    display(image_gray, std::vector<Rect>());

    CascadeClassifier classifier(face_cascade_file);
    if(classifier.empty()) {
        qDebug() << "classifier is empty!";
        return std::vector<Rect>();
    }
    std::vector<Rect> faces;
    classifier.detectMultiScale(image_gray, faces, 1.1, 2, 0|CV_HAAR_SCALE_IMAGE, Size(50, 50));
    return faces;
}

void FaceDetector::display(Mat image, const std::vector<Rect> &faces) {
    if(image.data == NULL) {
        qDebug() << "null Mat";
        return ;
    }
    for(size_t i=0;i<faces.size();++i) {
        Point center(faces[i].x+faces[i].width*0.5, faces[i].y+faces[i].height*0.5);
        ellipse(image,center, Size(faces[i].width*0.5, faces[i].height*0.5), 0, 0, 360, Scalar(255,0,255), 4, 8, 0);

    }

    imshow("test", image);
    waitKey();
}
