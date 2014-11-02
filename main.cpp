//#include "mainwindow.h"
//#include <QApplication>
#include "facedetector.h"

const int Pictures = 6;
char *picPath[50] = {"images/face01.jpg",
                     "images/face02.jpg",
                     "images/face03.jpg",
                     "images/face04.jpg",
                     "images/face05.jpg",
                     "images/face06.jpg"};

int main(int argc, char *argv[])
{
    // ## TODO: transform cvMat to QImage, remove highgui because it depends on x11
//    QApplication a(argc, argv);
//    MainWindow w;
//    w.show();

//    return a.exec();
    FaceDetector detector;
    for(int i=0;i<Pictures;++i) {
        detector.exec(picPath[i]);
    }
}
