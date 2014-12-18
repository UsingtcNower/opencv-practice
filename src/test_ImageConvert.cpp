#include "test.h"
#include "ImageConvert.h"
#include "opencv2/highgui/highgui_c.h"
#include <QSharedPointer>
#include <QLabel>
#include <QFileInfo>
#include <QMainWindow>
#include <QByteArray>
#include <QDebug>

void test_QImage2IplImage() {
    QSharedPointer<QImage> src(new QImage("images/beauty01.jpg"));
    Q_ASSERT(!src->isNull());
    showQImage(src.data());
    IplImage *dst = QImage2IplImage_s(src.data());
    assert(dst);
    showIplImage(dst);
}

void test_IplImage2QImage() {
    IplImage *src = cvLoadImage("images/beauty01.jpg");
    assert(src);
    showIplImage(src);
    QImage *dst = IplImage2QImage_s(src);
    Q_ASSERT(!dst->isNull());
    showQImage(dst);
}

void showIplImage(IplImage *img) {
    cvNamedWindow("test");
    cvShowImage("test", img);
    //cvWaitKey();
}

void showQImage(QImage *img) {
    QLabel *label = new QLabel;
    label->setPixmap(QPixmap::fromImage(*img));
    QMainWindow *mainWindow = new QMainWindow;
    mainWindow->setCentralWidget(label);
    mainWindow->show();
}
