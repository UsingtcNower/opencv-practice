#-------------------------------------------------
#
# Project created by QtCreator 2014-10-20T05:50:42
#
#-------------------------------------------------

QT       += core gui

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

TARGET = FaceDetector
TEMPLATE = app


SOURCES += main.cpp\
        mainwindow.cpp

HEADERS  += mainwindow.h

FORMS    += mainwindow.ui

OTHER_FILES += \
    TrainingSet/haarcascade_eye_tree_eyeglasses.xml \
    TrainingSet/haarcascade_frontalface_alt.xml
