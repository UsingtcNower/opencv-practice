#include <opencv2/opencv.hpp>
#include <iostream>
#include <cmath>

using namespace cv;
using namespace std;

void main()
{
	VideoCapture capture("test.flv");
	if(!capture.isOpened()) {
		cout<< "Failed to open video!" << endl;
		return ;
	}

	Mat frame;
	while(capture.read(frame)) {
		imshow("test",frame);
		waitKey(33);
	}
}