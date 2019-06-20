#include <stdio.h>
#include <stdlib.h>

//#include <core/core.hpp>
//#include <highgui/highgui.hpp>
//#include <opencv.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <vector>
using namespace cv;
using namespace std;


void icvprCcaByTwoPass(const cv::Mat& _binImg, cv::Mat& _lableImg);
Mat icvprCcaBySeedFill(const cv::Mat& _binImg);
int GlueDetectionAlgorithm(unsigned char *img, unsigned char *basemaskImg, unsigned char *mask, unsigned char *retImg, int type, int width, int height, double *ret, double * input);