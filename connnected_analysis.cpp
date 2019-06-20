/*  Connected Component Analysis/Labeling By Two-Pass Algorithm*/
#include <iostream>
#include <string>
#include <list>
#include <vector>
#include <map>
#include <iostream>
#include <stack>


#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace cv;
using namespace std;


Mat icvprCcaBySeedFill(const cv::Mat& _binImg)
{
	// connected component analysis (4-component)
	// use seed filling algorithm
	// 1. begin with a foreground pixel and push its foreground neighbors into a stack;
	// 2. pop the top pixel on the stack and label it with the same label until the stack is empty
	//
	// foreground pixel: _binImg(x,y) = 1
	// background pixel: _binImg(x,y) = 0


	assert(_binImg.type() == CV_8UC1);

	/*label prepare*/
	int mrows = _binImg.rows;
	int mcols = _binImg.cols;
	Mat BinaryImage = Mat(Size(mcols, mrows), CV_8UC1);
	for (int i = 0; i < mcols; i++)
	{
		for (int j = 0; j < mrows; j++)
		{
			if (_binImg.at<uchar>(j, i) >= 255)
			{
				BinaryImage.at<uchar>(j, i) = 1;
			}
			else
			{
				BinaryImage.at<uchar>(j, i) = 0;
			}
		}
	}

	Mat _lableImg;
	BinaryImage.convertTo(_lableImg, CV_32SC1);
	int maxlabelcount = 0;
	int maxlabel = 0;
	int label = 1;  // start by 2

	int rows = _binImg.rows - 1;
	int cols = _binImg.cols - 1;
	for (int i = 1; i < rows - 1; i++)
	{
		int* data = _lableImg.ptr<int>(i);
		for (int j = 1; j < cols - 1; j++)
		{
			if (data[j] == 1)
			{
				std::stack<std::pair<int, int> > neighborPixels;
				neighborPixels.push(std::pair<int, int>(i, j));     // pixel position: <i,j>
				int tmpcount = 0;
				++label;  // begin with a new label
				while (!neighborPixels.empty())
				{
					// get the top pixel on the stack and label it with the same label
					std::pair<int, int> curPixel = neighborPixels.top();
					int curX = curPixel.first;
					int curY = curPixel.second;
					_lableImg.at<int>(curX, curY) = label;
					tmpcount++;
					// pop the top pixel
					neighborPixels.pop();

					// push the 4-neighbors (foreground pixels)
					if (_lableImg.at<int>(curX, curY - 1) == 1)
					{// left pixel
						neighborPixels.push(std::pair<int, int>(curX, curY - 1));
					}
					if (_lableImg.at<int>(curX, curY + 1) == 1)
					{// right pixel
						neighborPixels.push(std::pair<int, int>(curX, curY + 1));
					}
					if (_lableImg.at<int>(curX - 1, curY) == 1)
					{// up pixel
						neighborPixels.push(std::pair<int, int>(curX - 1, curY));
					}
					if (_lableImg.at<int>(curX + 1, curY) == 1)
					{// down pixel
						neighborPixels.push(std::pair<int, int>(curX + 1, curY));
					}
				}
				if (tmpcount > maxlabelcount)
				{
					maxlabel = label;
					maxlabelcount = tmpcount;
				}
			}
		}
	}
	/*find max label*/
	if (maxlabel > 1 && maxlabelcount > 0)
	{
		for (int i = 0; i < mcols; i++)
		{
			for (int j = 0; j < mrows; j++)
			{
				if (_lableImg.at<int >(j, i) == maxlabel)
				{
					BinaryImage.at<uchar>(j, i) = 255;
				}
				else
				{
					BinaryImage.at<uchar>(j, i) = 0;
				}
			}
		}
		return BinaryImage;
	}
	else
	{
		return _binImg;
	}


}
