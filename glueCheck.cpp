#include <stdio.h>
#include <stdlib.h>

//#include <opencv2/core/core.hpp>
//#include <opencv2/imgproc/imgproc.hpp>
//#include <opencv2/highgui/highgui.hpp>
//#include <opencv/cv.h>
#include <opencv2/opencv.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <vector>
#include "glueCheckApi.h"
#include <math.h>
#include <time.h>
#include <string>
#include <map>
#include <io.h>
#include <iostream>
#include <unordered_map>
#include <unordered_map>


#define CX_DEBUG    1
#define USING_TMPLATE_MATCHING 1
#define PI          3.1415926
#define MASK_MARGIN 80
#define STEP_SIZE   64
#define MASK_THD    90

using namespace cv;
using namespace std;
int templateMatchAndCompensation(unsigned char *img, unsigned char *basemaskImg, unsigned char *mask, int width, int height)
{
	Mat capImg = Mat(Size(width, height), CV_8UC1, img);
	Mat maskImg = Mat(Size(width, height), CV_8UC1, basemaskImg);
	Mat glue_mask = Mat(Size(width, height), CV_8UC1, mask);

	/*1.0 resize to 320x240*/
	Mat tinycapImg;
	Mat tinymaskImg;
	resize(capImg, tinycapImg, Size(320, 240));
	resize(maskImg, tinymaskImg, Size(320, 240));

	/*1.1 template selection on capImg, center of the image */
	Mat img_template;
	int t_x = 20;
	int t_y = 20;
	int t_w = 280;
	int t_h = 200;
	img_template = tinycapImg(Rect(t_x, t_y, t_w, t_h));

	/*1.2 template matching*/
	Mat image_matched;
	matchTemplate(tinymaskImg, img_template, image_matched, cv::TM_CCOEFF_NORMED);

	double minVal, maxVal;
	Point minLoc, maxLoc;
	/*find max match location*/
	minMaxLoc(image_matched, &minVal, &maxVal, &minLoc, &maxLoc);

	/*moving distance restore*/
	int d_x = 4 * (t_x - maxLoc.x);
	int d_y = 4 * (t_y - maxLoc.y);
	printf("debug,mask x moving=%d, y moving=%d\r\n", d_x, d_y);
	if (abs(d_x) > 256 || abs(d_y) > 192)
	{
		printf("Warning!! moving greate than expected!!\r\n");

		return 0;
	}
	/*1.3 compensation to mask*/
	Mat mask_comp = Mat::zeros(height, width, CV_8UC1);
	for (int w = 0; w < width; w++)
	{
		for (int h = 0; h < height; h++)
		{
			if ((h + d_y) > 0 && (w + d_x) > 0 && (h + d_y) < height && (w + d_x) < width)
				mask_comp.at<uchar>(h, w) = glue_mask.at<uchar>(h + d_y, w + d_x);
		}
	}

	memcpy(glue_mask.data, mask_comp.data, width*height);
#if CX_DEBUG
	imwrite("D:\\workspace\\gluecheck\\testdata\\0_maskcomped.jpg", mask_comp);
#endif
	return 0;
}

/*function: adaptive threashold segmentation binary step by step*/
Mat adaptiveThresholdStep(Mat glue_roi, Mat glue_mask, int stepsize)
{
	int width = glue_roi.cols;
	int height = glue_roi.rows;
	Mat BinaryImage = Mat(Size(width, height), CV_8UC1);

	int step = (width + stepsize / 2) / stepsize;

	Mat glue_masked;
	Mat inv;
	bitwise_not(glue_mask, inv);//
	add(glue_roi, inv, glue_masked);
	//imshow("test", glue_masked);
	//waitKey(0);
	imwrite("D:\\workspace\\gluecheck\\testdata\\glue_masked.jpg", glue_masked);

	Rect stepRec;
	/*binary segmentation step by step*/
	for (int i = 0; i < step; i++)
	{
		stepRec.x = i * stepsize;
		stepRec.y = 0;
		stepRec.width = min(stepsize, (width - i * stepsize));
		stepRec.height = height;

		Mat tmpAnalysis;
		tmpAnalysis = glue_roi(stepRec);


		Mat stepBinaryImg;
		threshold(tmpAnalysis, stepBinaryImg, 0, 255, CV_THRESH_OTSU | CV_THRESH_BINARY_INV);

		stepBinaryImg.copyTo(BinaryImage(stepRec));
	}

	//close morph
	Mat element = getStructuringElement(MORPH_ELLIPSE, Size(9, 9));
	Mat closeImage;
	morphologyEx(BinaryImage, closeImage, MORPH_OPEN, element);

	Mat glue_out_inv;
	bitwise_not(closeImage, glue_out_inv);//

	Mat img_masked;
	glue_out_inv.copyTo(img_masked, glue_mask);

	Mat glue_mask_inv;
	bitwise_not(glue_mask, glue_mask_inv);//

	Mat glue_outline;
	add(img_masked, glue_mask_inv, glue_outline);

	/*thd connect analysis*/
	/*connected analysis to get main glue*/
	Mat labelImg;
	Mat glue_outline_inv;
	Mat glue_label_inv;
	bitwise_not(glue_outline, glue_outline_inv);//
	labelImg = icvprCcaBySeedFill(glue_outline_inv);
	bitwise_not(labelImg, glue_label_inv);//
	return glue_label_inv;
}

/*function: using binayImg and edge Img of maskded to finetune a dedicated  glue binary*/
Mat mergeMaskedEdgeAndBinay(Mat binayImg, Mat edgeImg)
{
	int width = binayImg.cols;
	int height = binayImg.rows;
	Mat fineBinayImg = binayImg.clone();
	vector<int> upEdge;
	vector<int> downEdge;


	/*step1,using binayImg up and down edge to reduce edge image*/
	for (int i = 0; i < width; i++)
	{
		upEdge.push_back(-1);
		downEdge.push_back(-1);

		for (int j = 0; j < height; j++)
		{
			int a = binayImg.at<uchar>(j, i);
			edgeImg.at<uchar>(j, i) = 0;
			if (a == 0)
			{
				edgeImg.at<uchar>(j, i) = 255;
				upEdge[i] = j;
				break;
			}
		}

		for (int j = height - 1; j > 0; j--)        //获得下边界坐标
		{
			int a = binayImg.at<uchar>(j, i);
			edgeImg.at<uchar>(j, i) = 0;
			if (a == 0)
			{
				edgeImg.at<uchar>(j, i) = 255;
				downEdge[i] = j;
				break;
			}
		}
	}
	int margin = 20;
	/*step2,using 20% more edge detect and merge*/
	for (int i = 0; i < width; i++)
	{
		int k = 0;
		int up_h = upEdge[i];
		if (up_h > 0)
		{
			for (k = up_h + margin; k > up_h; k--)
			{
				if (edgeImg.at<uchar>(k, i) > 0)
				{
					edgeImg.at<uchar>(k, i) = 255;
					upEdge[i] = k;
					break;
				}
			}
			for (int m = up_h; m < k; m++)
			{
				edgeImg.at<uchar>(m, i) = 0;
			}
		}
		int down_h = downEdge[i];
		if (down_h > 0)
		{
			for (k = down_h - margin; k < down_h; k++)
			{
				if (edgeImg.at<uchar>(k, i) > 0)
				{
					edgeImg.at<uchar>(k, i) = 255;
					downEdge[i] = k;
					break;
				}
			}
			for (int m = k; m < down_h; m++)
			{
				edgeImg.at<uchar>(m, i) = 0;
			}
		}

	}

	/*step3, using stepsize filter to smoth the edge*/
	int myfilter[] = { 1,4,10,4,1 };
	for (int i = 2; i < width - 2; i++)
	{
		if (upEdge[i - 2] > 0 && upEdge[i - 1] > 0 && upEdge[i] > 0 && upEdge[i + 1] > 0 && upEdge[i + 2] > 0)
		{
			upEdge[i] = (upEdge[i - 2] * myfilter[0] + upEdge[i - 1] * myfilter[1] + upEdge[i] * myfilter[2] + upEdge[i + 1] * myfilter[3] + upEdge[i + 2] * myfilter[4]) / 20;
		}
		if (downEdge[i - 2] > 0 && downEdge[i - 1] > 0 && downEdge[i] > 0 && downEdge[i + 1] > 0 && downEdge[i + 2] > 0)
		{
			downEdge[i] = (downEdge[i - 2] * myfilter[0] + downEdge[i - 1] * myfilter[1] + downEdge[i] * myfilter[2] + downEdge[i + 1] * myfilter[3] + downEdge[i + 2] * myfilter[4]) / 20;
		}
	}

	/*step4, using edge to deal with binay image*/
	for (int i = 0; i < width; i++)
	{
		int up_h = upEdge[i];
		if (up_h > 0)
		{
			for (int k = 0; k < up_h; k++)
			{
				fineBinayImg.at<uchar>(k, i) = 255;
			}
			fineBinayImg.at<uchar>(up_h, i) = 0;
		}

		int down_h = downEdge[i];
		if (down_h > 0)
		{
			for (int k = down_h; k < height; k++)
			{
				fineBinayImg.at<uchar>(k, i) = 255;
			}
			fineBinayImg.at<uchar>(down_h, i) = 0;
		}
	}
#if CX_DEBUG
	imwrite("D:\\workspace\\gluecheck\\testdata\\edge_masked_reduce.jpg", edgeImg);
#endif


	return fineBinayImg;
}

/*using glue center line and it's direciton to evaluate size*/
int Distanceget(Mat BinaryStep, Mat affine_img_mask, int x_begin, int x_end, double *results)
{
	vector<Point> c1Point, c11Point;        //上边缘
	vector<Point> c2Point, c22Point;        //下边缘
	vector<Point> MPoint;                   //中心线
	vector<Point> MPoint2;
	Point pt1, pt11;                        //上边缘上的点
	Point pt2, pt22;                        //下边缘上的点
	Point pt3, pt33;                        //中心线上的点
	vector<float> Xielv;                    //中心线上的切线斜率
	vector<int>   direct;       //垂直方向的距离
	vector<float> result;       //
	vector<float> angle;        //切线角度
	vector<float> angle2;       //切线角度（正值）
	Mat images;                 //图像
	Mat images2;
	int max_index = 0;            //最大涂胶位置索引
	int min_index = 0;            //最小涂胶位置索引
	int max_indey = 0;          //最大涂胶位置索引
	int min_indey = 0;          //最小涂胶位置索引
	double maxangeltan = 0;     //
	double minangeltan = 0;     //
	double temp2;
	double xielv2[20];
	vector<double> aa;
	images = BinaryStep;
	images2 = affine_img_mask;
	//边缘坐标
	for (int i = 0; i < images.cols; i++)
	{
		for (int j = 0; j < images.rows; j++)        //获得上边界坐标
		{
			int a = images.at<uchar>(j, i);

			if (a == 0)
			{
				pt1.x = i;
				pt1.y = j;
				c1Point.push_back(pt1);
				for (int k = 0; k < images.rows; k++)
				{
					int b = images2.at<uchar>(k, i);
					if (b == 255)
					{
						pt11.x = i;
						pt11.y = k;
						c11Point.push_back(pt11);
						break;
					}
				}
				break;
			}
		}
		for (int jj = images.rows - 1; jj > -1; jj--)        //获得下边界坐标
		{
			int aa2 = images.at<uchar>(jj, i);
			if (aa2 == 0)
			{
				pt2.x = i;
				pt2.y = jj;
				c2Point.push_back(pt2);
				for (int k = images.rows - 1; k > -1; k--)
				{
					int b = images2.at<uchar>(k, i);
					if (b == 255)
					{
						pt22.x = i;
						pt22.y = k;
						c22Point.push_back(pt22);
						break;
					}
				}
				break;
			}
		}
	}
	// 胶体和mask的中线点坐标
	for (int i = 0; i < c1Point.size(); i++)
	{
		Point a1 = c1Point[i];
		Point a2 = c2Point[i];
		Point a11 = c11Point[i];
		Point a22 = c22Point[i];
		pt3.x = (a1.x + a1.x) / 2;  pt3.y = (a1.y + a2.y) / 2;
		pt33.x = (a11.x);  pt33.y = (a11.y + a22.y) / 2;
		direct.push_back(a2.y - a1.y);                 //   上下边界的距离
	//	MPoint2.push_back(pt33);                       //   mask中心线坐标
		MPoint.push_back(pt33);                       //   胶体中心线坐标
	}
	// 计算每段胶宽
	for (int i = 0; i < MPoint.size() - 1; i++)
	{
		if (i < 13)
		{
			Point begin = MPoint[1];
			Point end1 = MPoint[14];
			float a = end1.y - begin.y;
			float b = end1.x - begin.x;

			float temp = abs(a / b);                  //切线值
			temp2 = a / b;
			float temp3 = atan(temp);                 //切线角度
			float fix_direct = direct[i] * cos(temp3);
			result.push_back(fix_direct);             //最终结果
			angle2.push_back(temp2);
			angle.push_back(temp3);
		}
		else if (i < MPoint.size() - 16)
		{
			int nn = 13;
			double temp4 = 0;
			for (int kk = 0; kk < nn; kk++)
			{
				Point begin = MPoint[i - nn + kk];
				Point end1 = MPoint[i + kk + 1];
				float a = end1.y - begin.y;
				float b = end1.x - begin.x;
				xielv2[kk] = a / b / nn;
				temp4 = temp4 + xielv2[kk];
			}
			temp2 = temp4;
			float temp = abs(temp2);
			float temp3 = atan(temp);
			float fix_direct = direct[i] * cos(temp3);
			result.push_back(fix_direct);
			angle.push_back(temp3);
			angle2.push_back(temp2);
		}
		else
		{
			int   aa = MPoint.size();
			Point begin = MPoint[aa - 10];
			Point end1 = MPoint[aa - 1];
			float a = end1.y - begin.y;
			float b = end1.x - begin.x;
			float temp = abs(a / b);
			temp2 = a / b;
			float temp3 = atan(temp);
			float fix_direct = direct[i] * cos(temp3);
			result.push_back(fix_direct);
			angle.push_back(temp3);
			angle2.push_back(temp2);
		}
	}

	float max = result[MPoint.size()*0.15];  float min = result[MPoint.size()*0.15];
	// 获得最宽 +最窄的胶宽
	for (int i = MPoint.size()*0.1; i < angle.size()*0.9; i++)
	{
		if (result[i] > max)
		{
			max = result[i];
			Point maxcoord = MPoint[i];              //最大位置坐标
			max_index = maxcoord.x;                  //最大位置x方向索引
			max_indey = maxcoord.y;                  //最小位置x方向索引
			maxangeltan = angle2[i];
			continue;
		}
		else if (result[i] < min)
		{
			min = result[i];
			Point mincoord = MPoint[i];            //最小位置坐标
			min_index = mincoord.x;                //最小位置x方向索引
			min_indey = mincoord.y;                //最小位置x方向索引
			minangeltan = angle2[i];
			continue;
		}
	}
	results[0] = min;
	results[1] = min_index;
	results[2] = min_indey;
	results[3] = minangeltan;
	results[4] = max;
	results[5] = max_index;
	results[6] = max_indey;
	results[7] = maxangeltan;

	return 0;
}

Point getOriginalPoint(Point psrc, Mat affineM, int x_offset, int y_offset)
{
	int x_dst = 0;
	int y_dst = 0;
	double x_src = (double)psrc.x;
	double y_src = (double)psrc.y;
	double a0 = affineM.at<double>(0, 0);
	double b0 = affineM.at<double>(0, 1);
	double c0 = affineM.at<double>(0, 2);
	double a1 = affineM.at<double>(1, 0);
	double b1 = affineM.at<double>(1, 1);
	double c1 = affineM.at<double>(1, 2);

	x_dst = x_offset + (int)((b1*(x_src - c0) - b0 * (y_src - c1)) / (a0*b1 - a1 * b0));
	y_dst = y_offset + (int)((a1*(x_src - c0) - a0 * (y_src - c1)) / (b0*a1 - b1 * a0));

	return Point(x_dst, y_dst);
}
/*glue detection API*/

int GlueDetectionAlgorithm(unsigned char *img, unsigned char *basemaskImg, unsigned char *mask, unsigned char *retImg, int type, int width, int height, double *ret, double * input)
{
	Mat glue_img = Mat(Size(width, height), CV_8UC1, img);
	Mat glue_basemask_img = Mat(Size(width, height), CV_8UC1, basemaskImg);
	Mat glue_mask = Mat(Size(width, height), CV_8UC1, mask);
	Mat glue_ret = Mat(Size(width, height), CV_8UC3, retImg);
	int hret;

	/*step1.0,find mask image and curr capture image perspective relation,then compensate on the mask*/
#if USING_TMPLATE_MATCHING
	hret = templateMatchAndCompensation(img, basemaskImg, mask, width, height);
	if (hret != 0)
	{
		printf("warning: bad homograpy find,perhaps lost its location!! error code=%d\r\n", hret);
	}
#endif
	/*step1.1, find the mask contour of ROI rect*/
	vector<vector<Point> > contours;
	vector<Vec4i> hierarcy;

	findContours(glue_mask, contours, hierarcy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
	vector<Rect> boundRect(contours.size());
	vector<RotatedRect> box(contours.size());
	Point2f rect[4];
	int object_idx = 0;

	/*step1.2, min rect and rotated rect ,contors size should be only 1*/
	if (contours.size() > 1)
	{
		printf("warning: there should only one contours in the pattern image!but there is %d \r\n", contours.size());
		/*find max contours index*/
		double recmax = 0.0;
		for (int i = 0; i < contours.size(); i++)
		{
			box[i] = minAreaRect(Mat(contours[i]));
			boundRect[i] = boundingRect(Mat(contours[i]));
			if (recmax < boundRect[i].area())
			{
				recmax = boundRect[i].area();
				object_idx = i;
			}
		}
		printf("warning: using  %d contours as bigest one for analysis\r\n", object_idx);
	}
	else
	{
		box[object_idx] = minAreaRect(Mat(contours[object_idx]));
		boundRect[object_idx] = boundingRect(Mat(contours[object_idx]));

	}
	float angle;
	angle = box[object_idx].angle;
	Point2f center = box[object_idx].center;
	Point center_1 = center;
	/*step2,using Rect of mask to roated to horizon,and get it out */
	Mat glue_rec_mask = glue_mask(boundRect[object_idx]);
	if (center.x < boundRect[object_idx].x || center.y < boundRect[object_idx].y)
	{
		printf("err: min rotated_rect find error, not within the Rect\r\n");
	}
	else
	{
		double angle0 = angle;
		double scale = 1;
		printf("box.size.width=%f,box.size.height=%f \r\n", box[object_idx].size.width, box[object_idx].size.height);
		if (box[object_idx].size.width > box[object_idx].size.height)
		{
			angle0 = angle0 + 90;
		}
		//printf("angle%f\r\n", angle0);
		int target_width = 0;
		int target_height = 0;
		if (glue_rec_mask.cols < glue_rec_mask.rows)
		{
			angle0 = angle0 + 90;
			target_width = (int)(((double)glue_rec_mask.rows)*1.1 / abs(sin(angle0*PI / 180)));
			target_height = glue_rec_mask.cols;
			center.x = boundRect[object_idx].width / 2;
			center.y = boundRect[object_idx].height / 2;
		}
		else
		{
			angle0 = angle0 + 90;
			target_width = (int)(((double)glue_rec_mask.rows)*1.1 / abs(sin(angle0*PI / 180)));
			target_width = min(target_width, (int)(1.414 * glue_rec_mask.cols));
			target_height = glue_rec_mask.rows;
			center.x = boundRect[object_idx].width / 2;
			center.y = boundRect[object_idx].height / 2;
		}

		Mat roateM = getRotationMatrix2D(center, angle0, scale);
		roateM.at<double>(0, 2) += (target_width / 2 - boundRect[object_idx].width / 2);
		roateM.at<double>(1, 2) += target_height / 2 - boundRect[object_idx].height / 2;

		cout << "angle0==" << angle0 << endl;
		/*step2.1 Affine image translation to current rect,both mask and img*/
		Mat glue_rec_img = glue_img(boundRect[object_idx]);
		Mat affine_img;
		warpAffine(glue_rec_img, affine_img, roateM, Size(target_width, target_height));
#if CX_DEBUG
		imwrite("D:\\workspace\\gluecheck\\testdata\\0_glue_rec.jpg", glue_rec_img);
		imwrite("D:\\workspace\\gluecheck\\testdata\\0_glue_horizon.jpg", affine_img);
#endif
		int width = affine_img.cols;
		int height = affine_img.rows;

		/*step2.2 affine also mask image so get ROI translated*/
		Mat glue_mask_img = glue_mask(boundRect[object_idx]);
		Mat affine_img_mask;
		warpAffine(glue_mask_img, affine_img_mask, roateM, Size(target_width, target_height));
#if CX_DEBUG
		imwrite("D:\\workspace\\gluecheck\\testdata\\0_mask_rec.jpg", glue_mask_img);
		imwrite("D:\\workspace\\gluecheck\\testdata\\mask_roated.jpg", affine_img_mask);
		//imshow("test", affine_img_mask);
		//waitKey();
#endif
		/*step3, find mask x begin and end*/
		int mask_x_begin = 0;
		int mask_x_end = 0;
		int mask_thd = MASK_THD;
		for (int x = 0; x < width; x++)
		{
			int sum_all = 0;
			for (int y = 0; y < height; y++)
			{
				sum_all += affine_img_mask.at<uchar>(y, x);
			}
			sum_all = sum_all / 255;
			if (sum_all > mask_thd)
			{
				mask_x_begin = x;
				break;
			}

		}
		for (int x = width - 1; x > 0; x--)
		{
			int sum_all = 0;
			for (int y = 0; y < height; y++)
			{
				sum_all += affine_img_mask.at<uchar>(y, x);
			}
			sum_all = sum_all / 255;
			if (sum_all > mask_thd)
			{
				mask_x_end = x;
				break;
			}

		}
		//		printf("debug,mask begin  =%d, mask end =%d\r\n", mask_x_begin, mask_x_end);
				/*step4, using step adaptive threshold to binary the outline image and then do some MORPH*/
		Mat BinaryStep;
		int stepsize = STEP_SIZE;
		BinaryStep = adaptiveThresholdStep(affine_img, affine_img_mask, stepsize);

#if CX_DEBUG
		imwrite("D:\\workspace\\gluecheck\\testdata\\0_binaryImageStep.jpg", BinaryStep);
#endif


		/*step5,analysis affine image to get size at each x*/
		int  minsize = 0;
		int  maxsize = 0;

		int minx = 0;       //最小距离所在的中心线的坐标
		int miny = 0;       //最小距离所在的中心线的坐标
		int maxx = 0;       //最大距离所在的中心线的坐标
		int maxy = 0;       //最大距离所在的中心线的坐标
		int midx = 0;       //图像中心所在的中心线的坐标
		int midy = 0;       //图像中心所在的中心线的坐标
		int a_1, a_11;        //
		int a_2, a_22;        //
		double mintan;      //最小直线距离所在的斜率
		double maxtan;      //最长直线距离所在的斜率
		int mask_bargin = MASK_MARGIN;
		Mat checkImg = BinaryStep;
		int f_x_begin;
		int f_x_end;
		double results[10]; //传递输出参数
		//Point Pcmin;
		//Point Pcmax;
		//Point pmin1;
		//Point pmin2;
		//Point pmax1;
		//Point pmax2;
		f_x_begin = min((mask_x_begin + mask_bargin), mask_x_end);
		f_x_end = max(f_x_begin, (mask_x_end - mask_bargin));
		Distanceget(BinaryStep, affine_img_mask, mask_x_begin, mask_x_end, results);
		minsize = results[0];
		minx = results[1];
		miny = results[2];
		mintan = results[3];
		if (abs(mintan) < 0.01)
			mintan = 0;
		maxsize = results[4];
		maxx = results[5];
		maxy = results[6];
		maxtan = results[7];
		if (abs(maxtan) < 0.01)
			maxtan = 0;


		a_1 = 200 / 2 * sin(atan(mintan))*1.2;              // /sqrt(1 + mintan * mintan);  // 计算划线长度
		a_2 = 200 / 2 * sin(atan(maxtan))*1.2;              // /sqrt(1 + maxtan * maxtan);  // 计算划线长度

		cout << "a_1 = " << a_1 << "a_2 = " << a_2 << endl;
		printf("debug,minsize=%d at minx=%d, maxsize=%d at maxx=%d\r\n", minsize, minx, maxsize, maxx);

		/*step6: judge the correct and draw it with different color line on target image*/
		Mat dstImg = glue_img.clone();
		Mat affine_img_check;
		int x_offset = boundRect[object_idx].x;
		int y_offset = boundRect[object_idx].y;
		cvtColor(dstImg, dstImg, CV_GRAY2BGR);

		cvtColor(checkImg, affine_img_check, CV_GRAY2BGR);
		///////////////////////////////////////
		cv::Mat contours;
		cv::Canny(affine_img_check, contours, 200, 400);          //获得边界
		for (int i = 0; i < contours.cols; i++)
		{
			for (int j = 0; j < contours.rows; j++)        //获得上边界坐标
			{
				int a = contours.at<uchar>(j, i);
				if (a == 255)
				{
					Vec3b pixel;
					pixel[0] = 0;        //Blue
					pixel[1] = 255;      //Green
					pixel[2] = 0;         //Red
					Point ppp = getOriginalPoint(Point(i, j), roateM, x_offset, y_offset);
					dstImg.at<Vec3b>(ppp) = pixel;
					break;
				}
			}
			for (int jj = contours.rows - 1; jj > -1; jj--)        //获得下边界坐标
			{
				int aa2 = contours.at<uchar>(jj, i);
				if (aa2 == 255)
				{
					Vec3b pixel;
					pixel[0] = 0;        //Blue
					pixel[1] = 255;      //Green
					pixel[2] = 0;         //Red
					Point ppp = getOriginalPoint(Point(i, jj), roateM, x_offset, y_offset);
					dstImg.at<Vec3b>(ppp) = pixel;
					break;
				}
			}
		}
		///////////////////////////////////
		Scalar xminx, xmaxx;
		float minsize1 = minsize * input[2];
		float maxsize1 = maxsize * input[2];
		cout << "zui xiao = " << minsize1 << "zui da = " << maxsize1 << "inputmin=  " << input[0] << "inputmax= " << input[1] << endl;

		if (minsize1 < input[0])
			xminx = Scalar(0, 0, 255);
		else if (minsize1 < input[1])
			xminx = Scalar(0, 255, 0);
		else
			xminx = Scalar(0, 255, 255);
		if (maxsize1 < input[0])
			xmaxx = Scalar(0, 0, 255);
		else if (maxsize1 < input[1])
			xmaxx = Scalar(0, 255, 0);
		else
			xmaxx = Scalar(0, 255, 255);

		Point pp0, pp1;
		pp0 = getOriginalPoint(Point(minx, miny), roateM, x_offset, y_offset);
		pp1 = getOriginalPoint(Point(maxx, maxy), roateM, x_offset, y_offset);
		if (mintan < 0)
		{
			int ppp = a_1 / mintan;
			line(affine_img_check, Point(minx + a_1, miny - a_1 / mintan), Point(minx - a_1, miny + a_1 / mintan), xminx, 4);

			/*draw on origianl canvas */
			Point p0 = getOriginalPoint(Point(minx + a_1, miny - a_1 / mintan), roateM, x_offset, y_offset);
			Point p1 = getOriginalPoint(Point(minx - a_1, miny + a_1 / mintan), roateM, x_offset, y_offset);
			line(dstImg, p0, p1, xminx, 5);
		}
		else if (mintan > 0)
		{
			line(affine_img_check, Point(minx - a_1, miny + a_1 / mintan), Point(minx + a_1, miny - a_1 / mintan), xminx, 4);
			/*draw on origianl canvas */
			Point p0 = getOriginalPoint(Point(minx - a_1, miny + a_1 / mintan), roateM, x_offset, y_offset);
			Point p1 = getOriginalPoint(Point(minx + a_1, miny - a_1 / mintan), roateM, x_offset, y_offset);
			line(dstImg, p0, p1, xminx, 5);
		}
		else
		{
			line(affine_img_check, Point(minx, miny + minsize / 2 * 1.3), Point(minx, miny - minsize / 2 * 1.3), xminx, 4);
			/*draw on origianl canvas */
			Point p0 = getOriginalPoint(Point(minx, miny + minsize / 2 * 1.4), roateM, x_offset, y_offset);
			Point p1 = getOriginalPoint(Point(minx, miny - minsize / 2 * 1.4), roateM, x_offset, y_offset);
			line(dstImg, p0, p1, xminx, 5);
		}
		if (maxtan > 0)
		{
			line(affine_img_check, Point(maxx + a_2, maxy - a_2 / maxtan), Point(maxx - a_2, maxy + a_2 / maxtan), xmaxx, 4);

			/*draw on origianl canvas */
			Point p2 = getOriginalPoint(Point(maxx + a_2, maxy - a_2 / maxtan), roateM, x_offset, y_offset);
			Point p3 = getOriginalPoint(Point(maxx - a_2, maxy + a_2 / maxtan), roateM, x_offset, y_offset);
			line(dstImg, p2, p3, xmaxx, 5);
		}
		else if (maxtan < 0)
		{
			line(affine_img_check, Point(maxx - a_2, maxy + a_2 / maxtan), Point(maxx + a_2, maxy - a_2 / maxtan), xmaxx, 4);

			/*draw on origianl canvas */
			Point p2 = getOriginalPoint(Point(maxx - a_2, maxy + a_2 / maxtan), roateM, x_offset, y_offset);
			Point p3 = getOriginalPoint(Point(maxx + a_2, maxy - a_2 / maxtan), roateM, x_offset, y_offset);
			line(dstImg, p2, p3, xmaxx, 5);
		}
		else
		{
			line(affine_img_check, Point(maxx, maxy + maxsize / 2 * 1.4), Point(maxx, maxy - maxsize / 2 * 1.3), xmaxx, 4);

			/*draw on origianl canvas */
			Point p2 = getOriginalPoint(Point(maxx, maxy + maxsize / 2 * 1.4), roateM, x_offset, y_offset);
			Point p3 = getOriginalPoint(Point(maxx, maxy - maxsize / 2 * 1.4), roateM, x_offset, y_offset);
			line(dstImg, p2, p3, xmaxx, 5);
		}

		char minglue[20], maxglue[20];
		sprintf(minglue, "%.1f", minsize1);
		sprintf(maxglue, "%.1f", maxsize1);
		int putmin = MIN((width - 40), minx - 20);
		int putmax = MIN((width - 40), maxx - 20);;
		//cout << xminx << "jdjdj " << xmaxx << endl;
		putText(dstImg, minglue, pp0 - Point(20, 20), CV_FONT_HERSHEY_COMPLEX_SMALL, 1.5, xminx);
		putText(dstImg, maxglue, pp1 - Point(20, 20), CV_FONT_HERSHEY_COMPLEX_SMALL, 1.5, xmaxx);

#if CX_DEBUG
		imwrite("D:\\workspace\\gluecheck\\testdata\\0_glue_horizon_check.jpg", affine_img_check);
#endif
		/*step8,return just*/
		box[object_idx].points(rect);
		for (int j = 0; j < 4; j++)
		{
			line(dstImg, rect[j], rect[(j + 1) % 4], Scalar(255, 0, 0), 2, 8);
		}
		dstImg.copyTo(glue_ret);

		ret[0] = minsize1;
		ret[1] = maxsize1;

	}
	return 0;
}

/*

void getFiles(string path, vector<string>& files)
{
	//文件句柄
	long long hFile = 0;//这个地方需要特别注意，win10用户必须用long long 类型，win7可以用long类型
	//文件信息
	struct _finddata_t fileinfo;
	string p;
	if ((hFile = _findfirst(p.assign(path).append("\\*").c_str(), &fileinfo)) != -1)
	{
		do
		{
			//如果是目录,迭代之
			//如果不是,加入列表
			if ((fileinfo.attrib &  _A_SUBDIR))
			{
				if (strcmp(fileinfo.name, ".") != 0 && strcmp(fileinfo.name, "..") != 0)
					getFiles(p.assign(path).append("\\").append(fileinfo.name), files);
			}
			else
			{
				files.push_back(p.assign(path).append("\\").append(fileinfo.name));
			}
		} while (_findnext(hFile, &fileinfo) == 0);
		_findclose(hFile);
	}
}
*/

unordered_map<string, int> defineRegion() {
    //TODO
    /**
    * @brief Find the center of each ROI mask (region of interest)
    * @param
    */

    unordered_map<string, int> ret;
    ret["f5_C_mask.bmp"] = 484;
    ret["f6_C_mask.bmp"] = 467;
    ret["f1_C_mask.bmp"] = 583;
    ret["f4_C_mask.bmp"] = 502;
    ret["f3_C_mask.bmp"] = 502;
    ret["f2_C_mask.bmp"] = 533;

    return ret;
}

Mat processOutput(Mat m) { //4-dimensional Mat with NCHW dimensions order.
    Mat ret = Mat(m.size[2], m.size[3], CV_8UC1);
    //float* pt0 = m.ptr<float>(0);
    //float* pt1 = &pt0[0];


    for (int i = 0; i < m.size[2]; i++) {
        float* data = m.ptr<float>(i);
        for (int j = 0; j < m.size[3]; j++) {
            if (data[j] >= 0.5) ret.at<uchar>(j, i) = 255;
            else ret.at<uchar>(j, i) = 0;
        }
    }
    //cout << ret <<endl;

    return ret;

}

Mat forwardProp(dnn::Net net, Mat blob){
    net.setInput(blob);
    Mat output = net.forward();
    Mat ret = processOutput(output);
    //cout << "check output" << endl;
    //cout << output.size << endl;
    return ret;
}

void predict(dnn::Net net, Mat img, Mat &retMask, int center, int offset, int cropsize) {
    int h = img.rows;
    int w = img.cols;
    cout << "t1" << img.rows << endl;
    int startx = max(center - offset, 0);
    int endx = min(center + offset, w);

    for (int i = 0; i < h; i += cropsize) {
        for (int j = startx; j < endx; j += cropsize) {
            Mat curr = img(Rect(i, j, cropsize, cropsize));
            cout << "aa"<< curr.size <<endl;
            retMask(Rect(i, j, cropsize, cropsize)) = forwardProp(net, curr);

        }
    }

}

int getCenterFromName(string name) {
    unordered_map<string, int> centermap = defineRegion();
    int center = centermap[name];
    return center;
}


void glueDetectionDnn(Mat img, Mat &retMask, string weights, int center, int cropsize, int offset){

    int h = img.rows;
    int w = img.cols;
    //int cropNum = (h * w) / (cropsize * cropsize);
    //vector<Mat> matlist;

    dnn::Net net = cv::dnn::readNetFromTensorflow(weights);

    int startx = max(center - offset, 0);
    int endx = min(center + offset, w);

    for (int i = 0; i < h; i += cropsize) {
        for (int j = startx; j < endx; j += cropsize) {

            Mat curr = img(Rect(j, i, cropsize, cropsize));
             namedWindow("test1", WINDOW_AUTOSIZE);
            imshow("test1", curr);
            waitKey(0);
            //cout << curr.size << endl;
            //retMask(Rect(i, j, cropsize, cropsize)) = forwardProp(net, curr);
            //matlist.push_back(curr);
            cv::Mat blob = cv::dnn::blobFromImage(curr, 1./255, Size(cropsize,cropsize), 127.0, false, false, CV_32F);
            //cout << blob.size << endl;
            forwardProp(net, blob).copyTo(retMask(Rect(j, i, cropsize, cropsize)));
            //cout << retMask(Rect(i, j, cropsize, cropsize)) <<endl;
            namedWindow("test2", WINDOW_AUTOSIZE);
            imshow("test2", retMask(Rect(j, i, cropsize, cropsize)));
            waitKey(0);
        }
    }




    //cv::Mat blob = cv::dnn::blobFromImage(img, 1./255, Size(cropsize,cropsize), 127.0, false, false, CV_32FC1);
    //blob here has type CV_8U, can cause accuracy problem

    //return retMask;

}


int main()
{


    Mat img = imread("/home/zhuyipin/CV/GlueSegmentation/images/01409758_ASBLD0901_AS22_1_OK.bmp");
    Mat img_maskimg = imread("/home/zhuyipin/CV/GlueSegmentation/images/01409758_ASBLD0901_AS22_1_OK.bmp");
    Mat img_mask = imread("/home/zhuyipin/CV/GlueSegmentation/images/f1_C_mask.bmp");
    string modelpath = "/home/zhuyipin/CV/GlueSegmentation/models/unet_sh32.pb";

    Mat gray;
    Mat gray_mask;
    Mat gray_maskimg;




    cvtColor(img, gray, CV_BGR2GRAY);
    cvtColor(img_maskimg, gray_maskimg, CV_BGR2GRAY);
    cvtColor(img_mask, gray_mask, CV_BGR2GRAY);
    /*label of size per pixel and the correct glue size range*/

    double sizePerPixel = 1;
    double min_glue_thd = 9;
    double max_glue_thd = 14;
    double inputValue[6];

    inputValue[0] = 9;
    inputValue[1] = 12;
    inputValue[2] = 0.1;
    int type_line = 1;
    int width = img.cols;
    int height = img.rows;
    Mat retImg = Mat(Size(width, height), CV_32FC1); //8 bit with 3 channels
    double retValue[6];

    int cropsize = 160;
    int offset = 240;

    int center = getCenterFromName("f1_C_mask.bmp");
    //cout << "what is gray" << gray.rows << endl;
    Mat retMask = Mat::zeros(img.rows, img.cols, CV_8UC1);
    glueDetectionDnn(gray, retMask, modelpath, center, cropsize, offset);

    double minp = 0.0, maxp = 0.0;
    minMaxIdx(retMask, &minp, &maxp);
    cout << minp << " " <<  maxp<< endl;
    namedWindow("test", WINDOW_AUTOSIZE);
    imshow("test", retMask);
    waitKey(0);


    //gray: img convert to gray, 3 channels  to 1 channel
    //gray_mask: bbox mask , ROI mask
    //gray_masking: same as gray img
    //retImg: image returned after boundary detection. pass by reference
    //type_line:
    //width, height: width and height of original image
    //retValue[6]:  retValue[0] is min width (num of pixels), retValue[1] is max (num of pixels)
    //inputvalue[6]: 9, 12, 0.1

    //GlueDetectionAlgorithm(gray.data, gray_maskimg.data, gray_mask.data, retImg.data, type_line, width, height, retValue, inputValue);
/*
    glueDetectionDnn(modelpath)

    double min_measure = retValue[0] * sizePerPixel; //returnValue
    double max_measure = retValue[1] * sizePerPixel;

    bool result_bad_flag = false;
    if (min_measure < min_glue_thd)
    {
        printf("check out err: min size=%f less than thd=%f \r\n", min_measure, min_glue_thd);
        result_bad_flag = true;
    }
    if (max_measure > max_glue_thd)
    {
        printf("check out err: max size=%f max than thd=%f\r\n", max_measure, max_glue_thd);
        result_bad_flag = true;
    }
    if (result_bad_flag == false)
        printf("running result,glue min=%f,max=%f,pass!\r\n", min_measure, max_measure);

    imwrite("/home/zhuyipin/CV/GlueSegmentation/images/ret/ret.jpg", retImg);
*/

	return 0;


}
