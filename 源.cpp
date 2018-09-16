#include <io.h>
#include <string>
#include <vector>
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2\highgui\highgui.hpp>
#include <opencv2\imgproc\imgproc.hpp>

using namespace cv;
using namespace std;



//定义sobel算子用来计算梯度
int SOBEL_X[3][3] = { { -1, 0, 1 },
					{ -2, 0, 2 },
					{ -1, 0, 1 } };
int SOBEL_Y[3][3] = { { -1, -2, -1 },
					{ 0,  0,  0 },
					{ 1,  2,  1 } };


unsigned char clamp(unsigned char value)
{
	if (value > 255) 
	{value = 255;}
	else 
	{
		if (value < 0)
		{
			value = 0;
		}
	}
	return value;
}

//计算梯度函数
Mat GradientFilter(Mat src, double* angle)
{
	Mat dst = Mat(src.rows, src.cols, src.type());
	int height = src.rows;
	int width = src.cols;
	int row, col;
	float x, y, g;
	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			x = y = 0;
			for (int subrow = -1; subrow <= 1; subrow++)
			{
				for (int subcol = -1; subcol <= 1; subcol++)
				{
					row = i + subrow;
					col = j + subcol;
					if (row < 0 || row >= height)
						row = i;
					if (col < 0 || col >= width)
						col = j;
					x += SOBEL_X[subrow + 1][subcol + 1] * src.data[row * width + col];
					y += SOBEL_Y[subrow + 1][subcol + 1] * src.data[row * width + col];
				}
			}
			g = sqrt(x * x + y * y);
			dst.data[i * width + j] = clamp((unsigned char)g);
			if (x == 0)
				angle[i * width + j] = (y > 0 ? 180.0 : 0.0);
			else if (y == 0)
				angle[i * width + j] = 90;
			else
				angle[i * width + j] = atan(x / y) + 90;
		}
	}
	
	return dst;
}

//非最大值抑制，细化梯度边缘
Mat NonMaximalSuppression(Mat src, double* angle)
{
	Mat dst = Mat(src.rows, src.cols, src.type());
	int height = src.rows;
	int width = src.cols;
	int row, col;
	int index;
	unsigned char t;
	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			index = i * width + j;
			t = *(dst.data + index) = *(src.data + index);
			if ((angle[index] >= 0 && angle[index] < 22.5) || (angle[index] >= 157.5 && angle[index] < 180))
			{
				row = i;
				col = (j - 1) < 0 ? j : (j - 1);
				if (t < *(src.data + row * width + col))
					*(dst.data + index) = 0;

				row = i;
				col = (j + 1) >= width ? j : (j + 1);
				if (t < *(src.data + row * width + col))
					*(dst.data + index) = 0;
			}
			else if (angle[index] >= 22.5 && angle[index] < 67.5)
			{
				row = (i - 1) < 0 ? i : (i - 1);
				col = (j + 1) >= width ? j : (j + 1);
				if (t < *(src.data + row * width + col))
					*(dst.data + index) = 0;

				row = (i + 1) >= height ? i : (i + 1);
				col = (j - 1) < 0 ? j : (j - 1);
				if (t < *(src.data + row * width + col))
					*(dst.data + index) = 0;
			}
			else if (angle[index] >= 67.5 && angle[index] < 112.5)
			{
				row = (i - 1) < 0 ? i : (i - 1);
				col = j;
				if (t < *(src.data + row * width + col))
					*(dst.data + index) = 0;

				row = (i + 1) >= height ? i : (i + 1);
				col = j;
				if (t < *(src.data + row * width + col))
					*(dst.data + index) = 0;
			}
			else if (angle[index] >= 112.5 && angle[index] < 157.5)
			{
				row = (i - 1) < 0 ? i : (i - 1);
				col = (j - 1) < 0 ? j : (j - 1);
				if (t < *(src.data + row * width + col))
					*(dst.data + index) = 0;

				row = (i + 1) >= height ? i : (i + 1);
				col = (j + 1) >= width ? j : (j + 1);
				if (t < *(src.data + row * width + col))
					*(dst.data + index) = 0;
			}
		}
	}
	
	return dst;
}

//边缘连接
void edgeLink(Mat src, Mat dst, int row, int col, int threshold)
{
	int height = src.rows;
	int width = src.cols;
	int row1, col1;
	*(dst.data + row * width + col) = 255;
	for (int i = -1; i <= 1; i++)
	{
		for (int j = -1; j <= 1; j++)
		{
			row1 = row + i;
			col1 = col + j;
			row1 = row1 < 0 ? 0 : (row1 >= height ? (height - 1) : row1);
			col1 = col1 < 0 ? 0 : (col1 >= width ? (width - 1) : col1);
			if (*(src.data + row1 * width + col1) > threshold && *(dst.data + row1 * width + col1) == 0)
			{
				edgeLink(src, dst, row1, col1, threshold);
				return;
			}
		}
	}
}

Mat DoubleThresholdEdgeConnection(Mat src, int lowThreshold, int highThreshold)
{
	Mat dst = Mat(src.rows, src.cols, src.type(), Scalar(0));
	int height = src.rows;
	int width = src.cols;
	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			if (*(src.data + i * width + j) >= highThreshold && *(dst.data + i * width + j) == 0)
				edgeLink(src, dst, i, j, lowThreshold);
		}
	}
	return dst;
}



int main()
{
	int TL = 20;
	int TH = 60;
	Mat GaussianImg, gradientImg, NMSImg, finalImg, BinaryzationImg;
	string path, pic;
	double *angle;
	Mat srcImg, grayImg;
	srcImg = imread("van_original.jpg" );
	while (srcImg.empty())
	{
		cout << "未读取图片" << endl;
		srcImg = imread(path + "\\" + pic);
	}

	angle = (double *)malloc(sizeof(double) * srcImg.rows * srcImg.cols);

	cvtColor(srcImg, grayImg, CV_RGB2GRAY);
	imshow("原始图像",srcImg);

	GaussianBlur(grayImg, GaussianImg, Size(5, 5), 0, 0);//用高斯滤波器初步处理图像

	gradientImg = GradientFilter(GaussianImg, angle);
	imshow("原始梯度图", gradientImg);
	GaussianBlur(gradientImg, gradientImg, Size(9, 9), 0, 0);
	NMSImg = NonMaximalSuppression(gradientImg, angle);
	imshow("细化梯度边缘", NMSImg);
																	//free(angle);

	finalImg = DoubleThresholdEdgeConnection(NMSImg, TL, TH);
	imshow("边缘检测", finalImg);

	waitKey(0);
	return 0;
}