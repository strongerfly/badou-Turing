#include <iostream>
#include<iostream>
#include <opencv2\opencv.hpp>
#include <opencv2\imgproc\imgproc.hpp>
#include <math.h>
#include <atlstr.h>
#include <winbase.h>

using namespace std;
using namespace cv;


//最近邻插值算法
//width宽度 height高度
bool nearest_interp(cv::Mat src, cv::Mat& dst, double width, double height)
{
	try
	{


		//由缩放因子计算输出图像的尺寸（四舍五入）

		//创建输出图像
		dst = cv::Mat::zeros(Size(width, height), src.type());
		
		//double Scale_x = height / double(src.rows);
		//double Scale_y = width / double(src.cols);
		double Scale_x = height / double(src.rows);
		double Scale_y = width / double(src.cols);
		
		int row = dst.rows;
		int col = dst.cols;
		//uchar *data_dst;

		for (int i = 0; i < row-1; i++) {

			for (int j = 0; j < col-1; j++) {

				if (src.channels() == 1)
				{


					//插值计算，输出图像的像素点由原图像对应的最近的像素点得到（四舍五入）
					 //data_dst = dst.ptr<uchar>(i);
					int i_index = round(i / Scale_x);
					if (i_index > src.rows - 1) i_index = src.rows - 1;//防止越界
					//uchar *data_src = src.ptr<uchar>(i_index);
					int j_index = round(j / Scale_y);
					if (j_index > src.cols - 1) j_index = src.cols - 1;//防止越界
					/*if (a != i_index || b != j_index)
					{
						cout << endl;
					}*/

					dst.ptr<uchar>(i)[j] = src.ptr<uchar>(i_index)[j_index];
					int aa = src.ptr<uchar>(i_index)[j_index];
					if (dst.ptr<uchar>(i)[j] >= 255)dst.ptr<uchar>(i)[j] = 255;
					if (dst.ptr<uchar>(i)[j] <= 0)dst.ptr<uchar>(i)[j] = 0;
				}
				else
				{
					int i_index = round(i / Scale_x);
					if (i_index > src.rows - 1) i_index = src.rows - 1;//防止越界
					//插值计算，输出图像的像素点由原图像对应的最近的像素点得到（四舍五入）
					int j_index = round(j / Scale_y);
					if (j_index > src.cols - 1) j_index = src.cols - 1;//防止越界

					//uchar *data_src = src.ptr<uchar>(i_index, j_index);
					//uchar *data_dst = dst.ptr<uchar>(i, j);
					dst.ptr<uchar>(i, j)[0] = src.ptr<uchar>(i_index, j_index)[0];
					dst.ptr<uchar>(i, j)[1] = src.ptr<uchar>(i_index, j_index)[1];
					dst.ptr<uchar>(i, j)[2] = src.ptr<uchar>(i_index, j_index)[2];
					if (dst.ptr<uchar>(i, j)[0] >= 255)dst.ptr<uchar>(i, j)[0] = 255;
					if (dst.ptr<uchar>(i, j)[1] >= 255)dst.ptr<uchar>(i, j)[1] = 255;
					if (dst.ptr<uchar>(i, j)[2] >= 255)dst.ptr<uchar>(i, j)[2] = 255;
					if (dst.ptr<uchar>(i, j)[0] <= 0)dst.ptr<uchar>(i, j)[0] = 0;
					if (dst.ptr<uchar>(i, j)[1] <= 0)dst.ptr<uchar>(i, j)[1] = 0;
					if (dst.ptr<uchar>(i, j)[2] <= 0)dst.ptr<uchar>(i, j)[2] = 0;
				}
				//dst.at<uchar>(i, j) = src.at<uchar>(i_index, j_index);
			}
		}

	}
	catch (const std::exception&)
	{
		cout << "nearest_interp:error" << endl;
		return 0;
	}
	return 1;
	
	
}

//二值化
//src:原图
//dst:二值图
bool BGRtogray(Mat &src, Mat & dst)
{
	try
	{
		Mat gray;
		cvtColor(src, gray, COLOR_BGR2GRAY);
		dst = Mat::zeros(Size(gray.cols, gray.rows), gray.type());
		int row = gray.rows;
		int col = gray.cols;

		for (int i = 0; i < row-1; i++)
		{
			uchar *datasrc = gray.ptr<uchar>(i);
			uchar *datadst = dst.ptr<uchar>(i);
			
			for (int j = 0; j < col-1; j++)
			{

				datasrc[j] <= 125 ? datadst[j] = 255 : datadst[j] = 0;
			}
		}
	}
	catch (const std::exception&)
	{
		cout << "BGRtogray:error" << endl;
		return 0;
	}
	return 1;
}

//双线性插值
bool bilinear_interpolation(Mat src,Mat &dst,double width,double height)
{
	try
	{

		dst = Mat::zeros(Size(width, height), src.type());
		int dst_row = dst.rows;
		int dst_col = dst.cols;
		double Scale_x = height / double(src.rows);
		double Scale_y = width / double(src.cols);
		Point2i Q11, Q12, Q21, Q22;
		double src_x, src_y;
		uchar * data_dst;
		int color;
		for (int i = 0; i < dst_row; i++)
		{
			for (int j=0; j < dst_col; j++)
			{
			
					src_x = (double(i) + 0.5) / Scale_x - 0.5;
					src_y = (double(j) + 0.5) / Scale_y - 0.5;

					Q11.x = floor(src_x + 0.5);
					Q11.y = floor(src_y + 0.5);

					Q12.x = floor(src_x + 0.5);
					Q12.y = floor(src_y + 1.5 <= src.cols - 1 ? src_y + 1.5 : src.cols - 1);
					//Q12.y = floor(src_y + 1.5);

					Q21.x = floor(src_x + 1.5 <= src.rows - 1 ? src_x + 1.5 : src.rows - 1);
					//Q21.x = floor(src_x + 1.5);
					Q21.y = floor(src_y + 0.5);


					Q22.x = floor(src_x + 1.5 <= src.rows - 1 ? src_x + 1.5 : src.rows - 1);
					//Q22.x = floor(src_x + 1.5 );
					Q22.y = floor(src_y + 1.5 <= src.cols - 1 ? src_y + 1.5 : src.cols - 1);
					//Q22.y = floor(src_y + 1.5 );
				//单通道
				if (src.channels() == 1)
				{
					//double fr1 = (Q21.x - src_x) * src.at<uchar>(Q11.x, Q11.y) + (src_x - Q11.x)*src.at<uchar>(Q21.x, Q21.y);
					double fr1 = (Q21.x - src_x) * src.ptr<uchar>(Q11.x)[Q11.y] + (src_x - Q11.x)*src.ptr<uchar>(Q21.x)[Q21.y];

					double fr2 = (Q21.x - src_x) *src.ptr<uchar>(Q12.x)[Q12.y] + (src_x - Q11.x)*src.ptr<uchar>(Q22.x)[Q22.y];
					 color = floor((Q12.y - src_y) *fr1 + (src_y - Q11.y) *fr2 + 0.5);
				
					if (j >= dst_col - 2||i>=dst_row-2)color = src.ptr<uchar>(int(src_x + 0.5))[int(src_y + 0.5)];
					if (color <= 0) color = 0;
					if (color >= 255) color = 255;

					dst.ptr<uchar>(i)[j] = color;
				}
				//rgb通道
				else
				{
					for (int k = 0; k <3; k++)
					{
						double fr1 = (Q21.x - src_x) * src.ptr<uchar>(Q11.x,Q11.y)[k] + (src_x - Q11.x)*src.ptr<uchar>(Q21.x,Q21.y)[k];

						double fr2 = (Q21.x - src_x) *src.ptr<uchar>(Q12.x, Q12.y)[k] + (src_x - Q11.x)*src.ptr<uchar>(Q22.x, Q22.y)[k];
						 color = floor((Q12.y - src_y) *fr1 + (src_y - Q11.y) *fr2 + 0.5);

						if (j >= dst_col - 2 || i >= dst_row - 2)color = src.ptr<uchar>(int(src_x + 0.5), int(src_y + 0.5))[k];
						if (color <= 0) color = 0;
						if (color >= 255) color = 255;
						dst.ptr<uchar>(i,j)[k] = color;
					}

				}
		    
			}
	
		}
	}
	catch (const std::exception&)
	{
		cout << "bilinear_interpolation:error" << endl;
		return 0;
	}
	return 1;
}


int main()
{
	Mat srcimg = imread("C:\\Users\\lenovo\\Desktop\\圆盘0531\\pic_2d1.bmp");
	if (!srcimg.data) {
		return 0;
	}
	Mat srcgray;
	cvtColor(srcimg, srcgray, COLOR_BGR2GRAY);

	Mat dst_gray;
	Mat dst_nearest_interp;
	Mat dst_bilinear_interpolation;

	//二值化
	BGRtogray(srcimg, dst_gray);
	Mat gray;
	cvtColor(srcimg, gray, COLOR_BGR2GRAY);
	//最近邻插值
	nearest_interp(srcimg, dst_nearest_interp,200,800);

	//双线性插值
	bilinear_interpolation(srcimg, dst_bilinear_interpolation, 200, 800);

	cvNamedWindow("srcimg", 1);
	imshow("srcimg", srcimg);

	cvNamedWindow("dst_gray", 1);
	imshow("dst_gray", dst_gray);

	cvNamedWindow("dst_nearest_interp", 1);
	imshow("dst_nearest_interp", dst_nearest_interp);

	cvNamedWindow("dst_bilinear_interpolation", 1);
	imshow("dst_bilinear_interpolation", dst_bilinear_interpolation);


	waitKey(0);
	return 0;
}