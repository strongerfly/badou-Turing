#include <iostream>
#include<iostream>
#include <opencv2\opencv.hpp>
#include <opencv2\imgproc\imgproc.hpp>
#include <math.h>
#include <atlstr.h>
#include <winbase.h>
#include<time.h>

using namespace std;
using namespace cv;


clock_t startTime, endTime;

//src:����
//result:�����ǩ
//num���ֲ����
//means_z������ƽ��ֵ
//zmax:z���ֵ ����ȡ����
//zmin��z��Сֵ
//judge������ֹͣ��׼
bool my_kmeans(cv::Mat src, cv::Mat &result, int num, double &means_z, int& max_num, double zmax, double zmin, int diedai_num,double judge = 5)
{
	//��ʼ��Ϊ0
	means_z = 0;
	//��ʼ�������ǩ
	result = cv::Mat::zeros(src.size(), CV_8UC1);
	//cout << src.type() << endl;

	//����
	double *center_num, *center_num2;
	center_num = new double[num];
	//����2 ���������Ƿ���������
	center_num2 = new double[num];

	//������������
	int *gray_num;
	gray_num = new int[num];
	//��ֵ����
	double aatest = zmax - zmin;
	
	center_num[0] = zmax;
	//��ʼ��Ϊ0��
	gray_num[0] = 0;
	center_num[num-1] = zmin+1;
	gray_num[num-1] = 0;
	for (int i = 1; i < num-1; i++)
	{
		center_num2[i] = 0;
		gray_num[i] = 0;
		center_num[i] = ((i+1)* aatest / num);
	}
	//result.resize(num);
	//��������
	int diedai_Index = 0;
	//��һ�ν���͵ڶ��ν���Ĳ�ֵ
	double value = 999;

	while (value > (judge*10))
	{
		value = 0;
		diedai_Index++;

		//����
		for (int i = 0; i < src.rows; i++)
		{
			uchar *data_src = src.ptr<uchar>(i);
			if (src.type() == 5) {
				float *data_src = src.ptr<float>(i);
			}
			
			uchar *data_result = result.ptr<uchar>(i);
			for (int j = 0; j < src.cols; j++)
			{

				double src_gray = data_src[j];
				int index = 0;
				double min = 999;
				for (int k = 0; k < num; k++)
				{
					double dis = abs(center_num[k] - src_gray);
					if (min >= dis)
					{
						min = dis;
						index = k;
					}
				}
				data_result[j] = index;
				gray_num[index]++;
			}
		}

		for (int i = 0; i < num; i++)
		{
			center_num2[i] = center_num[i];
			center_num[i] = 0;
		}

		//�ö�Ӧ����ߵ�������ƽ��ֵ �õ��µ�����
		for (int i = 0; i < src.rows; i++)
		{
			int center_new = 0;
			uchar *data_src = src.ptr<uchar>(i);
			if (src.type() == 5) {
				float *data_src = src.ptr<float>(i);
			}
			uchar *data_result = result.ptr<uchar>(i);
			for (int j = 0; j < src.cols; j++)
			{
				//center_num[data_result[j]] += (double(data_src[j])/ double(gray_num[data_result[j]]));
				center_num[data_result[j]] += double(data_src[j]);
			}
		}
		//�õ�����������±�
		max_num = -9999;
		for (int i = 0; i < num; i++)
		{
			center_num[i] = center_num[i] / double(gray_num[i]);
			//value += abs(center_num2[i] - center_num[i]);
			value += abs(center_num2[i] - center_num[i]);
			if (gray_num[i] > max_num)
			{
				max_num = gray_num[i];
				means_z = i;
			}
			gray_num[i] = 0;
		}
		value *= 10.0;
		//����������ʮ�������� ��ʾ����ʧ��
		if (diedai_Index > diedai_num) {

			std::cout << "my_kmeans diedai Error" << std::endl;
			return 1;
		}
	}

	//��ֵƽ��ֵ
	means_z = center_num[int(means_z)];
	std::cout << "diedai_Index:" << diedai_Index << std::endl;

	delete[] center_num;
	delete[] center_num2;
	delete[] gray_num;
	return 0;
}


int main()
{

	Mat src = imread("C:\\Users\\lenovo\\Desktop\\1.jpg");
	if (src.empty())
	{
		cout << "image open error" << endl;
		return 1;
	}

	Mat gray;
	cvtColor(src, gray, COLOR_BGR2GRAY);

	const int num = 4;

	//imshow("gray", gray);
	//waitKey(0);
	startTime = clock();//��ʱ��ʼ
	Mat result;
	double means_z;
	int max_num;
	my_kmeans(gray, result, num, means_z, max_num, 255, 0,32, 0.1);

	cout << "means_z:" << means_z << "max_num:" << max_num << endl;

	int *array;
	array = new int[num];

	for (int i = 1; i < num + 1; i++)
	{
		array[i - 1] = i * 255 / num;
	}
	for (int i = 0; i < src.rows; i++)
	{
		uchar *data_gray = gray.ptr<uchar>(i);
		uchar *data_result = result.ptr<uchar>(i);
		for (int j = 0; j < src.cols; j++)
		{
			data_gray[j] = array[data_result[j]];
		}
	}

	endTime = clock();//��ʱ����
	cout << "The run time is: " << (double)(endTime - startTime) << "ms" << endl;


	imshow("gray2", gray);
	waitKey(0);
	return 0;
}