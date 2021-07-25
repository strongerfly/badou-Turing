#include<iostream>
#include<opencv2/opencv.hpp>

using namespace cv;
using namespace std;
Mat getPerspectiveMatrix(Mat& src, Mat& dst){
    assert(src.size[0]==dst.size[0] && src.size[0>=4]);
    int nums = src.size[0];
    Mat A = Mat::zeros(nums*2, 8, CV_32FC1);
    Mat B = Mat::zeros(nums*2, 1, CV_32FC1);
    Mat warpMatrix = Mat::ones(3, 3, CV_32FC1);
    
    //每对坐标构造两个方程，nums=4时四组坐标，共八个方程
    for(int i=0; i<nums; i++){
        //第一个方程
        A.at<float>(2*i, 0) = src.at<float>(i,0);
        A.at<float>(2*i, 1) = src.at<float>(i,1);
        A.at<float>(2*i, 2) = 1;
        A.at<float>(2*i, 6) = -src.at<float>(i,0)*dst.at<float>(i, 0);
        A.at<float>(2*i, 7) = -src.at<float>(i,1)*dst.at<float>(i, 0);

        //第二个方程
        A.at<float>(2*i+1, 3) = src.at<float>(i,0);
        A.at<float>(2*i+1, 4) = src.at<float>(i,1);
        A.at<float>(2*i+1, 5) = 1;
        A.at<float>(2*i+1, 6) = -src.at<float>(i,0)*dst.at<float>(i, 1);
        A.at<float>(2*i+1, 7) = -src.at<float>(i,1)*dst.at<float>(i, 1);
    }

    B = dst.reshape(0, 8).clone();
    Mat w = A.inv()*B;   //如果A不可逆时，返回矩阵w为全0矩阵
    for(int i=0; i<3; i++){
        for(int j=0; j<3; j++){
            if(i*3+j<8) warpMatrix.at<float>(i, j) = w.at<float>(i*3+j, 0); //矩阵最后一位保持为1
        }
    }
    return warpMatrix;
}


int main(int argc, char* argv[]){
    Mat src = (Mat_<float>(4, 2)<<10.0, 457.0, 395.0, 291.0, 624.0, 291.0, 1000.0, 457.0);
    Mat dst = (Mat_<float>(4, 2)<<46.0, 920.0, 46.0, 100.0, 600.0, 100.0, 600.0, 920.0);
    Mat warpMatrix = getPerspectiveMatrix(src, dst);
    cout << warpMatrix << endl;
    return 0;
}
