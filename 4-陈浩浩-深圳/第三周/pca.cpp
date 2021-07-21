#include<iostream>
#include<opencv2/opencv.hpp>

using namespace cv;
using namespace std;
void pca(Mat& src, Mat& dst, int k){
    // src: n*m, 表示n个数据，每个数据的维度为m
    Mat srcBack = src.clone();
    
    // 减去均值，中心化
    for(int i=0; i<src.cols-1; i++){
        Scalar m = mean(src.colRange(i, i+1));
        srcBack.colRange(i, i+1) -= m[0];
    }
    Mat covMat =srcBack.t()*srcBack/(srcBack.rows-1);   // m*m的协方差矩阵
    Mat eValuesMat, eVectorsMat;
    eigen(covMat, eValuesMat, eVectorsMat); //求解特征向量
    
    //特征值降序排列，得到前k个特征向量
    Mat idx;
    sortIdx(eValuesMat,idx,SORT_EVERY_COLUMN+SORT_DESCENDING);
    Mat projectMat = Mat::zeros(k, eVectorsMat.cols, CV_32FC1);  //k个特征向量组成的映射矩阵，k*m
    for(int i=0; i<k; i++){
        int rowIndex = idx.at<int>(i, 0);
        eVectorsMat.rowRange(rowIndex, rowIndex+1).copyTo(projectMat.rowRange(rowIndex, rowIndex+1));
    }
    dst = src*projectMat.t();  //映射矩阵，进行降维
}


int main(int argc, char* argv[]){
    // Mat samples = (Mat_<float>(5, 3) <<10,20, 10, 20, 10, 30, 60, 40, 60, 50, 60, 70, 30, 90, 30);
    Mat samples = Mat::zeros(100, 10, CV_32FC1);  
    randu(samples, Scalar(0), Scalar(256));   //随机初始化100*6的矩阵

    cout << samples.size()<< endl;
    Mat pcaSamples;
    pca(samples, pcaSamples, 6); //10维降维到6维
    cout <<pcaSamples.size() << endl;
    return 0;
}