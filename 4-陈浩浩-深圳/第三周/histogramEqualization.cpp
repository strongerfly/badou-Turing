#include<iostream>
#include<opencv2/opencv.hpp>
#include<string>

using namespace std;
using namespace cv;


void grayHistEqualization(Mat& grayImg, Mat& dstImg){
    Mat hist;
    Mat cumHist = Mat::zeros(Size(256, 1), CV_32FC1);
    Mat output = Mat::zeros(Size(256, 1), CV_32FC1);
    int h=grayImg.rows;
    int w = grayImg.cols;
    float branges[] = {0, 256};
    const float* ranges[] = {branges};
    const int histSize[] = {256};
    const int channels[] = {0};
    
    //统计单通道直方图
    calcHist(&grayImg, 1, channels, Mat(), hist, 1, histSize,ranges,true, false);
    
    for(int i=0; i<256; i++){
        //累积直方图
        if(i==0){
            cumHist.at<float>(0, i) = hist.at<float>(0,0);
        }else{
            cumHist.at<float>(0, i) = cumHist.at<float>(0, i-1) + hist.at<float>(0, i);
        }
        //计算像素映射关系
        float q = cumHist.at<float>(0,i)*256/(h*w) - 1;
        if(q>=0){
            output.at<float>(0, i) = q;
        }else{
            output.at<float>(0, i) = 0;
        }
    }
    
    //像素映射
    for(int r=0; r<h; r++){
        for(int c=0; c<w; c++){
            dstImg.at<uchar>(r, c) = (uchar)output.at<float>(0, grayImg.at<uchar>(r, c));
        }
    }        
}

int main(int argc, char* argv[]){
    if(argc<2){
        cout << " Wrong arguments!!, Right arguments are as fllows:" << endl;
        cout << "Example: ./histogramEqualization.exe imgPath" << endl;
        return -1;
    }
    string imgPath = argv[1];
    string savePath = "./HistLena.png";
    Mat srcImg = imread(imgPath);
    if(!srcImg.data){
        cout << "loading image failed: " << imgPath << endl;
        return -1;
    }
    // 灰度图直方图均衡化
    Mat dstImg = Mat::zeros(srcImg.cols, srcImg.rows, CV_8UC1);
    Mat grayImg;
    cvtColor(srcImg, grayImg, COLOR_BGR2GRAY);
    grayHistEqualization(grayImg, dstImg);
    imshow("srcImg", grayImg);
    imshow("dstImg", dstImg);
    waitKey(0);
    destroyAllWindows();


    //三通道直方图均衡化
    // Mat bgr[3];
    // Mat bgrHist[3];
    // split(srcImg, bgr); //分离通道
    // for(int i=0; i<3; i++){
    //     Mat dstImg = Mat::zeros(srcImg.cols, srcImg.rows, CV_8UC1);
    //     grayHistEqualization(bgr[i], dstImg);
    //     dstImg.copyTo(bgrHist[i]);
    // }
    // Mat bgrDistImg;
    // merge(bgrHist, 3, bgrDistImg);  //合并通道
    // cout <<bgrDistImg.at<Vec3b>(0,0) << endl;
    // imshow("srcImg", srcImg);
    // imshow("bgrDistImg", bgrDistImg);
    // waitKey(0);
    // destroyAllWindows();
    return 0;
}