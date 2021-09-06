#include<opencv2/opencv.hpp>
#include<string>
#include<ctime>
#include<iostream>

using namespace std;
using namespace cv;

void pepperSaltNoise(Mat& src, Mat& dst, double snRatio){
    int rows = src.rows;
    int cols = src.cols;
    int channels = src.channels();
    int noiseNum = (int) (snRatio*rows*cols);  // 信噪比，计算有多少像素点添加噪声
    dst = src.clone();
    srand(time(0));
    for(int i=0; i<noiseNum; i++){
        int r = rand()%rows;
        int c = rand()%cols;
        double temp = rand()%RAND_MAX; //0-1之间随机数
        if(channels==1){      // 单通道图像处理
            if(temp>0.5){
                dst.at<uchar>(r, c) = 0;
            }else{
                dst.at<uchar>(r, c) = 255;
            }
        }else if (channels==3)  // 3通道图像处理
        {
            if(temp>0.5){
                dst.at<Vec3b>(r, c) = Vec3b(0,0,0);
            }else{
                dst.at<Vec3b>(r, c) = Vec3b(255,255,255);
            }
        }
    }

}

int main(int argc, char* argv[]){
    if(argc<2){
        cout << " Wrong arguments!!, Right arguments are as fllows:" << endl;
        cout << "Example: ./pepperSaltNoise.exe imgPath" << endl;
        return -1;
    }
    string imgPath = argv[1];
    Mat img = imread(imgPath);
    if(!img.data){
        cout << "Loading image failed: " << imgPath << endl;
        return -1;
    }

    Mat noiseImg;
    pepperSaltNoise(img, noiseImg, 0.2);
    imshow("img", img);
    imshow("noiseImg", noiseImg);
    waitKey(0);
    destroyAllWindows();
    return 0;
}