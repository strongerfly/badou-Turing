#include<opencv2/opencv.hpp>
#include<string>
#include<iostream>
#include<ctime>
#include<random>

using namespace cv;
using namespace std;

void gaussianNoise(Mat& src, Mat& dst, double snRatio, double sigma=1, double mean=0){
    int rows = src.rows;
    int cols = src.cols;
    int channels = src.channels();

    int noiseNum = (int) rows*cols*snRatio;   // 信噪比，计算有多少像素点添加噪声
    
    dst = src.clone();

    //高斯分布
    default_random_engine e;
    normal_distribution<double> normal(mean, sigma);  // 高斯分布
    
    srand(time(0));
    for(int i=0; i<noiseNum; i++){
        int r = rand()%rows;
        int c = rand()%cols;
        if(channels==1){   // 单通道图像处理
            double temp = src.at<uchar>(r, c) + normal(e);
            temp = temp>255 ? 255: temp;
            temp = temp<0 ? 0: temp;
            dst.at<uchar>(r, c) = (uchar) temp;
        }else if (channels==3)  // 3通道图像处理
        {
            for(int j=1; j<3; j++){
                double temp = src.at<Vec3b>(r, c)[j] + normal(e);
                temp = temp>255 ? 255: temp;
                temp = temp<0 ? 0: temp;
                dst.at<Vec3b>(r, c)[j] = (uchar) temp;
            }
        }
    }

}



int main(int argc, char* argv[]){
    if(argc<2){
        cout << " Wrong arguments!!, Right arguments are as fllows:" << endl;
        cout << "Example: ./gaussianNoise.exe imgPath" << endl;
        return -1;
    }
    string imgPath = argv[1];
    Mat img = imread(imgPath);
    if(!img.data){
        cout << "loading image failed: " << imgPath << endl;
        return -1;
    }
    Mat noiseImg;
    gaussianNoise(img, noiseImg, 0.5, 8, 40);
    imshow("img", img);
    imshow("noiseImg", noiseImg);
    waitKey(0);
    destroyAllWindows();
    return 0;
}