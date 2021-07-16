#include<iostream>
#include<opencv2/opencv.hpp>
#include<string>
using namespace cv;
using namespace std;
void conv(Mat& srcImg, Mat& dstImg, Mat& kernel, int padding, int stride){
    int rows = srcImg.rows;
    int cols = srcImg.cols;
    int kr = kernel.rows;
    int kc = kernel.cols;

    //添加padding后的原图
    Mat paddingImg = Mat::zeros(rows+2*padding, cols+2*padding, CV_32FC3);
    srcImg.copyTo(paddingImg(Rect(padding, padding, cols, rows)));
    
    //计算卷积后图片的大小
    int dw = (int) (cols+2*padding-kc)/stride +1;
    int dh = (int) (rows+2*padding-kr)/stride +1;
    dstImg = Mat::zeros(dh, dw, CV_8UC3);
    
    int di=0, dj=0;
    Vec3f v={0,0,0};
    for(int i=0; i<(rows+2*padding-kr+1); i+=stride){
        for(int j=0; j<(cols+2*padding-kc+1); j+=stride){
            for(int k=0; k<kr; k++){
                for(int m=0;m<kc;m++){
                    v = v + paddingImg.at<Vec3f>(i+k, j+m)*kernel.at<float>(k,m);  //逐项相乘后相加
                }
            }
            dstImg.at<Vec3b>(di, dj) = (Vec3b) v;
            dj++;
            v={0,0,0};
        }
        di++;
        dj=0;
    }
}


int main(int argc, char* argv[]){
    if(argc<2){
        cout << " Wrong arguments!!, Right arguments are as fllows:" << endl;
        cout << "Example: ./conv.exe imgPath" << endl;
        return -1;
    }
    string imgPath = argv[1];
    Mat img = imread(imgPath);
    if(!img.data){
        cout << "loading img failed: " <<imgPath << endl;
        return -1;
    }
    Mat kernel = (Mat_<float>(3,3)<<1,0,-1,1,0,-1,1,0,-1); //3*3的卷积核
    cout << kernel << endl;  
    int padding = 1;    //边界填充
    int stride = 2;    //步长
    Mat dstImg;
    conv(img, dstImg, kernel, padding, stride);
    
    imshow("srcImg", img);
    imshow("dstImg", dstImg);
    waitKey(0);
    destroyAllWindows();
    return 0;
}