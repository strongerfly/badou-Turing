#include<iostream>
#include<opencv2/opencv.hpp>
#include<string>

using namespace cv;
using namespace std;

int main(int argc, char* argv[]){
    string imgPath = argv[1];
    Mat img = imread(imgPath);
    if(!img.data){
        cout << "loading image failed: " << imgPath << endl;
        return -1;
    }
    Mat edges;
    Canny(img, edges, 30, 200);

    imshow("img", img);
    imshow("edges", edges);
    waitKey(0);
    destroyAllWindows();

    return 0;
}

//void Canny(cv::InputArray image, cv::OutputArray edges, double threshold1, double threshold2, int apertureSize = 3, bool L2gradient = false)
// threshold1: 低阈值
// threshold2: 高阈值
// apertureSize：sobel算子大小，默认采用3*3
// L2gradient：true是采用L2方式计算梯度(更精确), false时采用L1