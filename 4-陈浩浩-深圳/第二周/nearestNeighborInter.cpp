#include<opencv2/opencv.hpp>
#include<string>

using namespace std;
using namespace cv;

void nearestNeighborInterpolation(Mat img, Mat dst, int width, int height){
    float wRatio = (float)width/img.cols;
    float hRatio = (float)height/img.rows;
    for(int i=0; i<dst.rows; i++){
        for(int j=0; j<dst.cols; j++){
            int srcI = round(i/hRatio);
            int srcJ = round(j/wRatio);
            dst.at<Vec3b>(i,j) = img.at<Vec3b>(srcI, srcJ);
        }
    }

}

int main(int argc, char* argv[]){
    // string imgFile = "F:\\tmp\\data\\1625045452754.jpg";    
    if(argc<4) {
        cout << " Wrong arguments!!, Right arguments are as fllows:" << endl;
        cout << "Example: ./biLinearInter.exe imgPath width height" << endl;
        return 0;
    }
    string imgFile = argv[1];
    int width = atoi(argv[2]);
    int height = atoi(argv[3]);
    Mat img = imread(imgFile);
    Mat dstImg = Mat::zeros(height, width, CV_8UC3);
    nearestNeighborInterpolation(img, dstImg, width, height);
    imshow("srcImg", img);
    imshow("dstImg", dstImg);
    waitKey(0);
    destroyAllWindows();
    return 0;
}