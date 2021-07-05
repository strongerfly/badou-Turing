#include<opencv2/opencv.hpp>
#include<string>
#include<algorithm>

using namespace std;
using namespace cv;
//双线性插值过程中：为了保持缩放后图片映射为原图时，图片的中心点重合，满足等式：
//srcI+0.5 = (dstI+0.5)*srcHeight/dstHeight
//srcJ+0.5 = (dstJ+0.5)*srcWidth/dstWidth
void biLinearInterpolation(Mat img, Mat dstImg, int width, int height){
    float wRatio = (float)width/img.cols;
    float hRatio = (float)height/img.rows;
    for(int i=0; i<height; i++){
        for(int j=0; j<width; j++){
            float srcI = (i+0.5)/hRatio-0.5;
            float srcJ = (j+0.5)/wRatio-0.5;
            int leftTopI = max((int) srcI, 0);
            int leftTopJ = max((int) srcJ, 0);
            int rightBottomI = min((int) (srcI+1), img.rows-1);
            int rightBottomJ = min((int) (srcJ+1), img.cols-1);
            for(int k=0; k<3; k++){
                float topInter = img.at<Vec3b>(leftTopI, leftTopJ)[k]*(rightBottomJ-srcJ)+img.at<Vec3b>(leftTopI, rightBottomJ)[k]*(srcJ-leftTopJ);
                float bottomInter = img.at<Vec3b>(rightBottomI, leftTopJ)[k]*(rightBottomJ-srcJ)+img.at<Vec3b>(rightBottomI, rightBottomJ)[k]*(srcJ-leftTopJ);
                dstImg.at<Vec3b>(i, j)[k] = (uchar)(topInter*(rightBottomI-srcI)+bottomInter*(srcI-leftTopI));
                // cout <<(int) dstImg.at<Vec3b>(i, j)[k] << endl;
            }
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
    biLinearInterpolation(img, dstImg, width, height);
    imshow("srcImg", img);
    imshow("dstImg", dstImg);
    waitKey(0);
    destroyAllWindows();

    return 0;
}