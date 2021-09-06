#include<opencv2/opencv.hpp>
#include<string>
using namespace std;
using namespace cv;

void rgb2gray(Mat& img, Mat& gray){
    int rows = img.rows;
    int cols = img.cols;
    for(int i=0; i<rows; i++){
        for(int j=0; j<cols; j++){
            Vec3b vec = img.at<Vec3b>(i, j);
            gray.at<uchar>(i, j) = (11*vec[2]+59*vec[1]+30*vec[0])/100;
            // cout <<gray.at<int>(i, j) <<",";   
        }
    }
}

int main(int argc, char* argv[]){
    // string imgFile = "F:\\tmp\\data\\1625045452754.jpg";
    string imgFile = argv[1];
    Mat img = imread(imgFile);
    Mat imgGray = Mat(img.rows,img.cols, CV_8UC1, Scalar(0));
    // Mat imgGray2 = Mat::zeros(img.rows,img.cols, CV_8UC1);
    rgb2gray(img, imgGray);
    cout << imgGray.size() << endl;
    // cout << img.type() << endl;
    imshow("img", img);
    imshow("imgGray2", imgGray);
    waitKey(0);
    destroyAllWindows();
    
    return 0;
}
