#include<opencv2/opencv.hpp>
#include<iostream>
#include<string>

using namespace cv;
using namespace std;


// 均值哈希算法
void aHash(Mat& src, vector<int>& hashCode){
    Mat sImg, grayImg;
    resize(src, sImg, Size(8, 8));
    cvtColor(sImg, grayImg, COLOR_BGR2GRAY);
    int meanValue = mean(grayImg)[0];  // 返回值为Scalar对象，取第一个
    for(int i=0; i<8; i++){
        for(int j=0; j<8; j++){
            if(grayImg.at<uchar>(i, j) > meanValue){
                hashCode.push_back(1);
            }else{
                hashCode.push_back(0);
            }
        }
    }
}

// 差值哈希算法
void dHash(Mat& src, vector<int>& hashCode){
    Mat sImg, grayImg;
    resize(src, sImg, Size(9, 8));
    cvtColor(sImg, grayImg, COLOR_BGR2GRAY);
    
    for(int i=0; i<8; i++){
        for(int j=0; j<8; j++){
            if(grayImg.at<uchar>(i, j) > grayImg.at<uchar>(i, j+1)){
                hashCode.push_back(1);
            }else{
                hashCode.push_back(0);
            }
        }
    }
}

// 感知哈希算法
void pHash(Mat& src, vector<int>& hashCode){
    Mat sImg, grayImg, dctMatrix;
    
    resize(src, sImg, Size(32, 32));
    cvtColor(sImg, grayImg, COLOR_BGR2GRAY);
    
    grayImg.convertTo(grayImg, CV_32F);  // 转换为float32
    dct(grayImg, dctMatrix);            // 进行dct变换 （离散余弦变换）
    Mat dctMatrixROI = dctMatrix(Rect(0, 0, 8, 8));  //截取左上角8*8的区域
    int meanValue = mean(dctMatrixROI)[0];

    for(int i=0; i<8; i++){
        for(int j=0; j<8; j++){
            if(dctMatrixROI.at<uchar>(i, j) > meanValue){
                hashCode.push_back(1);
            }else{
                hashCode.push_back(0);
            }
        }
    }
}

// 汉明距离
int hammingDistance(vector<int>& h1, vector<int>& h2){
    assert(h1.size()==h2.size());
    int d = 0;
    for(int i=0; i<h1.size(); i++){
        if(h1[i]!=h2[i]) d++;
    }
    return d;
}


// 采用hash算法和汉明距离，实现图像相似度比较
int main(int argc, char* argv[]){
    if(argc<3){
        cout << " Wrong arguments!!, Right arguments are as fllows:" << endl;
        cout << "Example: ./imageHsah.exe imgPath1 imgPath2" << endl;
        return -1;
    }

    string imgPath1 = argv[1];
    Mat img1 = imread(imgPath1);
    if(!img1.data){
        cout << "Read img failed :" << imgPath1 << endl;
        return -1;
    }

    string imgPath2 =  argv[2];
    Mat img2 = imread(imgPath2);
    if(!img2.data){
        cout << "Read img failed :" << imgPath1 << endl;
        return -1;
    }

    vector<int> h1, h2;
    aHash(img1, h1);
    aHash(img2, h2);
    cout<< "hashCode1:"; 
    for(int i:h1) cout << i; cout << endl;
    cout<< "hashCode2:"; 
    for(int i:h2) cout << i; cout << endl;
    cout << "aHash difference: " << hammingDistance(h1, h2) << endl;

    h1.clear();
    h2.clear();
    dHash(img1, h1);
    dHash(img2, h2);
    cout<< "hashCode1:"; 
    for(int i:h1) cout << i; cout << endl;
    cout<< "hashCode2:"; 
    for(int i:h2) cout << i; cout << endl;
    cout << "dHash difference: " << hammingDistance(h1, h2) << endl;


    h1.clear();
    h2.clear();
    pHash(img1, h1);
    pHash(img2, h2);
    cout<< "hashCode1:"; 
    for(int i:h1) cout << i; cout << endl;
    cout<< "hashCode2:"; 
    for(int i:h2) cout << i; cout << endl;
    cout << "pHash difference: " << hammingDistance(h1, h2) << endl;
}