#include<iostream>
#include<opencv2/opencv.hpp>
#include<vector>
#include<set>

using namespace std;


bool ransacFitLine(vector<cv::Point2d>& points, cv::Vec4f& bestParam, vector<cv::Point2d>& inlinerPoints, int initInlinerNums, int iter, double thresh, int EndInlinerNums){
/*
    @param points:  所有的样本点.
    @param bestParam:  拟合最好的直线方程
    @param inlinerPoints:  拟合最好时的内点集合
    @param initInlinerNums： 用来拟合直线的初始化内点个数
    @param iter： 最大迭代次数
    @param thresh： 阈值，用来判断某个点是否在拟合的直线上
    @param EndInlinerNums： 阈值，内点个数超过阈值时停止迭代
 */

    set<int> randomIndex;
    vector<cv::Point2d> randomPoints;
    vector<cv::Point2d> tempInliners;
    int n = points.size();
    double minErr=DBL_MAX;
    bool flag = false;

    while(iter>=0){

        randomIndex.clear();
        randomPoints.clear();
        tempInliners.clear();

        // 随机选取initInlinerNums个点，拟合直线，其他点作为测试点
        for(int i=0; i<initInlinerNums; i++){
            int index = rand()%n;
            randomIndex.insert(index);
            randomPoints.push_back(points[index]);
        }

        // 采用随机点拟合直线
        cv::Vec4f lineParam;
        cv::fitLine(randomPoints, lineParam, cv::DIST_L2, 0, 0.01, 0.01);
        double linek = lineParam[1]/lineParam[0];
        
        // 计算测试点是否也在拟合直线上，并统计误差, 在直线上的点当作inliner
        double sumErr = 0;
        for(int i=0; i<n; i++){
            if(randomIndex.find(i)==randomIndex.end()){
                double err = linek*(points[i].x - lineParam[2]) + lineParam[3] - points[i].y;
                err = err*err; // 差值的平方作为误差
                if (err < thresh){
                    tempInliners.push_back(points[i]);
                    sumErr += err;
                }
            }
        }
        sumErr = sumErr/(n-initInlinerNums);  // 误差平方和的平均值
        
        // 当前inliner内点个数超过阈值时
        if(tempInliners.size()>EndInlinerNums){
            if(minErr>sumErr){  // 记录最小的平均误差，
                minErr = sumErr;
                bestParam = lineParam;
                inlinerPoints.assign(tempInliners.begin(), tempInliners.end()); 
                inlinerPoints.insert(inlinerPoints.end(), randomPoints.begin(), randomPoints.end());  // 合并所有的内点
                flag = true;
            }
        }
        iter--;
    }
    if(!flag) cout << "did't meet fit acceptance criteria" << endl;

    return flag;
}

int main(int argc, char* argv[]){
    cv::RNG rng(200);//seed为200

    vector<cv::Point2d> points;

    // 生成500个随机样本点
    int nSamples = 500;   // 生成样本数据个数
    double k = 1 + rng.gaussian(1);  // 0-2之间的随机斜率
    cout << k << endl;
    for(int i=0; i<nSamples; i++){
        double x = rng.uniform(40, 600);
        double y = k*x + rng.uniform(-30, 30);  // 加入随机噪声
        x += rng.uniform(-30, 30);             // 加入随机噪声
        points.push_back(cv::Point2d(x, y));
    }

    // 加入outlier
    for(int j=0; j< 100; j++){
        double x = rng.uniform(200, 600);
        double y = rng.uniform(10, 300);
        points.push_back(cv::Point2d(x, y));
    }


    cv::Mat img = cv::Mat(720, 720, CV_8UC3, cv::Scalar(255,255,255));

    //绘制point
    for(int i=0; i<points.size(); i++){
        cv::circle(img, points[i], 3, cv::Scalar(0, 0, 0), 2);
    }
    
    
     // 最小二乘法
    if(1){
        cv::Vec4f lineParam;
        fitLine(points, lineParam, cv::DIST_L2, 0.0, 0.01, 0.01);
        //获取点斜式的点和斜率, y=k(x-x0) + y0
        cv::Point point0;
        point0.x = lineParam[2];
        point0.y = lineParam[3];
        double linek = lineParam[1] / lineParam[0];
        

        //绘制最小二乘法拟合的直线
        cv::Point2d point1, point2;
        point1.x=10;
        point1.y = linek*(point1.x-point0.x) + point0.y;
        point2.x = 600;
        point2.y = linek*(point2.x-point0.x) + point0.y;
        cv::line(img, point1, point2, cv::Scalar(0, 255, 0), 4);
    }
    

    // ransac拟合直线
    if(1){   // 为了避免命名冲突，所以放在if局部范围内
        cv::Vec4f lineParam;
        vector<cv::Point2d> inlinerPoints;
        bool ret = ransacFitLine(points, lineParam, inlinerPoints, 50, 1000, 7e3, 300);
        if(ret){
            cout << ret << endl;
            cout << lineParam << endl;
            cout << inlinerPoints.size() << endl;
            
            //获取点斜式的点和斜率, y=k(x-x0) + y0
            cv::Point point0;
            point0.x = lineParam[2];
            point0.y = lineParam[3];
            double linek = lineParam[1] / lineParam[0];
            

            //绘制拟合的直线
            cv::Point2d point1, point2;
            point1.x=10;
            point1.y = linek*(point1.x-point0.x) + point0.y;
            point2.x = 600;
            point2.y = linek*(point2.x-point0.x) + point0.y;
            cv::line(img, point1, point2, cv::Scalar(0, 0, 255), 4);
        }else{
            cout << "Fitting failed, try to change parameters for ransac!!!" << endl;
        }
        
    }

    cv::imshow("img", img);
    cv::waitKey(0);
    cv::destroyAllWindows();

    return 0;
}