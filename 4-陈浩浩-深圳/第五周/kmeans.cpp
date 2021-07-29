#include<iostream>
#include<vector>
#include<ctime>
#include<cmath>
#include<cfloat>
using namespace std;


double calDistance(vector<double> a, vector<double> b){
    if(a.size()!=b.size()) cerr<< "Dimensions of a and b must be the same!!!" << endl;
    double distance = 0;
    for(int i=0; i<a.size(); i++){
        distance += pow(a[i]-b[i], 2);
    }
    return sqrt(distance);
}


void kmeans(vector<vector<double>> data, int k, int maxIter, vector<vector<double>>& centroids, vector<int>& labels){
    if(data.size()==0) {
        cerr << "data is Empty!!!";
        return; 
        }
    int n = data.size(), dim=data[0].size();

    //k个聚类中心，随机初始化
    for(int i=0; i<k; i++){
        int index = rand()%n;
        centroids[i] = data[index];
    }

    vector<vector<double>> newCentroids(k, vector<double>(dim, 0));
    vector<int> count(k, 0);
    for(int it=0; it<maxIter; it++){

        for(int i=0; i<n; i++){
            double minDistance = DBL_MAX;
            for(int j=0; j<k; j++){
                double dis = calDistance(centroids[j], data[i]);
                if(dis<minDistance){
                    minDistance = dis;
                    labels[i] = j;   // 标记每个元素所属类别
                }
            }
            count[labels[i]] += 1;    //记录每个类别中心的样本个数
            for(int m=0; m<dim; m++){
                newCentroids[labels[i]][m] += data[i][m];  //累加每个类别中心样本的值
            }
        }

        // 计算新的聚类中心，并更新原始聚类中心
        for(int j=0; j<k; j++){
            for(int m=0; m<dim; m++){
                if(count[j]>0) newCentroids[j][m] /= count[j];
            }
        }

        centroids.assign(newCentroids.begin(), newCentroids.end());
        newCentroids.assign(k, vector<double>(dim, 0));
        count.assign(k, 0);
    }
}

int main(int argc, char* argv[]){
    int n = 200;  // 假设共200条数据
    int dim = 2;  //假设每条数据维度为2
    vector<vector<double>> samples(n, vector<double>(dim, 0));

    int seed = (int)time(0);
    srand(seed); //设置随机数种子
    for(int i=0; i<samples.size(); i++){
        for(int j=0; j<samples[i].size(); j++){
            samples[i][j] = rand()%1000;   //产生0-1000范围内的随机数
        }
    }

    cout <<"**********centroids:********"<<endl;
    int k = 4, maxIter=100;
    vector<vector<double>> centroids(k, vector<double>(dim, 0)); 
    vector<int> labels(n, 0);
    kmeans(samples, k, maxIter, centroids, labels);
    for(int i=0; i<centroids.size(); i++){
        for(int j=0; j<centroids[i].size(); j++){
        
            cout << centroids[i][j] <<",";
        }
        cout <<endl;
    }
    
    return 0;    
}