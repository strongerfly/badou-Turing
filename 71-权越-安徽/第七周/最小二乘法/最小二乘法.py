import pandas as pd
import math
sales=pd.read_csv('train_data.csv', sep='\s*,\s*', engine='python')  #读取CSV
X=sales['X'].values    #存csv的第一列
Y=sales['Y'].values    #存csv的第二列

x_mean=sales['X'].mean()
y_mean=sales['Y'].mean()
n=4
s1=0
s2=0

for i in range(n):
    s1=s1+(X[i]-x_mean)*(Y[i]-y_mean)
    s2=s2+math.pow((X[i]-x_mean),2)
w=s1/s2
b=y_mean-w*x_mean

print("Coeff: {} Intercept: {}".format(w, b))