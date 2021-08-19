import numpy as np
import scipy.special
class NetWork:
    def __init__(self,input,hidden,output,lr):
        #输入
        self.input=input
        # 隐藏
        self.hidden=hidden
        # 输出
        self.output=output
        # 学习率
        self.lr=lr

        self.wih=np.random.normal(0,pow(self.hidden,-0.5),(self.hidden,self.input))
        self.who=np.random.normal(0,pow(self.output,-0.5),(self.output,self.hidden))

        self.activation=lambda x:scipy.special.expit(x)

    def train(self,input_list,target_list):
        inputs=np.array(input_list).T
        targets=np.array(target_list).T

        # 计算输入经过隐藏层
        hidden_inputs=np.dot(self.wih,inputs)

        # 隐藏层经过激活的输出
        hidden_outputs=self.activation(hidden_inputs)

        #输出层接收经过激活后的隐藏层信号
        final_inputs=np.dot(self.who,hidden_outputs)

        # 输出层经过激活的输出
        final_outputs=self.activation(final_inputs)

        # 计算误差
        output_errors=targets-final_outputs


        hidden_errors = np.dot(self.who.T, output_errors * final_outputs * (1 - final_outputs))
        # 根据误差计算链路权重的更新量，然后把更新加到原来链路权重上
        self.who += self.lr * np.dot(np.expand_dims((output_errors * final_outputs * (1 - final_outputs)),1),
                                        np.expand_dims(hidden_outputs,1).T)
        self.wih += self.lr * np.dot(np.expand_dims((hidden_errors * hidden_outputs * (1 - hidden_outputs)),1),
                                        np.expand_dims(inputs,1).T)

    def query(self,inputs):

        hidden_inputs=np.dot(self.wih,inputs)

        hidden_outputs=self.activation(hidden_inputs)

        final_inputs=np.dot(self.who,hidden_outputs)

        final_output=self.activation(final_inputs)

        print(final_output)

        return final_output


inputs = 784
hidden = 200
output = 10

lr = 0.1
# 初始化网络
n = NetWork(inputs, hidden, output, lr)

train_file=open("dataset/mnist_train.csv","r")
train_list=train_file.readlines()
train_file.close()

epochs=5

for i in range(epochs):
    for record in train_list:
        all_values=record.split(',')
        inputs=(np.asfarray(all_values[1:]))/255.0* 0.99 + 0.01
        targets=np.zeros(output)+0.01
        targets[int(all_values[0])]=0.99

        n.train(inputs,targets)

test_file=open("dataset/mnist_test.csv")
test_list=test_file.readlines()
test_file.close()

scores=[]

for record in test_list:
    all_values=record.split(',')
    correct_number=int(all_values[0])
    print("该图片对应数字为：",correct_number)

    inputs=(np.asfarray(all_values[1:]))/255.0*0.99+0.01

    outputs=n.query(inputs)

    label=np.argmax(outputs)

    print("预测结果为：",label)

    if label==correct_number:
        scores.append(1)
    else:
        scores.append(0)
scores_array=np.asarray(scores)

print("正确率:",scores_array.sum()/scores_array.size)