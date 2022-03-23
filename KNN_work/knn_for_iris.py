'''
    KNN on Iris

    author: GuoPingPan
    email: 731061720@qq.com
            or panguoping02@gmail.com

    brief: The code is using KNN to deal with Iris datasets with 3 classes.

'''

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split,KFold
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns


'''
    part1:
        load iris dataset
'''

def plot_iris(data,col1,col2):
    '''
        Use seaborn to show the dataset
    '''

    sns.lmplot(x=col1,y=col2,data=data,hue="class",fit_reg=False)
    plt.xlabel(col1)
    plt.ylabel(col2)
    plt.title("iris_data plot with : "+col1+" "+col2)
    plt.show()

def load_dataset():
    '''
        Load Iris datasets
    '''

    feature_name = ["sepal_length","sepal_width","petal_length","petal_width",'class']
    class_name = ['Iris-setosa','Iris-versicolor','Iris-virginica']

    iris = pd.read_csv('../dataset/iris.data')
    iris.columns = feature_name
    # print(iris)
    plot_iris(iris,"petal_length","petal_width")
    # 经过数据分析后发现用 "petal_length","petal_width" 两个特征的区分度较为明显
    x = iris.iloc[:,2:-1].values
    y = iris.iloc[:,-1].values

    return train_test_split(x,y,train_size=0.7)

x_train,x_test,y_train,y_test = load_dataset()




'''
    part2:
        data processing
'''

# 标准化
def data_processing_standard():
    global x_train,x_test
    transfer = StandardScaler()
    x_train = transfer.fit_transform(x_train)
    x_test = transfer.transform(x_test)

# 归一化
def data_processing_minmax():
    global x_train,x_test
    transfer = MinMaxScaler()
    x_train = transfer.fit_transform(x_train)
    x_test = transfer.transform(x_test)


def data_pruning(X,y):
    '''
        Data Pruning

        reason:
         after data analysis, discover there are some noises around the boundary

        :param threshold
            if the num of nearby elements without same class is bigger than threshold,
            then abandon it.
    '''
    i = 0
    while(i<X.shape[0]):
        tmp1 = np.sum(np.square(X[:i] - X[i]),axis=1)
        tmp2 = np.sum(np.square(X[i+1:] - X[i]),axis=1)
        tmp = np.concatenate((tmp1,tmp2))
        index = np.argsort(tmp)
        for j in range(3):
            k = index[j]
            if k >= i:  #由于 tmp 并没有计算 i 这一行,因此当索引值大于等于 i 时其实对应的是 y[i+1] 类别
                k += 1
            if y[k] != y[i]:
                X = np.concatenate((X[:i],X[i+1:]))
                y = np.concatenate((y[:i],y[i+1:]))
                break
        i += 1
    return X,y

'''
    这里相对于是对所有的数据进行剪辑，本应该写在分开训练集和测试集之前
    不然会产生一种误解就是不允许测试集中有噪声出现
'''

# 剪辑前数目
# print(x_train.shape)
# print(y_train.shape)
# print(x_test.shape)
# print(y_test.shape)

x_train,y_train = data_pruning(x_train,y_train)
x_test,y_test = data_pruning(x_test,y_test)


''' show the data after data_pruning '''
xx = np.concatenate((x_train,x_test))
yy = np.concatenate((y_train,y_test))
show = pd.DataFrame({"petal_length": xx[:,0],"petal_width": xx[:,1],"class": yy})
plot_iris(show, "petal_length", "petal_width")

# 剪辑后数目
# print(x_train.shape)
# print(y_train.shape)
# print(x_test.shape)
# print(y_test.shape)

# why will the accuary be better without data processing?
# 以上为使用两个特征
# 当我选择使用四个特征时，使用了归一化或标准化后，准确率反而下降，现在尚未分析出是什么问题
# data_processing_standard()
# data_processing_minmax()




'''
    part3:
        build model
'''

class Knn:
    def __init__(self,neighbour):
        self.neighbour = neighbour
    def train(self,X_train,y_train):
        self.X = X_train
        self.y = y_train
    def predict(self,X_test,algorithm):
        self.y_pre = []
        if algorithm == "E":    #Euclidean
            for x in X_test:
                index = np.argsort(np.sqrt(np.sum(np.square(self.X - x),axis=1)))[:self.neighbour]
                # print(self.y[index])
                # print(self.count_max(self.y[index]))
                self.y_pre.append(self.count_max(self.y[index]))
            self.y_pre = np.array(self.y_pre)

        elif algorithm == "M":   #Manhattan
            for x in X_test:
                index = np.argsort(np.sum(np.abs(self.X - x),axis=1))[:self.neighbour]
                self.y_pre.append(self.count_max(self.y[index]))
                # self.y_pre.append(np.argmax(np.bincount(self.y[index])))
            self.y_pre = np.array(self.y_pre)

    def score(self,y_test):
        score = 0
        score = np.sum(self.y_pre == y_test)*1.0/self.y_pre.shape[0]
        return score

    def count_max(self,data):
        class_dict = dict(Counter(data))
        predict = None
        max_time = 0
        for key,value in class_dict.items():
            if value > max_time:
                predict = key
                max_time = value
        return predict

'''
    part4
        train and test
'''

def main():
    # print(x_train.shape)
    # print(y_train.shape)

    # k个neighbor
    k_num = np.arange(1,9,2)

    # 记录最好的参数与分数
    best_X = None
    best_Y = None
    best_Score =0

    # 对于每个k的对应预测准确率
    acc_for_k = []

    # 10折交叉验证
    n_splits = 10
    kfold = KFold(n_splits=n_splits,shuffle=True)

    for epoach,k in enumerate(k_num):
        classifer = Knn(k)
        # for epoach in range(epoachs):
        print("-"*100)
        print(f"epoch: {epoach}")
        print("-"*100)
        for train_index,test_index in kfold.split(x_train):
            X_train,X_test = x_train[train_index],x_train[test_index]
            Y_train,Y_test = y_train[train_index],y_train[test_index]
            # print(Y_train.shape)
            classifer.train(X_train,Y_train)
            classifer.predict(X_test,'M')
            score = classifer.score(Y_test)
            if score > best_Score:
                best_Score = score
                best_X = classifer.X
                best_Y = classifer.y
            print(f"score: {score} best_Score: {best_Score} k:{k}")

        classifer.train(best_X,best_Y)
        classifer.predict(x_test,'M')
        score = classifer.score(y_test)

        acc_for_k.append(score)
        print(f"the score of prediction on test dataset : {score}")

    plt.plot(k_num,acc_for_k)
    plt.scatter(k_num,acc_for_k,marker='o')
    for k,a in zip(k_num,acc_for_k):
        plt.text(k,a-0.01,('k:'+str(k),'acc:'+str(round(a,4))))
    plt.xlabel('k')
    plt.ylabel('acc')
    plt.ylim(0.9,1)
    plt.show()

main()