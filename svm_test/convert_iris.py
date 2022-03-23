import pandas as pd
from sklearn.model_selection import train_test_split

''' 
    convert data to numpy
    the classes use label encoding
'''
class_name = ['Iris-setosa','Iris-versicolor','Iris-virginica']
iris = pd.read_csv('../dataset/iris.data')
data = iris.values
data[data[:,4]==class_name[0],4] = 0
data[data[:,4]==class_name[1],4] = 1
data[data[:,4]==class_name[2],4] = 2

x_train,x_test,y_train,y_test=train_test_split(data[:,:-1],data[:,-1],random_state=22,test_size=0.3)


''' convert training data '''
with open('./data/iris.train','w') as f:
    for i,each in enumerate(x_train):
        features = [f' {i+1}:{each[i]}' for i in range(len(each))]
        str1 = f'{y_train[i]}'
        for feature in features:
            str1 += feature
        print(str1)
        f.write(str1+'\n')

''' convert testing data '''
with open('./data/iris.test','w') as f:
    for i,each in enumerate(x_test):
        features = [f' {i+1}:{each[i]}' for i in range(len(each))]
        str1 = f'{y_test[i]}'
        for feature in features:
            str1 += feature
        print(str1)
        f.write(str1+'\n')