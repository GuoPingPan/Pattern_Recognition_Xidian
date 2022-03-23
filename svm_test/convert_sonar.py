import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np

''' 
    convert data to numpy
    the classes use label encoding
'''
class_name = ['R','M']
sonar = pd.read_csv('../dataset/sonar.all-data.csv')
data = sonar.values
data[data[:,-1]==class_name[0],-1] = 0
data[data[:,-1]==class_name[1],-1] = 1

x_train,x_test,y_train,y_test=train_test_split(data[:,:-1],data[:,-1],random_state=22,test_size=0.3)

print(x_train.shape)
''' convert training data '''
with open('./data/sonar.train','w') as f:
    for i,each in enumerate(x_train):
        features = [f' {i+1}:{each[i]}' for i in range(len(each))]
        str1 = f'{y_train[i]}'
        for feature in features:
            str1 += feature
        print(str1)
        f.write(str1+'\n')

''' convert testing data '''
with open('./data/sonar.test','w') as f:
    for i,each in enumerate(x_test):
        features = [f' {i+1}:{each[i]}' for i in range(len(each))]
        str1 = f'{y_test[i]}'
        for feature in features:
            str1 += feature
        print(str1)
        f.write(str1+'\n')