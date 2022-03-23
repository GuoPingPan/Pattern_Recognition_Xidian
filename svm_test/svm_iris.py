'''
    SVM on Iris

    author: GuoPingPan
    email: 731061720@qq.com
            or panguoping02@gmail.com

    brief: The code is using SVM to deal with Iris datasets with 3 classes.
'''

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns
import random
from collections import Counter
from sklearn import svm
from sklearn.model_selection import GridSearchCV,train_test_split

'''
    part1:
        load iris dataset
'''

def plot_iris(data, col1, col2):
    '''
        Use seaborn to show the dataset
    '''
    plt.figure(figsize=(4,8))
    sns.set_style('white')
    sns.lmplot(x=col1, y=col2, data=data, hue="class", fit_reg=False,palette={'Iris-setosa':'teal','Iris-versicolor':'tomato', 'Iris-virginica':'palegreen'})
    plt.xlabel(col1)
    plt.ylabel(col2)
    # plt.ylim(0,3)
    # plt.xlim(0,8)
    plt.title("iris_data plot with : " + col1 + " " + col2)
    plt.show()


def load_dataset():
    '''
        Load Iris datasets
    '''
    feature_name = ["sepal_length", "sepal_width", "petal_length", "petal_width", 'class']
    class_name = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']

    iris = pd.read_csv('../dataset/iris.data',header=None)
    iris.columns = feature_name
    # plot_iris(iris, "sepal_length", "sepal_width")
    X = iris.iloc[:,:-1].values
    y = iris.iloc[:,-1].values
    y[y==class_name[0]] = 0
    y[y==class_name[1]] = 1
    y[y==class_name[2]] = 2
    y = y.astype(int)
    return X,y

X,y = load_dataset()
x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=22)

'''
    part2:
        data processing
'''

# 标准化
transfer = None
def data_processing_standard():
    global x_train,x_test,transfer
    transfer = StandardScaler()
    x_train = transfer.fit_transform(x_train)
    x_test = transfer.transform(x_test)
# 归一化
def data_processing_minmax():
    global x_train,x_test,transfer
    transfer = MinMaxScaler()
    x_train = transfer.fit_transform(x_train)
    x_test = transfer.transform(x_test)

# 进行标准化
# data_processing_standard()
data_processing_minmax()

'''
    part3:
        build model
'''

# method one
def method_one():
    model = svm.SVC()
    param_dict = {'C':[pow(2,i) for i in range(-5,7,2)],'kernel':['rbf','linear','poly','sigmoid']}
    estimator = GridSearchCV(estimator=model,param_grid=param_dict,cv=7,verbose=50)
    estimator.fit(x_train,y_train)
    print(f"the best param:{estimator.best_params_}")
    print(f"the best score:{estimator.best_score_}")
    print(f"the best estimator:{estimator.best_estimator_}")
    print(f"the best best_index:{estimator.best_index_}")

    score = estimator.score(x_test,y_test)
    print(f"predict score:{score}")

# method_one()


# method two actucally use libsvm
def method_two():
    global x_train,y_train,x_test,y_test,X
    clf1 = svm.SVC(kernel='linear',decision_function_shape='ovr',verbose=50)
    clf2 = svm.SVC(kernel='rbf',decision_function_shape='ovr',verbose=50)
    clf3 = svm.SVC(kernel='poly',decision_function_shape='ovr',verbose=50)

    clf1.fit(x_train,y_train)
    clf2.fit(x_train,y_train)
    clf3.fit(x_train,y_train)

    # training score
    print('************* Training Score *************')
    print(f'linear training score:{clf1.score(x_train,y_train)}')
    print(f'rbf training score:{clf2.score(x_train,y_train)}')
    print(f'poly training score:{clf3.score(x_train,y_train)}')

    # testing score
    print('************* Testing Score *************')
    print(f'linear testing score:{clf1.score(x_test,y_test)}')
    print(f'rbf testing score:{clf2.score(x_test,y_test)}')
    print(f'poly testing score:{clf3.score(x_test,y_test)}')

    import matplotlib as mpl
    import matplotlib.pyplot as plt

    ''' draw the result '''
    X = X[:,2:]
    x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=22)

    clf1 = svm.SVC(kernel='linear',decision_function_shape='ovr',verbose=50)
    clf2 = svm.SVC(kernel='rbf',decision_function_shape='ovr',verbose=50)
    clf3 = svm.SVC(kernel='poly',decision_function_shape='ovr',verbose=50)

    clf1.fit(x_train,y_train)
    clf2.fit(x_train,y_train)
    clf3.fit(x_train,y_train)

    x1_min,x1_max = X[:,0].min()-0.1,X[:,0].max()+0.1
    x2_min,x2_max = X[:,1].min()-0.1,X[:,1].max()+0.1
    x1,x2 = np.mgrid[x1_min:x1_max:100j,x2_min:x2_max:100j]
    grid_data = np.stack((x1.flat,x2.flat),axis=1)

    grid_predict1 = clf1.predict(grid_data).reshape(x1.shape)
    grid_predict2 = clf2.predict(grid_data).reshape(x1.shape)
    grid_predict3 = clf3.predict(grid_data).reshape(x1.shape)

    mpl.rcParams['font.sans-serif'] = [u'SimHei']
    mpl.rcParams['axes.unicode_minus'] = False
    cm_light = mpl.colors.ListedColormap(['#A0FFA0', '#FFA0A0', '#A0A0FF'])
    cm_dark = mpl.colors.ListedColormap(['g', 'r', 'b'])

    # choose a predictor, each one use two different features to train
    show_who = grid_predict3

    plt.pcolormesh(x1,x2,show_who,shading='auto',cmap=cm_light)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', s=50, cmap=cm_dark)
    plt.xlabel("sepal_length", fontsize=13)
    plt.ylabel("sepal_width", fontsize=13)
    plt.xlim(x1_min, x1_max)
    plt.ylim(x2_min, x2_max)
    plt.title(u'iris_classification', fontsize=15)
    plt.show()

method_two()