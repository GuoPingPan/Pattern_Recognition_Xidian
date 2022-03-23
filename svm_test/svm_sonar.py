'''
    SVM on Sonar

    author: GuoPingPan
    email: 731061720@qq.com
            or panguoping02@gmail.com

    brief: The code is using SVM to deal with Sonar datasets with 2 classes.
'''
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import random
from collections import Counter
from sklearn import svm
from sklearn.model_selection import GridSearchCV,train_test_split
from sklearn.decomposition import PCA

'''
    part1:
        load iris dataset
'''
# 加载数据
def load_dataset():

    sonar = pd.read_csv('../dataset/sonar.all-data.csv', header=None)
    print(f'\nsome data of sonar:\n{sonar.head()}')
    print(f'\nthe shape of sonar dataset:{sonar.shape}')
    return train_test_split(sonar.iloc[:,:-1],sonar.iloc[:,-1],test_size=0.3,random_state=10)

x_train,x_test,y_train,y_test = load_dataset()

'''
    part2:
        data processing
'''

# 标准化
transfer = None
def data_processing_standard():
    global x_train,x_test,transfer
    transfer = StandardScaler()
    x_train = transfer.fit_transform(x_train)#这里标准化并不一定是均值为0
    x_test = transfer.transform(x_test)
# 归一化
def data_processing_minmax():
    global x_train,x_test,transfer
    transfer = MinMaxScaler()
    x_train = transfer.fit_transform(x_train)
    x_test = transfer.transform(x_test)

def pca_down_dimension(n_components):
    global x_train_pca, x_test_pca
    pca = PCA(n_components=n_components)
    x_train_pca = pca.fit_transform(x_train)
    x_test_pca  = pca.transform(x_test)
    # print('\npac ratio list:\n',pca.explained_variance_ratio_)
    print('\nsum of pca ratio:',np.sum(pca.explained_variance_ratio_))
    print("x_train's features shape:",x_train_pca.shape[1])
    return (n_components,np.sum(pca.explained_variance_ratio_))

data_processing_standard()

'''
    part3:
        main
'''
x_train_pca = None
x_test_pca =  None

''' ********************** search the best model ********************** '''
def search(x_train,x_test,y_train,y_test):
    model = svm.SVC()
    param_dict = {'C':[pow(2,i) for i in range(-5,7,2)],'kernel':['rbf','linear','poly']}
    estimator = GridSearchCV(estimator=model,param_grid=param_dict,cv=7)
    estimator.fit(x_train,y_train)
    print(f"the best param:{estimator.best_params_}")
    print(f"the best score:{estimator.best_score_}")
    print(f"the best estimator:{estimator.best_estimator_}")
    print(f"the best best_index:{estimator.best_index_}")

    score = estimator.score(x_test,y_test)
    print(f"the testing score:{score}")
    return (estimator.best_score_,estimator.best_params_)

def search_best_param(param_list):
    best_score=0
    best_param=None
    for each in param_list:
        pca_param,model_param = each
        if model_param[0]>best_score:
            best_score = model_param[0]
            best_param = each
    return best_param

def get_best_model():
    param_list = []
    epoach = 0
    for n in range(30,50,2):
        print(f'************************ {epoach} ************************')
        epoach += 1
        pca_param = pca_down_dimension(n)
        # print(x_train_pca.shape)
        model_param = search(x_train_pca,x_test_pca,y_train,y_test)
        param_list.append((pca_param,model_param))

    print(f'\nparam_list:\n{param_list}')
    best_param = search_best_param(param_list)
    print(f"\nbest_param:")
    print(f"\n\tpca_param:{best_param[0]}")
    print(f"\n\tmodel_param:{best_param[1]}")

# get_best_model()#(0.882312925170068, {'C': 8, 'kernel': 'rbf'})


''' ********************** search the best pca params ********************** '''

def best_model(x_train,x_test,y_train,y_test):
    model = svm.SVC(C=8,kernel='rbf')
    model.fit(x_train,y_train)
    score = model.score(x_test,y_test)
    return score

def get_best_pac():
    best_score = 0
    best_param = None
    for i in range(30,61):
        pca_param = pca_down_dimension(i)
        score = best_model(x_train_pca,x_test_pca,y_train,y_test)
        if(score > best_score):
            best_score = score
            best_param = (pca_param,score)
        print(f'score:{score}')
    print(f'\nthe best pca params and score:{best_param}')

get_best_pac()#((35, 0.9765959968023169), 0.8412698412698413)


import joblib
from sklearn.model_selection import cross_val_score

def save_best_model():
    pca = PCA(n_components=35)
    x_train_ = pca.fit_transform(x_train)
    x_test_ = pca.transform(x_test)
    model = svm.SVC(C=8,kernel='rbf')
    cval  = cross_val_score(model,x_train_,y_train,cv=7,verbose=50)
    print(np.mean(cval))

    # x = np.concatenate((x_train_,x_test_))
    # y = np.concatenate((y_train,y_test))
    model.fit(x_train_,y_train)
    score = model.score(x_test_,y_test)
    print(score)

    joblib.dump(pca, 'model/svm_sonar_pca.joblib')
    joblib.dump(model, 'model/svm_sonar.joblib')
    print('model successfully save!')

# save_best_model()

# 这里有一个问题，如果我是用训练集去训练并且保留了pca降维矩阵，那也就是说样本数据中，只有30%是没有见过的，
# 然后在随机打乱时，抽到训练集的结果出来一定是判断准确的，那准确率将会以保存模型的分数为下界，这算不算过拟合呢？
# SVM防止过拟合的方式就是增大C值
# 这里又出现了一个问题：如果你GridCVSearch中选择分数最高的param是对验证集最好的分数，不是训练集
# 交叉验证是为了在训练过程中得到最好的参数（尤其是数据样本较少时），而训练集是来验证泛化能力的

# 待解决问题
# 1.是否过拟合？
# 2.交叉验证能抑制过拟合吗？能

def load_best_model():
    model = joblib.load('model/svm_sonar.joblib')
    pca = joblib.load('model/svm_sonar_pca.joblib')
    # pca_down_dimension(42)
    # x = np.concatenate((x_train,x_test))
    # y = np.concatenate((y_train,y_test))
    x_ = pca.transform(x_test)
    score = model.score(x_,y_test)
    print(score)

# load_best_model()
