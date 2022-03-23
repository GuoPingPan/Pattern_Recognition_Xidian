'''
    Fisher on Iris

    author: GuoPingPan
    email: 731061720@qq.com
            or panguoping02@gmail.com

    brief: The code is using Fisher to deal with Iris datasets with 3 classes.

'''


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split,KFold
import sklearn


'''
    part1:
        load iris dataset
'''
feature_name = ["sepal_length","sepal_width","petal_length","petal_width",'class']
class_name = ['Iris-setosa','Iris-versicolor','Iris-virginica']

iris = pd.read_csv('../dataset/iris.data')
iris.columns = feature_name
# print(iris)


'''
    part2:
        coding by the theory of Fisher.
        
        1.There is a 3 classes problem, so we can split into 3 classes.
          And make 3 projected vector and 3 bias.
         :param [w12 w13 w23] 3 projected vector
         :param [bias_12 bias_13 bias_23] 3 bias
         :param [n1,n2,n3] num of samples of each class

        2. We need to compute 3 means and 3 covariances
         :param [m1,m2,m3] means
         :param [Sw1,Sw2,Sw3] covariances 
         
        3. Compute the w,b by Sw and m
         theorem: w12 = (Sw1 + Sw2).inv * (m1 - m2) 
                  bias_12 =  ( n1*w12.dot(m1) + n2*w12.dot(m2) )/(n1+n2)
        
        4. Classify by the 3 discriminator output
          :param [g12,g13,g23] 3 output by discriminator
          if x is input: 
            g12 = w12.dot(x) + bias12,g13 = w13.dot(x) + bias13,g23 = w23.dot(x) + bias23
           if g12 > 0 and g13 > 0 and y_test[i] == class_name[0]:
                N1 += 1     #classify x is class 1
            elif g12 < 0 and g23 > 0 and y_test[i] == class_name[1]:
                N2 += 1
            elif g13 < 0 and g23 < 0 and y_test[i] == class_name[2]:
                N3 += 1 
'''

def get_mean_and_num(x,y):

    '''
        Compute mean and num

    :param[in]  x:features
    :param[in]  y:label
    :return:    m1,m2,m3,n1,n2,n3
    '''

    m1 = np.zeros(4)
    m2 = np.zeros(4)
    m3 = np.zeros(4)
    n1,n2,n3 = 0,0,0
    for i,each in enumerate(y):
        if each == class_name[0]:
            m1 += x[i]
            n1 += 1
        elif each == class_name[1]:
            m2 += x[i]
            n2 +=1
        elif each == class_name[2]:
            m3 += x[i]
            n3 +=1
    return (m1/n1).reshape(4,1),(m2/n2).reshape(4,1),(m3/n3).reshape(4,1),n1,n2,n3

def getSw(m):
    '''
        Compute Sw
    :param   m:mean
    :return: Sw
    '''
    return m.dot(m.T)

def get_w_and_bias(Sw,m1,m2,n1,n2):
    '''
        Compute w and bias
    '''

    w = np.linalg.pinv(Sw).dot(m1 - m2)
    bias = (n1*w.T.dot(m1) + n2*w.T.dot(m2))/(n1+n2)
    return w,bias.squeeze()

'''
--------------            train             --------------
'''
epochs = 200
def main():

    # load data
    y = iris.iloc[:,-1].values
    x = iris.iloc[:,:-1].values

    # record the best model
    acc_total = 0
    w_best = np.array((4,3))
    bias_best = np.array((3,1))
    acc_best = 0

    for epoch in range(epochs):
        acc = 0

        # use 10 fold
        fold_num = 10
        kfold = KFold(n_splits=fold_num,shuffle=True)

        for x_train_index,x_test_index in kfold.split(x):

            # training
            x_train,x_test= x[x_train_index],x[x_test_index]
            y_train,y_test = y[x_train_index],y[x_test_index]
            m1,m2,m3,n1,n2,n3 = get_mean_and_num(x_train,y_train)
            # print(m1,m2,m3)
            Sw_1 = getSw(m1)
            Sw_2 = getSw(m2)
            Sw_3 = getSw(m3)
            # print(Sw_1)
            w_12,bias_12 = get_w_and_bias(Sw_1+Sw_2,m1,m2,n1,n2)
            w_13,bias_13 = get_w_and_bias(Sw_1+Sw_3,m1,m3,n1,n3)
            w_23,bias_23 = get_w_and_bias(Sw_2+Sw_3,m2,m3,n2,n3)

            # evaluating
            N1,N2,N3 = 0,0,0
            for i,x_ in enumerate(x_test):
                g12 = w_12.T.dot(x_.reshape(4,1)) + bias_12
                g13 = w_13.T.dot(x_.reshape(4,1)) + bias_13
                g23 = w_23.T.dot(x_.reshape(4,1)) + bias_23
                if g12 > 0 and g13 > 0 and y_test[i] == class_name[0]:
                    N1 += 1
                elif g12 < 0 and g23 > 0 and y_test[i] == class_name[1]:
                    N2 += 1
                elif g13 < 0 and g23 < 0 and y_test[i] == class_name[2]:
                    N3 += 1

        # record the best model
        acc = (N1+N2+N3)/len(x_test)
        if(acc>acc_best):
            w_best = np.hstack((w_12,w_13,w_23))
            bias_best = np.array([[bias_12],[bias_13],[bias_23]])
            acc_best = acc
        acc_total  += acc
        print(f"epoach: [{epoch:3d}/{epochs:3d}] acc: {acc:5f} acc_best: {acc_best:5f}")

    print("-"*20)
    print(" Final Result ")
    print(f"acc_best: {acc_best} \n"
          f"w_best:\n {w_best} \n"
          f"bias_best:\n {bias_best} \n"
          f"acc_average:\n {acc_total/epochs}")

if __name__ == '__main__':
    main()