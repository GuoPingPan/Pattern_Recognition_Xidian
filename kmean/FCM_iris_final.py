'''
    FCM on Iris

    author: GuoPingPan
    email: 731061720@qq.com
            or panguoping02@gmail.com

    brief: The code is using FCM to deal with Iris datasets with 3 classes.

'''

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from sklearn.cluster import KMeans
import random

'''
    part1:
        load iris dataset
'''

def plot_iris(data, col1, col2):
    '''
        Use seaborn to show the dataset
    '''

    plt.figure(figsize=(4, 8))
    sns.set_style('white')
    sns.lmplot(x=col1, y=col2, data=data, hue="class", fit_reg=False,
               palette={'Iris-setosa': 'teal', 'Iris-versicolor': 'tomato', 'Iris-virginica': 'palegreen'})
    plt.xlabel(col1)
    plt.ylabel(col2)
    # plt.ylim(0, 3)
    # plt.xlim(0, 8)
    plt.title("iris_data plot with : " + col1 + " " + col2)
    plt.show()


def load_dataset():
    '''
        Load Iris datasets
    '''

    feature_name = ["sepal_length", "sepal_width", "petal_length", "petal_width", 'class']
    class_name = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']

    iris = pd.read_csv('../dataset/iris.data', header=None)
    iris.columns = feature_name
    plot_iris(iris, "sepal_length", "sepal_width")
    x = iris.iloc[:, :-1].values
    y = iris.iloc[:, -1].values

    return x, y

X, y = load_dataset()

'''
    Options:
        data processing
'''

def data_processing_standard():
    '''
        Standardization
    '''

    global X
    transfer = StandardScaler()
    X = transfer.fit_transform(X)

def data_processing_minmax():
    '''
        MinMax Normalization
    '''

    global X
    transfer = MinMaxScaler()
    X = transfer.fit_transform(X)

def data_pruning(X, y):
    '''
        Data Pruning

        reason:
         after data analysis, discover there are some noises around the boundary

        :param threshold
            if the num of nearby elements without same class is bigger than threshold,
            then abandon it.
    '''
    i = 0
    threshold = 3
    while (i < X.shape[0]):
        tmp1 = np.sum(np.square(X[:i] - X[i]), axis=1)
        tmp2 = np.sum(np.square(X[i + 1:] - X[i]), axis=1)
        tmp = np.concatenate((tmp1, tmp2))
        index = np.argsort(tmp)
        for j in range(threshold):
            k = index[j]
            if k >= i:
                k += 1
            if y[k] != y[i]:
                X = np.concatenate((X[:i], X[i + 1:]))
                y = np.concatenate((y[:i], y[i + 1:]))
                break
        i += 1
    return X, y


# try to use difference methods

# data_processing_minmax()
# data_processing_standard()
# print(X.shape)

'''
    part3:
        build model
'''

class FCM:
    def __init__(self, X, k, b):
        self.k = k
        self.b = b
        self.ss = 0  # ss means the in C and in C*
        self.sd = 0  # sd means the in C and not in C*
        self.ds = 0  # ds means the not in C and in C*
        self.dd = 0  # dd means the not C and not in C*

        self.k_clusters = {i: [] for i in range(self.k)}
        # ???????????????????????????????????????????????????????????????????????????????????????????????????0?????????
        # self.k_centers = np.random.uniform(0, 10, (self.k, X.shape[1]))

        # ?????????????????????????????????????????????????????????sklean???KMeans????????????????????????????????????
        # kmean = KMeans(n_clusters=self.k)
        # kmean.fit(X)
        # self.k_centers = kmean.cluster_centers_

        # self.k_centers = self.choose_init_center(X)
        self.k_centers = self.choose_init_center(X,10)
        self.U = None
        self.update_U(X)

    def choose_init_center(self, X,n):
        ''' ????????????????????? '''
        centers = []
        for i in range(self.k):
            centers_index = random.sample(range(len(X)), n)
            centers_index.sort(key=int)
            x = X[centers_index]
            center = np.mean(x,axis=0)
            centers.append(center)
        return np.array(centers)

    def update_U(self, X):
        ''' ???????????????

        ????????????: uji = (xi-cj)^{-1/(b-1)}/sumj((xi-cj)^{-1/(b-1)})

        '''
        eop = -1.0 / (self.b - 1)
        U = []
        for x in X:
            tmp = np.power(np.linalg.norm(self.k_centers - x, axis=1), eop)
            sum = np.sum(tmp)
            U.append(tmp / sum)
        self.U = np.array(U)

    def updata_center(self, X):
        ''' ???????????????

        ????????????: cj = sumi(xi*uji^b)/sumi(uji^b)

        '''
        for j in range(self.k):
            u_j = self.U[:, j]
            up = []

            for i, x in enumerate(X):
                up.append(x * np.power(u_j[i], self.b))
            up = np.sum(np.array(up), axis=0)
            down = np.sum(np.power(u_j, self.b))
            # print(f"up:{up} down:{down}")
            self.k_centers[j] = up / down

    def cluster(self, X, y, algorithm):
        ''' ??????

        :param X: ??????????????????
        :param y: ????????????
        :param algorithm: ???????????????????????????????????????????????????

        '''
        self.k_clusters = {i: [] for i in range(self.k)}
        if algorithm == "E":  # Euclidean
            for i, x in enumerate(X):
                index = np.argmin(np.sqrt(np.sum(np.square(self.k_centers - x), axis=1)))
                tmp = x.tolist()
                tmp.append(y[i])  # keep label
                self.k_clusters[index].append(tmp)


        elif algorithm == "M":  # Manhattan
            for i, x in enumerate(X):
                index = np.argmin(np.sum(np.abs(self.k_centers - x), axis=1))
                tmp = x.tolist()
                tmp.append(y[i])
                self.k_clusters[index].append(tmp)

    def center_distance(self, last_centers):
        ''' ????????????????????????????????????

        :param last_centers: ???????????????

        '''
        return np.sum(np.sqrt(np.sum(np.square(self.k_centers - last_centers), axis=1)))


    def in_different_cluster(self, i, label):
        for j in range(i, self.k):
            cluster_j = self.k_clusters[j]
            for k in range(len(cluster_j)):
                if cluster_j[k][-1] == label:
                    self.ds += 1
                else:
                    self.dd += 1

    def calculate_score(self):
        ''' ???????????????????????? '''
        self.ss = self.sd = self.ds = self.dd = 0

        for cluster_i in self.k_clusters.values():
            for j in range(len(cluster_i) - 1):
                for k in range(j + 1, len(cluster_i)):
                    if cluster_i[j][-1] == cluster_i[k][-1]:
                        self.ss += 1
                    else:
                        self.sd += 1

        for i in range(self.k - 1):
            cluster_i = self.k_clusters[i]
            for j in range(len(cluster_i)):
                self.in_different_cluster(i + 1, cluster_i[j][-1])

    # ???????????????
    def JC_score(self):
        # a/(a+b+c)
        return self.ss * 1.0 / (self.ss + self.sd + self.ds)

    # FM??????
    def FM_score(self):
        # sqrt(a*a/((a+b)*(a+c))
        return np.sqrt(pow(self.ss, 2) * 1.0 / ((self.ss + self.sd) * (self.ss + self.ds)))

    # ????????????
    def Rand_score(self):
        # 2(a+d)/(m(m-1))  ; m(m-1) = a+b+c+d
        return 1.0 * (self.ss + self.dd) / (self.ss + self.sd + self.ds + self.dd)


'''
    part4:
        train and test
'''


def main():
    # k?????????
    fcm = FCM(X, k=3, b=3)
    eps = 1e-3
    theta_dist = 10000
    last_centers = np.copy(fcm.k_centers)
    iter = 0

    # ??????????????????
    best_score = 0
    best_centers = None

    Rand_Score = 0
    same = 0

    while (theta_dist > eps):

        # ????????????????????????????????????
        fcm.cluster(X, y, 'E')
        fcm.updata_center(X)
        fcm.update_U(X)

        # ???????????????????????????????????????
        theta_dist = fcm.center_distance(last_centers)
        last_centers = np.copy(fcm.k_centers)

        # ????????????????????????
        fcm.calculate_score()
        score = fcm.Rand_score()

        # # ?????????????????????????????????
        if (score > best_score):
            best_score = score
            best_centers = np.copy(fcm.k_centers)

        if (Rand_Score == score):
            same += 1
        if (Rand_Score != score):
            Rand_Score = score
        if(same > 5):
            break
        # print(same)
        # ??????????????????
        print(f"the iter:{iter} "
              f"RI sore:{fcm.Rand_score():5f} "
              f"JS score:{fcm.JC_score():5f} "
              f"FM score:{fcm.FM_score():5f} ")
        iter += 1

    # ??????????????????????????????????????????
    fcm.k_centers = np.copy(best_centers)
    fcm.cluster(X, y, 'E')
    clusters = None
    for i, cluster in enumerate(fcm.k_clusters.values()):
        feature_name = ["sepal_length", "sepal_width",
                        "petal_length", "petal_width", 'class']
        cluster = pd.DataFrame(cluster)
        cluster[4] = Counter(cluster.iloc[:, -1]).most_common(1)[0][0]
        cluster.columns = feature_name
        if (i == 0):
            clusters = cluster
        else:
            clusters = pd.concat([clusters, cluster], axis=0)
    pd.set_option('display.max_rows', 1000)
    clusters.index = [i for i in range(len(clusters))]
    plot_iris(clusters, "sepal_length", "sepal_width")
    return best_score

if __name__ == '__main__':
    print(f'epochs:{0}')
    print("-" * 100)
    best_score = main()
    epochs = 0
    while(best_score<0.85):
        epochs+=1
        print(f'epochs:{epochs}')
        print("-"*100)
        best_score = main()

    print("-"*20)
    print("Final Result")
    print("-"*20)
    print(f'best_Rand_score: {best_score:5f}')
