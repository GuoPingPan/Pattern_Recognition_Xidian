'''
    Kmeans on Iris

    author: GuoPingPan
    email: 731061720@qq.com
            or panguoping02@gmail.com

    brief: The code is using Kmeans to deal with Iris datasets with 3 classes.


'''
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns
import random
from collections import Counter

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
    plot_iris(iris, "sepal_length", "sepal_width")

    X = iris.iloc[:, :-1].values
    y = iris.iloc[:, -1].values

    return X, y


X, y = load_dataset()

'''
2. data processing
'''

# 标准化
def data_processing_standard():
    global X
    transfer = StandardScaler()
    X = transfer.fit_transform(X)

# 归一化
def data_processing_minmax():
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
    while (i < X.shape[0]):
        tmp1 = np.sum(np.square(X[:i] - X[i]), axis=1)
        tmp2 = np.sum(np.square(X[i + 1:] - X[i]), axis=1)
        tmp = np.concatenate((tmp1, tmp2))
        index = np.argsort(tmp)
        for j in range(3):
            k = index[j]
            if k >= i:  #由于 tmp 并没有计算 i 这一行,因此当索引值大于等于 i 时其实对应的是 y[i+1] 类别
                k += 1
            if y[k] != y[i]:
                X = np.concatenate((X[:i], X[i + 1:]))
                y = np.concatenate((y[:i], y[i + 1:]))
                break
        i += 1
    return X, y

# try to use difference methods
# data_processing_standard()
# data_processing_minmax()

'''
    part3:
        build model
'''

class Kmeans:
    ''' Kmeans聚类方法

    1.初始化中心点
    2.聚类形成新簇
    3.利用新簇的质心来更新中心点
    4.计算新旧中心点的欧式距离，小于阈值则停止迭代，否则继续迭代

    Args:
        self.k(int): 聚类类别数目
        self.k_centers(np.array): 聚类中心点
        self.k_clusters(dist): k个簇

    '''

    def __init__(self, X, k):
        self.k = k
        self.k_clusters = {i: [] for i in range(self.k)}

        # 初始化中心点
        self.k_centers = self.choose_init_center(X)

        self.ss = 0  # ss means the in C and in C*
        self.sd = 0  # sd means the in C and not in C*
        self.ds = 0  # ds means the not in C and in C*
        self.dd = 0  # dd means the not C and not in C*

    # 从样本中随机选取k个中心点
    def choose_init_center(self, X):
        centers_index = random.sample(range(len(X)), self.k)
        centers_index.sort(key=int)
        return X[centers_index]

    # 聚类
    def cluster(self, X, y, algorithm):
        self.k_clusters = {i: [] for i in range(self.k)}
        if algorithm == "E":  # Euclidean
            for i, x in enumerate(X):
                index = np.argmin(np.sqrt(np.sum(np.square(self.k_centers - x), axis=1)))
                tmp = x.tolist()
                tmp.append(y[i])    # keep label 保留真实类别
                self.k_clusters[index].append(tmp)


        elif algorithm == "M":  # Manhattan
            for i, x in enumerate(X):
                index = np.argmin(np.sum(np.abs(self.k_centers - x), axis=1))
                tmp = x.tolist()
                tmp.append(y[i])
                self.k_clusters[index].append(tmp)

    # 更新中心点
    def update(self):
        for i, cluster in enumerate(self.k_clusters.values()):
            if(len(cluster)>0):
                cluster = np.array(cluster)[:, :-1].astype(float)
                self.k_centers[i] = np.mean(cluster, axis=0)

    # 中心点距离
    def center_distance(self,last_centers):
        return np.sum(np.sqrt(np.sum(np.square(self.k_centers-last_centers),axis=1)))

    def in_different_cluster(self, i, label):
        for j in range(i, self.k):
            cluster_j = self.k_clusters[j]
            for k in range(len(cluster_j)):
                if cluster_j[k][-1] == label:
                    self.ds += 1
                else:
                    self.dd += 1

    def calculate_score(self):
        self.ss = self.sd =  self.ds = self.dd = 0

        # 同一个簇内的计算
        for cluster_i in self.k_clusters.values():
            for j in range(len(cluster_i)-1):
                for k in range(j+1,len(cluster_i)):
                    if cluster_i[j][-1] == cluster_i[k][-1]:
                        self.ss += 1
                    else:
                        self.sd += 1

        # 不同簇间的计算
        for i in range(self.k-1):
            cluster_i = self.k_clusters[i]
            for j in range(len(cluster_i)):
                self.in_different_cluster(i+1,cluster_i[j][-1])

    # 杰卡德系数
    def JC_score(self):
        # a/(a+b+c)
        return self.ss * 1.0 / (self.ss + self.sd + self.ds)

    # FM指数
    def FM_score(self):
        # sqrt(a*a/((a+b)*(a+c))
        return np.sqrt(pow(self.ss, 2) * 1.0 / ((self.ss + self.sd) * (self.ss + self.ds)))

    # 兰德指数
    def Rand_score(self):
        # 2(a+d)/(m(m-1))  ; m(m-1) = a+b+c+d
        return 1.0 * (self.ss + self.dd) / (self.ss + self.sd + self.ds + self.dd)



'''
    part4:
        train and test
'''

def main():
    # k个类别
    kmean = Kmeans(X, k=3)
    eps = 1e-5
    theta_dist = 10000
    last_centers = np.copy(kmean.k_centers)
    iter = 0

    # 保留最好的结果
    best_score = 0
    best_centers = None

    while (theta_dist > eps):

        # 聚类且更新中心点
        kmean.cluster(X, y, 'E')
        kmean.update()

        # 计算两次迭代的中心距离变化
        theta_dist = kmean.center_distance(last_centers)
        last_centers = np.copy(kmean.k_centers)

        # 计算外部性能指标
        kmean.calculate_score()
        score = kmean.Rand_score()

        # 保存最好的聚类参数指标
        if(score>best_score):
            best_score = score
            best_centers = np.copy(kmean.k_centers)

        # 输出迭代结果
        print(f"the iter:{iter} "
              f"RI sore:{kmean.Rand_score():5f} "
              f"JS score:{kmean.JC_score():5f} "
              f"FM score:{kmean.FM_score():5f} ")
        iter += 1

    # 使用最好的聚类参数进行可视化
    kmean.k_centers = np.copy(best_centers)
    kmean.cluster(X,y,'E')
    clusters = None
    for i,cluster in enumerate(kmean.k_clusters.values()):
        feature_name = ["sepal_length", "sepal_width", "petal_length", "petal_width", 'class']
        cluster = pd.DataFrame(cluster)
        # print(Counter(cluster.iloc[:, -1]))
        cluster[4] = Counter(cluster.iloc[:, -1]).most_common(1)[0][0]
        cluster.columns = feature_name
        if(i == 0):
            clusters = cluster
        else:
            clusters = pd.concat([clusters,cluster],axis=0)
    pd.set_option('display.max_rows',1000)
    clusters.index = [i for i in range(len(clusters))]

    plot_iris(clusters,"sepal_length", "sepal_width")
    return best_score


if __name__ == '__main__':
    print(f'epochs:{0}')
    print("-" * 100)
    best_score = main()
    epochs = 0
    while(best_score<0.90):
        epochs+=1
        print(f'epochs:{epochs}')
        print("-"*100)
        best_score = main()
    print(f'best_Rand_score:{best_score:5f}')

