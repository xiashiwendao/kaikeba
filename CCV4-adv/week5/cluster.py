
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
#from sklearn import datasets
from sklearn.datasets import load_iris
from sklearn.cluster import DBSCAN
from sklearn.metrics import mean_squared_error
from sklearn.metrics import average_precision_score as aps
from sklearn.metrics import cohen_kappa_score as acs

def kmean_plus(X, y):
    estimator = KMeans(n_clusters=3, init='k-means++')#构造聚类器
    estimator.fit(X)#聚类
    label_pred = estimator.labels_ #获取聚类标签
    # print(label_pred)
    x0 = X[label_pred==0]
    x1 = X[label_pred==1]
    x2 = X[label_pred==2]
    plt.scatter(x=x0[:,0], y=x0[:, 1], c='red', marker='o')
    plt.scatter(x=x1[:,0], y=x1[:, 1], c='green', marker='o')
    plt.scatter(x=x0[:,0], y=x0[:, 1], c='blue', marker='o')
    plt.xlabel("petal length")
    plt.ylabel("petal width")
    # plt.show()

    mse = mean_squared_error(y, label_pred)
    asp_val = acs(y, label_pred)

    # print("++++++++ kmean++ mse: ", mse, " ++++++++")
    print("++++++++ kmean++ aps: ", asp_val, " ++++++++")

    
def kmean(X, y):
    estimator = KMeans(n_clusters=3, init='random')#构造聚类器
    estimator.fit(X)#聚类
    label_pred = estimator.labels_ #获取聚类标签
    # print(label_pred)
    x0 = X[label_pred==0]
    x1 = X[label_pred==1]
    x2 = X[label_pred==2]
    plt.scatter(x=x0[:,0], y=x0[:, 1], c='red', marker='o')
    plt.scatter(x=x1[:,0], y=x1[:, 1], c='green', marker='o')
    plt.scatter(x=x0[:,0], y=x0[:, 1], c='blue', marker='o')
    plt.xlabel("petal length")
    plt.ylabel("petal width")
    # plt.show()

    mse = mean_squared_error(y, label_pred)
    asp_val = acs(y, label_pred)

    # print("++++++++ kmean mse: ", mse, " ++++++++")
    print("++++++++ kmean aps: ", asp_val, " ++++++++")

def dbscan(X, y):
    label_pred = DBSCAN(eps = 0.7, min_samples = 10).fit_predict(X)
    # plt.scatter(X[:, 0], X[:, 1], c=label_pred)
    # plt.show()
    set(label_pred)
    mse = mean_squared_error(y, label_pred)
    asp_val = acs(y, label_pred)
    # print("++++++++ DBSCAN mse: ", mse, " ++++++++")
    print("++++++++ DBSCAN aps: ", asp_val, " ++++++++")

if __name__ == "__main__":
    
    iris = load_iris()
    X = iris.data[:]
    # print(X)
    y = iris.target[:]
    plt.scatter(X[:,0], X[:,1], c='red', marker='o', label='see')
    plt.xlabel('petal length')
    plt.ylabel('petal width')
    
    kmean(X, y)
    kmean_plus(X, y)
    dbscan(X, y)



