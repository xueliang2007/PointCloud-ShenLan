# 文件功能：实现 GMM 算法

import numpy as np
from numpy import *
import pylab
import random
import math
import copy

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from scipy.stats import multivariate_normal
plt.style.use('seaborn')


class GMM(object):
    def __init__(self, n_clusters, max_iter=50):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.W = None
        self.Mu = None
        self.Var = None
        self.Pi = None

    # 屏蔽开始
    # 更新W(后验概率, 隐含变量, 即每个数据点属于每一类的概率)
    def update_W(self, X, Mu, Var, Pi):
        n_points = len(X)
        pdfs = np.zeros((n_points, self.n_clusters))
        for i in range(self.n_clusters):
            # Multivariate normal probability density function
            pdfs[:, i] = Pi[i] * multivariate_normal.pdf(X, Mu[i], np.diag(Var[i]))
        W = pdfs / pdfs.sum(axis=1).reshape(-1, 1)
        return W

    # 更新pi（每一类的权重）
    def update_Pi(self, W):
        Pi = W.sum(axis=0) / W.sum()
        return Pi
        
    # 更新Mu（每一类的期望）
    def update_Mu(self, X, W):
        Mu = np.zeros((self.n_clusters, 2))
        for i in range(self.n_clusters):
            Mu[i] = np.average(X, axis=0, weights=W[:, i])
        return Mu

    # 更新Var（每一类的方差）
    def update_Var(self, X, Mu, W):
        Var = np.zeros((self.n_clusters, 2))
        for i in range(self.n_clusters):
            Var[i] = np.average((X - Mu[i]) ** 2, axis=0, weights=W[:, i])
        return Var

    # 计算log似然函数LikeliHood
    def logLH(self, X, Pi, Mu, Var):
        n_points = len(X)
        pdfs = np.zeros((n_points, self.n_clusters))
        for i in range(self.n_clusters):
            pdfs[:, i] = Pi[i] * multivariate_normal.pdf(X, Mu[i], np.diag(Var[i]))
        return np.mean(np.log(pdfs.sum(axis=1)))

    def plot_clusters(self, X, Mu, Var, Mu_true=None, Var_true=None, is_show=True):
        if not is_show:
            return

        colors = ['b', 'g', 'r']
        plt.figure(figsize=(10, 8))
        plt.axis([-10, 15, -5, 15])
        plt.scatter(X[:, 0], X[:, 1], s=5)
        ax = plt.gca()
        for i in range(self.n_clusters):
            plot_args = {'fc': 'None', 'lw': 2, 'edgecolor': colors[i], 'ls': ':'}
            ellipse = Ellipse(Mu[i], 3 * Var[i][0], 3 * Var[i][1], **plot_args)
            ax.add_patch(ellipse)
        if (Mu_true is not None) & (Var_true is not None):
            for i in range(self.n_clusters):
                plot_args = {'fc': 'None', 'lw': 2, 'edgecolor': colors[i], 'alpha': 0.5}
                ellipse = Ellipse(Mu_true[i], 3 * Var_true[i][0], 3 * Var_true[i][1], **plot_args)
                ax.add_patch(ellipse)
        plt.show()
    # 屏蔽结束
    
    def fit(self, data, Mu_true=None, Var_true=None):
        # 作业3
        # 屏蔽开始
        n_points, d_dims = len(data), np.shape(data)[1]
        center_random = 0
        if center_random:   # select centern points by random or designated
            mu_idx = random.sample(range(n_points), self.n_clusters)
            print('mu_idx', mu_idx)
            Mu = data[mu_idx, :]
        else:
            Mu = [[0, -1], [6, 0], [0, 9]]
        Var = [[1]*d_dims] * self.n_clusters
        W = np.ones((n_points, self.n_clusters)) / self.n_clusters
        Pi = W.sum(axis=0) / W.sum()


        loglh = []
        for i in range(self.max_iter):
            self.plot_clusters(data, Mu, Var, Mu_true=Mu_true, Var_true=Var_true, is_show=False)
            loglh.append(self.logLH(data, Pi, Mu, Var))
            # E-step
            W = self.update_W(data, Mu, Var, Pi)    # posterior probability
            # M-step
            Mu = self.update_Mu(data, W)            # mean
            Var = self.update_Var(data, Mu, W)
            Pi = self.update_Pi(W)

            print('log-likehood:%.3f' % loglh[-1])
            if len(loglh) >= 2 and math.fabs(loglh[-1] - loglh[-2]) < 1e-3:
                print('iter:{} break for the model fit well！'.format(i))
                break

        # save GMM model parameters
        self.W = copy.deepcopy(W)
        self.Mu = copy.deepcopy(Mu)
        self.Var = copy.deepcopy(Var)
        self.Pi = copy.deepcopy(Pi)
        # 屏蔽结束
    
    def predict(self, data):
        # 屏蔽开始
        n_points = len(data)
        pdfs = np.zeros((n_points, self.n_clusters))
        for i in range(self.n_clusters):
            # Multivariate normal probability density function
            pdfs[:, i] = self.Pi[i] * multivariate_normal.pdf(data, self.Mu[i], np.diag(self.Var[i]))
        W_predict = pdfs / pdfs.sum(axis=1).reshape(-1, 1)
        W_table = np.argsort(W_predict, axis=1)
        cluster_idx = W_table[:, -1]

        return cluster_idx
        # 屏蔽结束


# 生成仿真数据
def generate_X(num, true_Mu, true_Var):
    # 第一簇的数据
    num1, mu1, var1 = num[0], true_Mu[0], true_Var[0]
    X1 = np.random.multivariate_normal(mu1, np.diag(var1), num1)
    # 第二簇的数据
    num2, mu2, var2 = num[1], true_Mu[1], true_Var[1]
    X2 = np.random.multivariate_normal(mu2, np.diag(var2), num2)
    # 第三簇的数据
    num3, mu3, var3 = num[2], true_Mu[2], true_Var[2]
    X3 = np.random.multivariate_normal(mu3, np.diag(var3), num3)
    # 合并在一起
    X = np.vstack((X1, X2, X3))
    # 显示数据
    plt.figure(figsize=(10, 8))
    plt.axis([-10, 15, -5, 15])
    plt.scatter(X1[:, 0], X1[:, 1], s=5)
    plt.scatter(X2[:, 0], X2[:, 1], s=5)
    plt.scatter(X3[:, 0], X3[:, 1], s=5)
    plt.show()
    return X


def compare(n_clusters, data, cluster_true, cluster_predict):
    diff_data = data[cluster_true != cluster_predict]

    # draw points according to predict cluster
    colors = ['red', 'green', 'blue', 'black']
    plt.figure(figsize=(10, 8))
    plt.axis([-10, 15, -5, 15])
    for i in range(n_clusters):
        x = data[cluster_predict == i]
        plt.scatter(x[:, 0], x[:, 1], c=colors[i], s=5)

    # draw points which predict result is diffenent from true cluster
    plt.scatter(diff_data[:, 0], diff_data[:, 1], c=colors[3], s=8)
    print('wrong data\'s ratio: {}/{}'.format(len(diff_data), len(data)))
    plt.show()


if __name__ == '__main__':
    # 生成数据
    num = [400, 600, 1000]
    true_Mu = [[0.5, 0.5], [5.5, 2.5], [1, 7]]
    true_Var = [[1, 3], [2, 2], [6, 2]]
    X = generate_X(num, true_Mu, true_Var)

    n_clusters = 3
    gmm = GMM(n_clusters=n_clusters)
    gmm.fit(X, true_Mu, true_Var)
    cluster_predict = gmm.predict(X)

    cluster_true = None
    for i in range(n_clusters):
        cluster_i = np.ones((num[i], 1), dtype=np.int)*i
        cluster_true = cluster_i if cluster_true is None else np.vstack((cluster_true, cluster_i))

    cluster_true = cluster_true.reshape(cluster_predict.shape)
    compare(n_clusters, X, cluster_true, cluster_predict)


    

