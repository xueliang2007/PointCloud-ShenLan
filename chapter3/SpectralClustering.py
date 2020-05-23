# 文件功能：实现 Spectral Clustering 算法
import numpy as np
from matplotlib import pyplot as plt
from sklearn import datasets
from sklearn.cluster import KMeans
from KMeans import K_Means
np.random.seed(10)
plt.style.use('seaborn')


class SP_Cultering(object):
    def __init__(self, n_clusters=2):
        self.n_clusters = n_clusters

    def calculateDistanceMatrix(self, X):
        """
        calculate Distance Matrix
        :param X:
        :return:
        """
        N = len(X)
        W = np.zeros((N, N))
        for i in range(N):
            x = X[i+1:, :] - X[i, :]
            dist = np.sum(x**2, axis=1)
            W[i+1:, i] = dist
        W = W + W.T
        return W

    def distTransToWeightKNN(self, W, k, sigma=1.0):
        """
        turn distance matrix(W, wij) to weight matrix(Adjacent, aij): aij = np.exp(-wij/2./sigma/sigma)
        :param W: distance matrix
        :param k: knn-k
        :param sigma:
        :return:
        """
        N = len(W)
        denominator = 1. / 2 / sigma / sigma
        Adjacent = np.zeros((N, N))
        for i in range(N):
            dist = W[:, i]
            idx_sorted = np.argsort(dist)  # part
            Adjacent[idx_sorted[:k + 1], i] = np.exp(-W[i, idx_sorted[:k + 1]] * denominator)
            Adjacent[i, idx_sorted[:k + 1]] = Adjacent[idx_sorted[:k + 1], i].T
        return Adjacent

    def calculateLaplacianMatrix(self, Adjacent, normalized='sym'):
        """
        calculate laplacian matrix
        :param Adjacent:
        :param normalized: 'sym' or 'rm'
        :return:
        """
        Degree = np.sum(Adjacent, axis=1)
        L = np.diag(Degree) - Adjacent
        if normalized == 'sym':
            # L = D^(-0.5)*L*D^(-0.5)
            DegreeSqrt = np.diag(1. / (Degree**0.5))
            L = np.dot(np.dot(DegreeSqrt, L), DegreeSqrt)
        elif normalized == 'rm':
            # L = D^(-1)*L
            L = np.dot(np.diag(1./Degree), L)
        return L

    def calculateYMatrix(self, Laplacian, eigValueGap=False, bShowGap=False):
        r, V = np.linalg.eig(Laplacian)  # OK
        idx_sorted = np.argsort(r)

        if eigValueGap:
            rn = min(self.n_clusters * 3, len(Laplacian))
            rr = r[idx_sorted][:rn]

            gap = [np.fabs(rr[k-1] - rr[k]) for k in range(1, rn)]
            self.n_clusters = gap.index(max(gap)) + 1

            Y = V[:, idx_sorted][:, :self.n_clusters]
        else:
            Y = V[:, idx_sorted][:, :self.n_clusters]

        if bShowGap:
            rn = min(self.n_clusters * 3, len(Laplacian))
            rr = r[idx_sorted][:rn]
            plt.figure()
            plt.scatter(range(rn), rr, s=5, marker='+')
            plt.show()

        return Y

    def fit(self, X, eigValueGap=False, bShowGap=False):
        W = self.calculateDistanceMatrix(X)
        Adjacent = self.distTransToWeightKNN(W, k=10)
        Laplacian = self.calculateLaplacianMatrix(Adjacent, normalized='rm')
        Y = self.calculateYMatrix(Laplacian, eigValueGap=eigValueGap, bShowGap=bShowGap)

        MY_KNN = 1
        if MY_KNN:
            knn = K_Means(self.n_clusters)
            knn.fit(Y)
            labels = knn.predict(Y)
        else:
            labels = KMeans(n_clusters=self.n_clusters).fit(Y).labels_
        return labels

    def plot(self, data, labels_my):
        # draw points according to predict cluster
        colors = ['red', 'green', 'blue', 'black']
        plt.figure(figsize=(10, 8))
        for i in range(self.n_clusters):
            x = data[labels_my == i]
            plt.scatter(x[:, 0], x[:, 1], c=colors[i], s=5)
        plt.show()


if __name__ == '__main__':
    data, label_gt = datasets.make_circles(n_samples=500, factor=0.5, noise=0.05)

    spc = SP_Cultering(n_clusters=2)
    label_my = spc.fit(data, eigValueGap=False, bShowGap=False)
    spc.plot(data, label_my)

