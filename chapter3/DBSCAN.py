# 文件功能：实现 DBSCAN 算法

import numpy as np
import pylab
import random
import math
import copy
from sklearn import datasets
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from scipy.spatial import KDTree
from scipy.stats import multivariate_normal
plt.style.use('seaborn')


class DBSCAN(object):
    def __init__(self, r=0.15, min_samples=5):
        self.r = r
        self.min_samples = min_samples
        self.n_cluster = 0

    def fit(self, X):
        N = len(X)
        tags = np.array([0] * N)  # 0-NoVisited, 1-CorePoint, 2-BorderPoint, 3-outlier
        kdtree = KDTree(X, leafsize=8)

        labels = [-1] * N   # element: -1: outliers
        for i in range(N):
            if tags[i]:  # continue if not visited
                continue

            # get current core point's neighbor
            neighbor_ids = kdtree.query_ball_point(X[i, :], r=self.r)
            if len(neighbor_ids) < self.min_samples:
                tags[i] = 3  # outliers
                continue

            tags[i] = 1  # core point
            labels[i] = self.n_cluster  # current class id
            neighbor_points = X[neighbor_ids]
            while True:
                # update core point and refind its neighbor as the same class, and repeat
                neighbor_ids_update = set()
                for ii, (index, pp) in enumerate(zip(neighbor_ids, neighbor_points)):
                    if tags[index]:
                        continue
                    ng_ids = kdtree.query_ball_point(pp, r=self.r)
                    if len(ng_ids) < self.min_samples:
                        tags[index] = 2    # border point
                        continue
                    tags[index] = 1
                    labels[index] = self.n_cluster
                    for id_ in ng_ids:
                        if not tags[id_]:   # No Visited
                            neighbor_ids_update.add(id_)
                # 从neighbor中剔除tag不为0的元素
                neighbor_ids = np.array([i for i in neighbor_ids_update if tags[i] == 0])
                if len(neighbor_ids) == 0:
                    break
                neighbor_points = X[neighbor_ids]
            self.n_cluster += 1  # next class

            isshow = 1
            if isshow:
                colors = ['#377eb8', '#ff7f00', '#4daf4a',
                          '#999999', '#e41a1c', '#dede00',
                          '#f781bf', '#a65628', '#984ea3',
                          '#ffa445', '#ee1122', '#dd5652']
                plt.figure()
                for itt in range(self.n_cluster):
                    lla = np.array([i_ for i_ in range(N) if labels[i_] == itt])
                    if len(lla) == 0:
                        continue
                    data = X[lla, :]
                    plt.scatter(data[:, 0], data[:, 1], c=colors[itt], s=5)
                if len(lla) == 0:
                    continue
                data = X[lla, :]
                lla = np.array([i_ for i_ in range(N) if labels[i_] == -1])
                plt.scatter(data[:, 0], data[:, 1], c=colors[self.n_cluster], s=5)
                plt.show()

        print('class: ', self.n_cluster)
        return self.n_cluster


if __name__ == '__main__':
    X, label_gt = datasets.make_circles(n_samples=500, factor=0.5, noise=0.05)

    dbscan = DBSCAN()
    dbscan.fit(X)

    

