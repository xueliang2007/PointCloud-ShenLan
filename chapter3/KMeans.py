# 文件功能： 实现 K-Means 算法
# xueliang(xueliang2007@qq.com)

import random
import numpy as np
from itertools import cycle, islice
from matplotlib import pyplot as plt


class K_Means(object):
    # k是分组数；tolerance‘中心点误差’；max_iter是迭代次数
    def __init__(self, n_clusters=2, tolerance=0.0001, max_iter=300):
        self.k_ = n_clusters
        self.tolerance_ = tolerance
        self.max_iter_ = max_iter
        self.centers_ = None

    def fit(self, data):
        # 作业1
        # 屏蔽开始
        is_plt_show = 0
        n = data.shape[0]
        centers_index = random.sample([i for i in range(n)], self.k_)
        self.centers_ = data[centers_index]
        centers_last = self.centers_.copy()

        for i in range(self.max_iter_):
            if is_plt_show:
                color = ['red', 'blue', 'green', 'black', 'purple']
                marker = ['x', 'o', 'v', 's', 'd', '+', 'p', '8']
                plt.figure()
                if i == 0:  # draw all the data points at the begin time
                    plt.scatter(data[:, 0], data[:, 1], c=color[-1], marker=marker[-1])
                    for ki in range(self.k_):
                        plt.scatter(self.centers_[ki][0], self.centers_[ki][1], c=color[ki], marker=marker[ki])
                    plt.show()

            dist_matrix = None
            points_k = [None for i in range(self.k_)]
            # calculate the distance between every point and centers
            for ki in range(self.k_):
                dist_ki = np.linalg.norm(np.expand_dims(self.centers_[ki], axis=0) - data, axis=1).reshape((n, -1))
                dist_matrix = dist_ki if dist_matrix is None else np.hstack((dist_matrix, dist_ki))
            class_idx = np.argsort(dist_matrix, axis=1)  # ensure the class id of every point, 每一行第一列的值即为类别索引

            for kj in range(n):  # part into k class
                points_k[class_idx[kj, 0]] = data[kj, :] if points_k[class_idx[kj, 0]] is None \
                                                         else np.vstack((points_k[class_idx[kj, 0]], data[kj, :]))
            for ki in range(self.k_):  # update centers
                self.centers_[ki] = np.mean(points_k[ki], axis=0)

            # draw the process of K-Means
            if is_plt_show:
                plt.figure()
                for ki in range(self.k_):
                    plt.scatter(points_k[ki][:, 0], points_k[ki][:, 1], c=color[ki], marker=marker[ki])
                    plt.scatter(self.centers_[ki][0], self.centers_[ki][1], c=color[ki], marker=marker[ki])
                plt.show()

            # determine if the center is changing
            center_diff = centers_last - self.centers_
            center_diff = np.linalg.norm(center_diff, axis=1).sum()
            if center_diff < self.tolerance_:
                # print('iter: ', i+1)
                break
            centers_last = self.centers_.copy()  # deep clone!
        # 屏蔽结束

    def predict(self, p_datas):
        result = []
        # 作业2
        # 屏蔽开始
        pass
        n = p_datas.shape[0]
        dist_matrix = None
        for ki in range(self.k_):   # calculate the distance between every query point and centers
            dist_ki = np.linalg.norm(np.expand_dims(self.centers_[ki], axis=0) - p_datas, axis=1).reshape((n, -1))
            dist_matrix = dist_ki if dist_matrix is None else np.hstack((dist_matrix, dist_ki))
        class_idx = np.argsort(dist_matrix, axis=1)  # ensure the class id of every point, 每一行第一列的值即为类别索引
        result = class_idx[:, 0]
        # 屏蔽结束
        return result


if __name__ == '__main__':
    x = np.array([[1, 2], [1.5, 1.8], [5, 8], [2, 2], [8, 8], [8, 10], [6, 9], [1, 0.6], [9, 11], [3, 2],
                  [4, 4], [5, 3.4], [4, 2], [7, 6], [6, 7], [3, 4], [5, 1], [2, 4], [8.5, 7], [9.5, 7]])
    k_means = K_Means(n_clusters=2)
    k_means.fit(x)

    query = np.array([[3, 1], [4, 3], [7, 9], [8, 6]])
    cat = k_means.predict(query)
    print(cat)

