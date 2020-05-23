# 文件功能：实现 Mean-Shift 算法
import copy
import numpy as np
from matplotlib import pyplot as plt
from sklearn import datasets
from scipy.spatial import KDTree
from matplotlib.patches import Circle
np.random.seed(1)
plt.style.use('seaborn')


class Mean_Shift(object):
    def __init__(self, r=1.5, max_iter=50, nms=0.01, tolerance=1e-6, lessK=5):
        """
        initialization
        :param r: radius
        :param max_iter:
        :param nms: NMS
        :param tolerance: determine the centers changing or not
        """
        self.r = r
        self.max_iter = max_iter
        self.nms = nms
        self.tolerance = tolerance
        self.lessK = lessK
        self.n_clusters = -1
        self.centers = None

    def genMeshGridPoints(self, X, grid):
        """
        generate the grid point to cover all the data points
        :param X: data points
        :param grid: size of grid
        :return:
        """
        ptp_x, ptp_y = np.ptp(X[:, 0], axis=0), np.ptp(X[:, 1], axis=0)
        minx, miny = np.amin(X[:, 0]), np.amin(X[:, 1])
        dx, dy = int(ptp_x / grid), int(ptp_y / grid)
        mx = [minx + i * grid for i in range(dx)]
        my = [miny + i * grid for i in range(dy)]

        grid_points = [None] * dx * dy
        for i, x_ in enumerate(mx):
            for j, y_ in enumerate(my):
                # print('id:{}, {}'.format(i * dy + j, [x_, y_]))
                grid_points[i * dy + j] = [x_, y_]
        return np.array(grid_points)

    def fit(self, X, grid=1.):
        """
        fit the Means-Shift Model
        :param X:
        :param grid:
        :return:
        """
        centers = self.genMeshGridPoints(X, grid)
        nc = len(centers)

        kdtree = KDTree(X, leafsize=8)
        centers_update = copy.deepcopy(centers)
        for i in range(self.max_iter):
            print('iter: ', i)
            delete_idx = []
            rnn_nums = [0] * nc
            for j in range(nc):
                query = centers[j, :]
                neighbot_index = kdtree.query_ball_point(query, r=self.r)
                rnn_nums[j] = len(neighbot_index)
                if len(neighbot_index) > self.lessK:
                    # if the current center has neighbors, then recompute the center
                    center = np.mean(X[neighbot_index, :], axis=0)
                    centers[j, :] = center
                else:
                    # if the current center has no neighbors, record the index and prepare to delete it
                    delete_idx.append(j)

            # delete some centers who has no neighbors
            if delete_idx:
                centers = np.delete(centers, delete_idx, axis=0)
                centers_update = np.delete(centers_update, delete_idx, axis=0)
                rnn_nums = np.delete(np.array(rnn_nums), delete_idx, axis=0)
                nc = len(centers)

            # NMS, Non-Maximum Suppression
            # 如果两个中心点坐标之差很小，则剔除radiusNN点较少的中心点
            delete_idx = set()
            for jj in range(nc):
                diff_c = centers - centers[jj, :]
                diff = np.sum(diff_c ** 2, axis=1)
                diff[jj] = 1e3  # ensure the current center will not be delete
                nms_idx = np.array(range(nc))[diff < self.nms]  # get the point indexs that around the current center
                for k in nms_idx:
                    if rnn_nums[k] < rnn_nums[jj]:  # record and delete the index whose neighbor'size is less
                        delete_idx.add(k)
            if delete_idx:
                delete_idx = list(delete_idx)
                centers = np.delete(centers, delete_idx, axis=0)
                centers_update = np.delete(centers_update, delete_idx, axis=0)
                nc = len(centers)

            show = 1
            if show:
                plt.figure()
                plt.scatter(datas[:, 0], datas[:, 1], s=5, color='r')
                plt.scatter(centers[:, 0], centers[:, 1], c='blue', s=8)
                ax = plt.gca()
                plot_args = {'fc': 'None', 'lw': 2, 'edgecolor': 'b', 'ls': ':'}
                for j in range(nc):
                    circle = Circle(centers[j, :], radius=self.r, **plot_args)
                    ax.add_patch(circle)
                plt.show()

            # determine whether the iteration terminates or not
            diff_sum = np.sum(np.fabs(centers_update - centers))
            if diff_sum < self.tolerance:
                print('centers do not change, break at iter:{}!'.format(i))
                break
            else:
                centers_update = copy.deepcopy(centers)

        # TODO: need NMS again

        self.n_clusters, self.centers = len(centers), copy.deepcopy(centers)

    def predict(self, data):
        dist_matrix = np.zeros((len(data), self.n_clusters))
        for i in range(self.n_clusters):
            dist = data - self.centers[i, :]
            dist_matrix[:, i] = np.sum(dist**2, axis=1)
        idx_sorted = np.argsort(dist_matrix, axis=1)
        labels = idx_sorted[:, 0]

        show = 1
        if show:
            plt.figure()
            colors = ['red', 'green', 'blue', 'black',
                      '#377eb8', '#ff7f00', '#4daf4a',
                      '#999999', '#e41a1c', '#dede00',
                      '#f781bf', '#a65628', '#984ea3']
            for j in range(self.n_clusters):
                data = datas[labels == j]
                plt.scatter(data[:, 0], data[:, 1], s=5, color=colors[j])
            plt.show()
        return labels


if __name__ == '__main__':
    datas, labels = datasets.make_blobs(n_samples=800, centers=4, center_box=(-5, 5), cluster_std=[1.0, 0.75, 0.5, 0.1])
    plt.scatter(datas[:, 0], datas[:, 1], s=5, color='r')
    plt.show()

    ms = Mean_Shift()
    ms.fit(datas)

    labels_my = ms.predict(datas)
