# use numpy to implement KNN and RadiusNN
import time
import numpy as np
from benchmark import read_velodyne_bin as read_bin


def get_euclidean_dist(p1, p2):
    """
    return the euclidean distance between p1 and p2
    :param p1: np.array(1*3)
    :param p2: np.array(1*3)
    :return:
    """
    p = p1 - p2
    dist = np.dot(p, p.T)
    return np.sqrt(dist)


def knn_sorted(dist_vector, k, points):
    """
    get the k minest distance form 'dist_vector', and return their index in 'points'
    :param dist_vector:
    :param k:
    :param points:
    :return:
    """
    dist_idx = np.asarray([i for i in range(dist_vector.shape[1])])
    sorted_index = np.argsort(dist_vector[0])
    dist_idx = dist_idx[sorted_index]
    dist_idx_k = dist_idx[0:k]
    return points[dist_idx_k, :], dist_vector[0, dist_idx_k]


def knn_neighbor(db_np, k):
    """
    get k nearest neighbor for every point in 'db_np'
    :param db_np: np.array(N*3)
    :param k:
    :return:
    """
    n = db_np.shape[0]
    knn_dist = np.zeros((n, k), dtype=np.float)
    knn_points = np.zeros((n, k, 3), dtype=np.float)
    for i in range(n):
        query = db_np[i, :]
        dp_array = np.zeros((1, n), dtype=np.float)
        for j in range(n):
            if i == j:
                dp_array[0, j] = 1e4
            else:
                p2 = db_np[j, :]
                dp_array[0, j] = get_euclidean_dist(query, p2)

        point_part, dist_part = knn_sorted(dp_array, k, db_np)
        knn_points[i, :, :] = point_part
        knn_dist[i, :] = dist_part
    return knn_points, knn_dist


def rnn_sorted(dist_vector, r, points):
    r_matrix = np.asarray([r for i in range(dist_vector.shape[1])])
    r_matrix = r_matrix.reshape((1, dist_vector.shape[1]))
    num = np.sum((dist_vector < r_matrix).astype(np.int))

    dist_idx = np.asarray([i for i in range(dist_vector.shape[1])])
    sorted_index = np.argsort(dist_vector[0])
    dist_idx = dist_idx[sorted_index]
    dist_idx_k = dist_idx[0:num]
    return points[dist_idx_k, :], dist_vector[0, dist_idx_k]


def radiusnn_neighbor(db_np, r=0.5):
    """
    get all nearest neighbors within distance of 'r' for every point in 'db_np'
    :param db_np: np.array(N*3)
    :param r: radius
    :return:
    """
    n = db_np.shape[0]
    rnn_dist, rnn_points = [], []
    for i in range(n):
        query = db_np[i, :]
        dp_array = np.zeros((1, n), dtype=np.float)
        for j in range(n):
            if i == j:
                dp_array[0, j] = 1e4
            else:
                p2 = db_np[j, :]
                dp_array[0, j] = get_euclidean_dist(query, p2)

        point_part, dist_part = rnn_sorted(dp_array, r, db_np)
        rnn_points.append(point_part)
        rnn_dist.append(dist_part)
    return rnn_points, rnn_dist


def main():
    # Attention: it'll cost lots of time when size of point_datas with BF method

    # generate point_datas from bin_file or by random
    bin_or_random_data = 0
    if bin_or_random_data:
        bin_path = '/home/snow/ShenLan/PointCloud/chapter2/lesson2code/000000.bin'
        db_np = read_bin(bin_path)
        db_np = db_np.T
    else:
        db_np = np.random.rand(100, 3)

    # KNN
    bf_time = time.time()
    knn_points, knn_dist = knn_neighbor(db_np, k=3)
    knn_time = (time.time() - bf_time)*1000
    print('knn_time: {}ms'.format(knn_time))

    # RadioNN
    bf_time = time.time()
    rnn_points, rnn_dist = radiusnn_neighbor(db_np, r=0.5)
    rnn_time = (time.time() - bf_time)*1000
    print('rnn_time: {}ms'.format(rnn_time))


if __name__ == '__main__':
    main()
