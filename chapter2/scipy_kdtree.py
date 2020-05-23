# 使用 scipy.spatial.KDTree 查询最近邻的点云, 同时记录时间
import time
import numpy as np
from scipy.spatial import KDTree


def main():
    db_np = np.random.rand(2000, 3)

    begin_t = time.time()
    kdt = KDTree(db_np, leafsize=10)    # 构建 KDTree

    x = np.array([0, 0, 0])             # 查询点query
    revals, idxs = kdt.query(x, k=3)    # 查询最近的3个邻居点， 返回距离和最近邻点的索引
    kd_time = (time.time() - begin_t) * 1000
    print('knn_time: {}ms'.format(kd_time))


if __name__ == '__main__':
    main()

