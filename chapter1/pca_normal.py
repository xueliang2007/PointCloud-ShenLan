# 实现PCA分析和法向量计算，并加载数据集中的文件进行验证

import open3d as o3d 
import os
import numpy as np
from pyntcloud import PyntCloud

# 功能：计算PCA的函数
# 输入：
#     data：点云，NX3的矩阵
#     correlation：区分np的cov和corrcoef，不输入时默认为False
#     sort: 特征值排序，排序是为了其他功能方便使用，不输入时默认为True
# 输出：
#     eigenvalues：特征值
#     eigenvectors：特征向量
def PCA(data, correlation=False, sort=True):
    # 作业1
    # 屏蔽开始
    data_np = np.array(data)
    x_bar = np.mean(data_np, axis=0)
    # print('x_bar: {}'.format(x_bar))
    X_tilde = data_np - x_bar
    H = np.dot(X_tilde.T, X_tilde)
    eigenvectors, eigenvalues, _ = np.linalg.svd(H)
    eigenvalues = np.sqrt(eigenvalues)
    # 屏蔽结束
    if sort:
        sort = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[sort]
        eigenvectors = eigenvectors[:, sort]

    return eigenvalues, eigenvectors


def main():
    # 指定点云路径
    cat_index = 0  # 物体编号，范围是0-39，即对应数据集中40个物体
    root_dir = '/home/snow/ShenLan/Dtasets/ModelNet40-ply'  # 数据集路径
    cat = os.listdir(root_dir)
    file_name = os.path.join(root_dir, cat[cat_index], 'train', cat[cat_index]+'_0002.ply')  # 默认使用第一个点云

    # 加载原始点云
    point_cloud_pynt = PyntCloud.from_file(file_name)
    point_cloud_o3d = point_cloud_pynt.to_instance("open3d", mesh=False)
    # o3d.visualization.draw_geometries([point_cloud_o3d])  # 显示原始点云

    # 从点云中获取点，只对点进行处理
    points = point_cloud_pynt.points
    print('total points number is:', points.shape)

    # 用PCA分析点云主方向
    w, v = PCA(points, sort=False)
    point_cloud_vector = v[:, 2]  #点云主方向对应的向量
    print('the main orientation of this pointcloud is: ', point_cloud_vector)
    # TODO: 此处只显示了点云，还没有显示PCA
    # 计算质心并绘制主方向
    mesh_frame = o3d.geometry.create_mesh_coordinate_frame(size=2, origin=[0, 0, 0])
    center = np.mean(np.array(points), axis=0)
    points_xl = np.vstack((center, center + point_cloud_vector))
    colors = [[0, 0, 1] for i in range(len(points_xl))]
    lines = [[0, 1]]

    line_pcd = o3d.LineSet()
    line_pcd.lines = o3d.utility.Vector2iVector(lines)
    line_pcd.colors = o3d.utility.Vector3dVector(colors)
    line_pcd.points = o3d.utility.Vector3dVector(points_xl)
    o3d.visualization.draw_geometries([point_cloud_o3d, line_pcd, mesh_frame])
    frame = o3d.geometry.create_mesh_coordinate_frame()

    
    # # 循环计算每个点的法向量
    pcd_tree = o3d.geometry.KDTreeFlann(point_cloud_o3d)
    normals = []
    # 作业2
    # 屏蔽开始
    points_np = np.array(points, dtype=np.float64)
    for i, xi in enumerate(points_np):
        nn, ids, _ = pcd_tree.search_knn_vector_3d(xi.reshape((3, 1)), knn=20)  # 1.negihbors
        xi_neighbors = points_np[ids]
        xw, xv = PCA(xi_neighbors, sort=False)   # 2. PCA
        normals.append(xv[:, 2])    # 3. normal
    # 由于最近邻搜索是第二章的内容，所以此处允许直接调用open3d中的函数
    # 屏蔽结束
    normals = np.array(normals, dtype=np.float64)
    # # TODO: 此处把法向量存放在了normals中
    point_cloud_o3d.normals = o3d.utility.Vector3dVector(normals)
    o3d.visualization.draw_geometries([point_cloud_o3d])


if __name__ == '__main__':
    main()
