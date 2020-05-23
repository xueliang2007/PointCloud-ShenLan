# 实现voxel滤波，并加载数据集中的文件进行验证
import random
import open3d as o3d 
import os
import numpy as np
from pyntcloud import PyntCloud

# 功能：对点云进行voxel滤波
# 输入：
#     point_cloud：输入点云
#     leaf_size: voxel尺寸
def voxel_filter(point_cloud, leaf_size):
    filtered_points = []
    # 作业3
    # 屏蔽开始
    X = np.array(point_cloud)
    # max - min
    xyz_maxbias = np.ptp(X, axis=0)
    #
    dxyz = xyz_maxbias/leaf_size
    Xmin = np.amin(X, axis=0)
    Hxyz = np.floor((X - Xmin)/dxyz)
    H = np.dot(Hxyz, dxyz)

    sort = H.argsort()
    H = H[sort]
    X_inorder = X[sort, :]

    # sampling
    islow = 0
    for ifast in range(1, H.shape[0]):
        if H[ifast] == H[islow]:
            continue
        else:
            if 1:
                point_filtered = np.mean(X_inorder[islow:ifast, :], axis=0)  # average sampling
            else:
                point_filtered = X_inorder[random.sample(range(islow, ifast), 1), :][0]  # random sampling
            filtered_points.append(point_filtered)
            islow = ifast
    print('data size:{}\tafter voxel size:{}'.format(np.shape(X)[0], len(filtered_points)))
    # 屏蔽结束

    # 把点云格式改成array，并对外返回
    filtered_points = np.array(filtered_points, dtype=np.float64)
    return filtered_points


def main():
    # # 从ModelNet数据集文件夹中自动索引路径，加载点云
    cat_index = 10  # 物体编号，范围是0-39，即对应数据集中40个物体
    root_dir = '/home/snow/ShenLan/Dtasets/ModelNet40-ply'  # 数据集路径
    cat = os.listdir(root_dir)
    file_name = os.path.join(root_dir, cat[cat_index], 'train', cat[cat_index]+'_0001.ply')  # 默认使用第一个点云
    point_cloud_pynt = PyntCloud.from_file(file_name)

    # 转成open3d能识别的格式
    point_cloud_o3d = point_cloud_pynt.to_instance("open3d", mesh=False)
    # o3d.visualization.draw_geometries([point_cloud_o3d])  # 显示原始点云

    # 调用voxel滤波函数，实现滤波
    filtered_cloud = voxel_filter(point_cloud_pynt.points, 50.0)
    point_cloud_o3d.points = o3d.utility.Vector3dVector(filtered_cloud)
    # 显示滤波后的点云
    o3d.visualization.draw_geometries([point_cloud_o3d])


if __name__ == '__main__':
    main()
