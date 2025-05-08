#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
场景重建器：使用SfM和MVS技术重建3D场景
"""

import numpy as np
import cv2
import open3d as o3d
from scipy.optimize import least_squares

class SceneReconstructor:
    """3D场景重建器，使用SfM和MVS技术重建3D场景"""
    
    def __init__(self):
        """初始化重建器"""
        self.camera_matrices = []  # 相机内参矩阵列表
        self.pose_matrices = []    # 相机姿态矩阵列表
        self.point_cloud = None    # 重建的点云
        self.mesh = None           # 重建的3D网格
        
    def estimate_camera_poses(self, keypoints_list, descriptors_list):
        """
        估计相机姿态
        
        Args:
            keypoints_list: 特征点列表
            descriptors_list: 描述符列表
            
        Returns:
            相机姿态矩阵列表
        """
        n_images = len(keypoints_list)
        matcher = cv2.BFMatcher()
        camera_poses = [np.eye(4)]  # 第一个相机为参考位置
        
        # 相机内参矩阵估计 (可通过相机标定获取更精确的参数)
        focal_length = 1000  # 假设的焦距
        principal_point = (640, 480)  # 假设的主点
        K = np.array([[focal_length, 0, principal_point[0]],
                      [0, focal_length, principal_point[1]],
                      [0, 0, 1]])
        self.camera_matrices = [K] * n_images
        
        # 逐对图像估计相对姿态
        for i in range(1, n_images):
            matches = matcher.knnMatch(descriptors_list[i-1], descriptors_list[i], k=2)
            
            # 应用Lowe's比率测试
            good_matches = []
            for m, n in matches:
                if m.distance < 0.7 * n.distance:
                    good_matches.append(m)
            
            # 提取匹配点坐标
            pts1 = np.float32([keypoints_list[i-1][m.queryIdx].pt for m in good_matches])
            pts2 = np.float32([keypoints_list[i][m.trainIdx].pt for m in good_matches])
            
            # 计算基础矩阵
            F, mask = cv2.findFundamentalMatrix(pts1, pts2, cv2.FM_RANSAC, 3.0)
            
            # 计算本质矩阵
            E = K.T @ F @ K
            
            # 从本质矩阵恢复相机姿态
            _, R, t, _ = cv2.recoverPose(E, pts1[mask.ravel() == 1], pts2[mask.ravel() == 1], K)
            
            # 构建4x4姿态矩阵
            pose = np.eye(4)
            pose[:3, :3] = R
            pose[:3, 3] = t.ravel()
            
            # 计算全局姿态 (相对于第一帧)
            global_pose = camera_poses[i-1] @ pose
            camera_poses.append(global_pose)
        
        self.pose_matrices = camera_poses
        return camera_poses
    
    def triangulate_points(self, keypoints_list, matches_list):
        """
        三角测量计算3D点坐标
        
        Args:
            keypoints_list: 特征点列表
            matches_list: 图像间的匹配列表
            
        Returns:
            3D点云
        """
        # 初始化点云
        points_3d = []
        point_colors = []
        
        # 对每对相邻图像进行三角测量
        for i in range(len(matches_list)):
            # 提取匹配点
            pts1 = np.float32([keypoints_list[i][m.queryIdx].pt for m in matches_list[i]])
            pts2 = np.float32([keypoints_list[i+1][m.trainIdx].pt for m in matches_list[i]])
            
            # 构建投影矩阵
            P1 = self.camera_matrices[i] @ np.hstack([np.eye(3), np.zeros((3, 1))])
            P2 = self.camera_matrices[i+1] @ np.hstack([self.pose_matrices[i+1][:3, :3], 
                                                      self.pose_matrices[i+1][:3, 3:4]])
            
            # 三角测量
            points_4d = cv2.triangulatePoints(P1, P2, pts1.T, pts2.T)
            
            # 转换为齐次坐标
            points_3d_homogeneous = points_4d / points_4d[3]
            points_3d.extend(points_3d_homogeneous[:3].T)
            
            # 添加颜色信息 (从第一张图获取)
            # 这里可以扩展为从多视角获取颜色并混合
            
        # 创建Open3D点云对象
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(np.array(points_3d))
        
        # 点云滤波和降噪
        pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
        
        self.point_cloud = pcd
        return pcd
    
    def generate_mesh(self, pcd):
        """
        从点云生成3D网格
        
        Args:
            pcd: 输入点云
            
        Returns:
            3D网格
        """
        # 计算法向量
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=0.1, max_nn=30))
        
        # 使用泊松表面重建
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd, depth=9, width=0, scale=1.1, linear_fit=False)[0]
        
        # 网格简化
        mesh = mesh.simplify_quadric_decimation(target_number_of_triangles=100000)
        
        # 网格平滑
        mesh = mesh.filter_smooth_taubin(number_of_iterations=10)
        
        self.mesh = mesh
        return mesh
    
    def save_reconstruction(self, pcd_file="scene_pointcloud.ply", mesh_file="scene_mesh.obj"):
        """
        保存重建结果
        
        Args:
            pcd_file: 点云文件保存路径
            mesh_file: 网格文件保存路径
        """
        if self.point_cloud is not None:
            o3d.io.write_point_cloud(pcd_file, self.point_cloud)
            print(f"点云已保存至 {pcd_file}")
            
        if self.mesh is not None:
            o3d.io.write_triangle_mesh(mesh_file, self.mesh)
            print(f"3D网格已保存至 {mesh_file}")
            
    def visualize_reconstruction(self):
        """
        可视化重建结果
        """
        vis = o3d.visualization.Visualizer()
        vis.create_window()
        
        if self.point_cloud is not None:
            vis.add_geometry(self.point_cloud)
        
        if self.mesh is not None:
            vis.add_geometry(self.mesh)
            
        # 添加坐标系
        coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)
        vis.add_geometry(coord_frame)
        
        # 设置视图
        vis.get_render_option().point_size = 2.0
        vis.get_render_option().background_color = np.array([0.1, 0.1, 0.1])
        
        # 运行可视化
        vis.run()
        vis.destroy_window()


if __name__ == "__main__":
    print("场景重建模块 - 请从main.py运行完整流程")
    print("此模块需要配合图像预处理模块使用") 